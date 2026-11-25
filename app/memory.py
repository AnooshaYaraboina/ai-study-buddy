"""Memory.

Short-term (per session): the last 6 turns verbatim + a running summary of
everything older (updated when turns fall out of the window).
Long-term (per student, across sessions): weak topics in SQLite, updated by
the grader and used by the quiz-maker to pick the next quiz.
"""

from __future__ import annotations

import time

from app import db
from app.llm import get_llm
from app.prompts import SUMMARY_SYSTEM

SHORT_TERM_TURNS = 6  # user+assistant exchanges kept verbatim


# ------------------------------------------------------------------ short-term
def load(student_id: str, session_id: str) -> dict:
    with db.connect() as con:
        rows = con.execute(
            "SELECT role, content, standalone FROM turns WHERE student_id=? AND session_id=? "
            "ORDER BY id DESC LIMIT ?", (student_id, session_id, SHORT_TERM_TURNS * 2)).fetchall()
        srow = con.execute("SELECT summary FROM summaries WHERE student_id=? AND session_id=?",
                           (student_id, session_id)).fetchone()
    return {
        "history": [dict(r) for r in reversed(rows)],
        "summary": srow["summary"] if srow else "",
        "weak_topics": weak_topics(student_id),
    }


def _offline_summary(old: str, turns: list[dict]) -> str:
    asked = [t.get("standalone") or t["content"] for t in turns if t["role"] == "user"]
    scores = [t["content"].split("\n")[0] for t in turns
              if t["role"] == "assistant" and t["content"].startswith("**Score")]
    parts = [old] if old else []
    if asked:
        parts.append("Asked about: " + "; ".join(a[:60] for a in asked))
    if scores:
        parts.append("Quiz results: " + ", ".join(s.strip("*") for s in scores))
    text = " | ".join(parts)
    return text[-600:]


def _update_summary(student_id: str, session_id: str) -> None:
    """Fold turns that left the short-term window into the running summary."""
    with db.connect() as con:
        ids = [r["id"] for r in con.execute(
            "SELECT id FROM turns WHERE student_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
            (student_id, session_id, SHORT_TERM_TURNS * 2)).fetchall()]
        if not ids:
            return
        window_start = min(ids)
        srow = con.execute("SELECT summary, upto_turn FROM summaries WHERE student_id=? AND "
                           "session_id=?", (student_id, session_id)).fetchone()
        old, upto = (srow["summary"], srow["upto_turn"]) if srow else ("", 0)
        evicted = [dict(r) for r in con.execute(
            "SELECT id, role, content, standalone FROM turns WHERE student_id=? AND session_id=? "
            "AND id > ? AND id < ? ORDER BY id", (student_id, session_id, upto, window_start))]
    if not evicted:
        return
    convo = "\n".join(f"{t['role']}: {t['content'][:300]}" for t in evicted)
    summary = get_llm().complete(
        SUMMARY_SYSTEM,
        [{"role": "user", "content": f"Current summary: {old or 'none'}\n\nNew turns:\n{convo}"}],
        task="summary", max_tokens=400, offline=lambda: _offline_summary(old, evicted),
    ).text.strip()
    with db.connect() as con:
        con.execute("INSERT OR REPLACE INTO summaries(student_id, session_id, summary, upto_turn) "
                    "VALUES (?,?,?,?)", (student_id, session_id, summary, evicted[-1]["id"]))


def save_turn(student_id: str, session_id: str, user_msg: str, standalone: str,
              reply: str) -> None:
    now = time.time()
    with db.connect() as con:
        con.execute("INSERT INTO turns(session_id, student_id, role, content, standalone, ts) "
                    "VALUES (?,?,?,?,?,?)", (session_id, student_id, "user", user_msg, standalone, now))
        con.execute("INSERT INTO turns(session_id, student_id, role, content, standalone, ts) "
                    "VALUES (?,?,?,?,?,?)", (session_id, student_id, "assistant", reply[:4000], None, now))
    _update_summary(student_id, session_id)


# ------------------------------------------------------------------ long-term
def update_weak_topics(student_id: str, results: list[dict]) -> None:
    now = time.time()
    with db.connect() as con:
        for r in results:
            topic = (r.get("topic") or "general").strip().lower()[:80]
            con.execute(
                "INSERT INTO weak_topics(student_id, topic, misses, hits, last_seen) VALUES "
                "(?,?,?,?,?) ON CONFLICT(student_id, topic) DO UPDATE SET "
                "misses = misses + excluded.misses, hits = hits + excluded.hits, "
                "last_seen = excluded.last_seen",
                (student_id, topic, 0 if r["ok"] else 1, 1 if r["ok"] else 0, now))


def weak_topics(student_id: str, limit: int = 5) -> list[str]:
    """Topics with more misses than hits, worst first."""
    with db.connect() as con:
        rows = con.execute(
            "SELECT topic FROM weak_topics WHERE student_id=? AND misses > hits "
            "ORDER BY (misses - hits) DESC, last_seen DESC LIMIT ?", (student_id, limit)).fetchall()
    return [r["topic"] for r in rows]
