"""SQLite app database: progress + cards (+ memory tables).

Every read/write takes ``student_id`` explicitly; the API layer derives it from
the bearer token, never from the request body, so students only ever see their
own rows.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS quiz_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    session_id TEXT,
    topic TEXT NOT NULL,
    question TEXT NOT NULL,
    correct INTEGER NOT NULL,
    source TEXT,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_attempts_student ON quiz_attempts(student_id, topic);

CREATE TABLE IF NOT EXISTS flashcards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    front TEXT NOT NULL,
    back TEXT NOT NULL,
    source TEXT,
    topic TEXT,
    ts REAL NOT NULL,
    UNIQUE(student_id, front)
);

CREATE TABLE IF NOT EXISTS weak_topics (
    student_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    misses INTEGER NOT NULL DEFAULT 0,
    hits INTEGER NOT NULL DEFAULT 0,
    last_seen REAL NOT NULL,
    PRIMARY KEY (student_id, topic)
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    standalone TEXT,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_turns_session ON turns(student_id, session_id, id);

CREATE TABLE IF NOT EXISTS summaries (
    student_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    upto_turn INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (student_id, session_id)
);

CREATE TABLE IF NOT EXISTS pending_quiz (
    student_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    quiz_json TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (student_id, session_id)
);
"""

_lock = threading.Lock()
_init_done: set[str] = set()


def _path() -> Path:
    return get_settings().db_path


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        if str(p) not in _init_done:
            with _lock:
                con.executescript(SCHEMA)
                _init_done.add(str(p))
        yield con
        con.commit()
    finally:
        con.close()


# ------------------------------------------------------------------ flashcards
def save_flashcard(student_id: str, front: str, back: str, source: str = "",
                   topic: str = "") -> dict:
    with connect() as con:
        con.execute(
            "INSERT INTO flashcards(student_id, front, back, source, topic, ts) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(student_id, front) DO UPDATE SET "
            "back=excluded.back, source=excluded.source, topic=excluded.topic, ts=excluded.ts",
            (student_id, front.strip(), back.strip(), source, topic, time.time()),
        )
        row = con.execute(
            "SELECT id FROM flashcards WHERE student_id=? AND front=?", (student_id, front.strip())
        ).fetchone()
    return {"id": row["id"], "front": front, "back": back, "source": source, "topic": topic}


def list_flashcards(student_id: str, limit: int = 50) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT id, front, back, source, topic FROM flashcards WHERE student_id=? "
            "ORDER BY ts DESC LIMIT ?", (student_id, limit)).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ attempts
def record_attempt(student_id: str, session_id: str, topic: str, question: str,
                   correct: bool, source: str = "") -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO quiz_attempts(student_id, session_id, topic, question, correct, source, ts)"
            " VALUES (?,?,?,?,?,?,?)",
            (student_id, session_id, topic, question, int(correct), source, time.time()),
        )


def progress(student_id: str) -> dict:
    with connect() as con:
        rows = con.execute(
            "SELECT topic, SUM(correct) AS right, COUNT(*) AS total FROM quiz_attempts "
            "WHERE student_id=? GROUP BY topic ORDER BY total DESC", (student_id,)).fetchall()
        weak = con.execute(
            "SELECT topic, misses, hits FROM weak_topics WHERE student_id=? AND misses > hits "
            "ORDER BY (misses - hits) DESC, last_seen DESC", (student_id,)).fetchall()
        cards = con.execute(
            "SELECT COUNT(*) AS n FROM flashcards WHERE student_id=?", (student_id,)).fetchone()
    topics = [{"topic": r["topic"], "right": r["right"], "total": r["total"],
               "accuracy": round(r["right"] / r["total"], 2)} for r in rows]
    total = sum(t["total"] for t in topics)
    right = sum(t["right"] for t in topics)
    return {
        "student_id": student_id,
        "questions_answered": total,
        "accuracy": round(right / total, 2) if total else None,
        "topics": topics,
        "weak_topics": [dict(r) for r in weak],
        "flashcards": cards["n"],
    }


# ------------------------------------------------------------------ pending quiz
def set_pending_quiz(student_id: str, session_id: str, quiz: list[dict] | None) -> None:
    with connect() as con:
        if quiz is None:
            con.execute("DELETE FROM pending_quiz WHERE student_id=? AND session_id=?",
                        (student_id, session_id))
        else:
            con.execute(
                "INSERT OR REPLACE INTO pending_quiz(student_id, session_id, quiz_json, ts) "
                "VALUES (?,?,?,?)", (student_id, session_id, json.dumps(quiz), time.time()))


def get_pending_quiz(student_id: str, session_id: str) -> list[dict] | None:
    with connect() as con:
        row = con.execute("SELECT quiz_json FROM pending_quiz WHERE student_id=? AND session_id=?",
                          (student_id, session_id)).fetchone()
    return json.loads(row["quiz_json"]) if row else None
