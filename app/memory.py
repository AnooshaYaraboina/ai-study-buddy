"""Memory (short-term turns). Extended in feat: memory + guards."""

from __future__ import annotations

import time

from app import db

SHORT_TERM_TURNS = 6


def load(student_id: str, session_id: str) -> dict:
    with db.connect() as con:
        rows = con.execute(
            "SELECT role, content, standalone FROM turns WHERE student_id=? AND session_id=? "
            "ORDER BY id DESC LIMIT ?", (student_id, session_id, SHORT_TERM_TURNS)).fetchall()
    history = [dict(r) for r in reversed(rows)]
    return {"history": history, "summary": "", "weak_topics": []}


def save_turn(student_id: str, session_id: str, user_msg: str, standalone: str,
              reply: str) -> None:
    now = time.time()
    with db.connect() as con:
        con.execute("INSERT INTO turns(session_id, student_id, role, content, standalone, ts) "
                    "VALUES (?,?,?,?,?,?)", (session_id, student_id, "user", user_msg, standalone, now))
        con.execute("INSERT INTO turns(session_id, student_id, role, content, standalone, ts) "
                    "VALUES (?,?,?,?,?,?)", (session_id, student_id, "assistant", reply[:4000], None, now))
