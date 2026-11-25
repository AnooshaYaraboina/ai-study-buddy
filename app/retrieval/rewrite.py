"""Rewrite follow-ups ("explain that again", "why?", "give an example") into full questions."""

from __future__ import annotations

import re

from app.llm import get_llm
from app.prompts import REWRITE_SYSTEM

_ANAPHORA = re.compile(
    r"\b(that|this|it|its|they|them|those|these|again|more|further|elaborate|above|"
    r"previous|same|one|example|why|simpler|differently)\b",
    re.IGNORECASE,
)
_PURE_FOLLOWUP = re.compile(
    r"^\s*(please\s+)?(can you\s+)?(explain|say|tell me|go over|describe)?\s*"
    r"(that|this|it)?\s*(again|more|in simpler terms|simpler|differently|more simply)?"
    r"\s*(please)?[?.!]*\s*$|^\s*(why|how|really|example|examples|more|go on|continue|"
    r"give (me )?an example|what about (that|it))\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def needs_rewrite(message: str, history: list[dict]) -> bool:
    if not any(t["role"] == "user" for t in history):
        return False
    words = message.split()
    return len(words) <= 12 and bool(_ANAPHORA.search(message) or _PURE_FOLLOWUP.match(message))


def _last_question(history: list[dict]) -> str:
    for t in reversed(history):
        if t["role"] == "user" and t.get("standalone"):
            return t["standalone"]
        if t["role"] == "user":
            return t["content"]
    return ""


def offline_rewrite(message: str, history: list[dict]) -> str:
    prev = _last_question(history)
    if not prev:
        return message
    if _PURE_FOLLOWUP.match(message):
        if re.search(r"example", message, re.I):
            return f"Give an example of: {prev}"
        if re.search(r"why", message, re.I):
            return f"Why? {prev}"
        return prev
    # "what about its limits?" -> substitute the pronoun with the previous question topic
    return f"{message.rstrip('?.! ')} (regarding: {prev})"


def rewrite_followup(message: str, history: list[dict]) -> str:
    """Return a standalone question (unchanged if it already is one)."""
    if not needs_rewrite(message, history):
        return message
    convo = "\n".join(f"{t['role']}: {t['content'][:400]}" for t in history[-6:])
    res = get_llm().complete(
        REWRITE_SYSTEM,
        [{"role": "user", "content": f"<conversation>\n{convo}\n</conversation>\n\n"
                                     f"Latest message: {message}"}],
        task="rewrite", max_tokens=300,
        offline=lambda: offline_rewrite(message, history),
    )
    out = res.text.strip().strip('"')
    return out or message
