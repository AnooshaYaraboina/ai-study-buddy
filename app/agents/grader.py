"""Grader: score + per-question feedback + tips; saves missed questions as flashcards."""

from __future__ import annotations

import re

from app import db
from app.agents.tools import SaveFlashcardInput, ToolContext, save_flashcard
from app.llm import get_llm
from app.prompts import GRADER_SYSTEM

_LET = "ABCD"


def looks_like_answers(text: str, n: int) -> bool:
    return len(parse_answers(text, n)) == n


def parse_answers(text: str, n: int) -> list[int]:
    """'A B C D A' / 'a,c,b,d,a' / '1:B 2:C ...' / 'ACBDA' -> [0,2,1,3,0]."""
    t = re.sub(r"^\s*(answers?|my answers? (are|is))\s*[:\-]?\s*", "", text.strip(), flags=re.I)
    pairs = re.findall(r"(\d+)\s*[:.)=\-]\s*([A-Da-d])\b", t)
    if pairs:
        d = {int(k): _LET.index(v.upper()) for k, v in pairs}
        return [d[i] for i in range(1, n + 1) if i in d] if len(d) == n else []
    letters = re.findall(r"\b([A-Da-d])\b", t)
    if len(letters) != n and re.fullmatch(r"[A-Da-d]{%d}" % n, t.replace(" ", "")):
        letters = list(t.replace(" ", ""))
    if len(letters) == n and not re.search(r"[E-Ze-z]{2,}", re.sub(r"\b[A-Da-d]\b", "", t)):
        return [_LET.index(c.upper()) for c in letters]
    return []


def grade(quiz: list[dict], answers: list[int], ctx: ToolContext) -> dict:
    results, missed = [], []
    for q, a in zip(quiz, answers):
        ok = a == q["answer"]
        results.append({
            "question": q["question"], "your_answer": _LET[a], "correct_answer": _LET[q["answer"]],
            "correct_text": q["options"][q["answer"]], "ok": ok,
            "explanation": q.get("explanation", ""), "source": q.get("source", ""),
            "topic": q.get("topic", ""),
        })
        db.record_attempt(ctx.student_id, ctx.session_id, q.get("topic") or "general",
                          q["question"], ok, q.get("source", ""))
        if not ok:
            missed.append(q)
            save_flashcard(ctx, SaveFlashcardInput(
                front=q["question"][:300], back=q["options"][q["answer"]][:1000],
                source=q.get("source", "")[:200], topic=(q.get("topic") or "")[:80]))
    score = sum(r["ok"] for r in results)

    def offline_tips() -> str:
        if not missed:
            return "Perfect score — great work! Try a harder topic next."
        tips = [f"- Q: {r['question'][:90]}… → correct: {r['correct_answer']}. "
                f"{r['correct_text'][:140]} {r['source']}" for r in results if not r["ok"]]
        return "\n".join(tips) + "\nNice effort — I saved the missed ones as flashcards " \
                                 "and I'll quiz you on them again next time."

    if missed:
        detail = "\n".join(
            f"Q: {r['question']}\nStudent: {r['your_answer']} | Correct: {r['correct_answer']} "
            f"({r['correct_text']})\nExplanation: {r['explanation']} {r['source']}"
            for r in results if not r["ok"])
        tips = get_llm().complete(GRADER_SYSTEM, [{"role": "user", "content": detail}],
                                  task="grade", max_tokens=800, offline=offline_tips).text
    else:
        tips = offline_tips()
    return {"score": score, "total": len(results), "results": results, "feedback": tips,
            "missed_topics": sorted({q.get("topic") or "general" for q in missed})}


def format_grade(g: dict) -> str:
    lines = [f"**Score: {g['score']}/{g['total']}**", ""]
    for i, r in enumerate(g["results"], 1):
        mark = "✅" if r["ok"] else "❌"
        extra = "" if r["ok"] else f" — correct: **{r['correct_answer']}** ({r['correct_text'][:120]})"
        lines.append(f"{mark} Q{i}: you chose {r['your_answer']}{extra} {r['source']}")
    lines += ["", g["feedback"]]
    return "\n".join(lines)
