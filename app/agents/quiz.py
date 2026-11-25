"""Quiz-maker: 5 MCQs as strict JSON, a robust parser, and an offline generator."""

from __future__ import annotations

import json
import random
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.llm import get_llm
from app.prompts import QUIZ_SCHEMA, QUIZ_SYSTEM, QUIZ_USER
from app.rag import cite, format_notes


class QuizParseError(ValueError):
    pass


class QuizQuestion(BaseModel):
    question: str = Field(min_length=5)
    options: list[str]
    answer: int
    explanation: str = ""
    source: str = ""
    topic: str = ""

    @field_validator("options")
    @classmethod
    def _four_distinct(cls, v: list[str]) -> list[str]:
        v = [str(o).strip() for o in v]
        if len(v) != 4 or any(not o for o in v):
            raise ValueError("need exactly 4 non-empty options")
        if len({o.lower() for o in v}) != 4:
            raise ValueError("options must be distinct")
        return v


_LETTERS = "ABCD"


def _coerce_answer(ans: Any, options: list[str]) -> int:
    if isinstance(ans, bool):
        raise ValueError("bad answer")
    if isinstance(ans, int):
        if 0 <= ans <= 3:
            return ans
        if 1 <= ans <= 4:  # 1-based slipped through
            return ans - 1
        raise ValueError("answer index out of range")
    s = str(ans).strip()
    m = re.fullmatch(r"\(?([A-Da-d])[).:]?(\s.*)?", s)
    if m:
        return _LETTERS.index(m.group(1).upper())
    if s.isdigit():
        return _coerce_answer(int(s), options)
    for i, o in enumerate(options):
        if s.lower() == str(o).strip().lower():
            return i
    raise ValueError(f"cannot map answer {ans!r}")


def _extract_json(raw: str) -> Any:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # first balanced {...} or [...]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = raw.find(open_c)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(raw)):
                ch = raw[i]
                if in_str:
                    esc = (ch == "\\") and not esc
                    if ch == '"' and not esc:
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == open_c:
                    depth += 1
                elif ch == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(raw[start : i + 1])
                        except json.JSONDecodeError:
                            break
            start = raw.find(open_c, start + 1)
    raise QuizParseError("no JSON object found")


def parse_quiz_json(raw: str | dict | list) -> list[QuizQuestion]:
    """Parse model output into validated questions. Invalid items are dropped."""
    data = raw if isinstance(raw, (dict, list)) else _extract_json(raw)
    items = data.get("questions", data.get("quiz", [])) if isinstance(data, dict) else data
    if isinstance(data, dict) and "question" in data and "options" in data:
        items = [data]
    if not isinstance(items, list):
        raise QuizParseError("questions must be a list")
    out: list[QuizQuestion] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            opts = it.get("options") or it.get("choices") or []
            if isinstance(opts, dict):  # {"A": "...", "B": "..."}
                opts = [opts[k] for k in sorted(opts)]
            opts = [re.sub(r"^\s*[A-Da-d][).:]\s+", "", str(o)) for o in opts]
            q = QuizQuestion(
                question=str(it.get("question", "")).strip(),
                options=opts,
                answer=_coerce_answer(it.get("answer", it.get("correct")), opts),
                explanation=str(it.get("explanation", "")),
                source=str(it.get("source", "")),
                topic=str(it.get("topic", "")),
            )
            out.append(q)
        except (ValidationError, ValueError, KeyError, TypeError):
            continue
    if not out:
        raise QuizParseError("no valid questions")
    return out


# ------------------------------------------------------------------ offline generator
_GENERIC_LABEL = re.compile(
    r"^(days?\b|\d|step\b|lesson\b|situation\b|action\b|outcome\b|example\b|e\.g\b|defen[cs]e\b|"
    r"mechanism\b|watch out\b|big picture\b|goal\b|why\b|how\b|what\b|rule\b|golden rule\b|tip\b|"
    r"note\b|key idea\b|idea\b|remember\b|mistake\b|fix\b|after\b|before\b|result\b|impact\b|"
    r"risk\b|test\b|tools?\b|deliverables?\b|done when\b|check\b|common mistake|worked example)",
    re.I)
_CW = re.compile(r"[A-Za-z]{3,}")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("▸", "")).strip()


def _quizzable(hit_idx: int, hit: dict) -> tuple[list, list, list]:
    """(qa pairs, definitions, statements) from one passage, all as (prompt, answer)."""
    from app.answer import _nice, split_units

    qas, defs, stmts = [], [], []
    for u in split_units(hit_idx, hit["text"]):
        body = _clean(u.body)
        if u.fragmentary or len(_CW.findall(body)) < 4 or len(body) > 220:
            continue
        if u.kind == "qa":
            qas.append((_clean(u.label), body))
        elif u.kind == "labeled" and not _GENERIC_LABEL.match(u.label) and len(u.label.split()) <= 5:
            defs.append((_nice(u.label), body))
        elif u.kind == "sent" and body.endswith((".", "!")):
            stmts.append(("", body))
    return qas, defs, stmts


def offline_quiz(topic: str, hits: list[dict], pool: list[dict], n: int = 5,
                 seed: int | None = None) -> list[QuizQuestion]:
    """Deterministic MCQs built from the notes (used without an API key).

    Questions come from the notes' own flashcards (Q/A), definitions ("Term: meaning")
    and key sentences; wrong options are answers to *other* questions in the notes.
    """
    rng = random.Random(seed if seed is not None else hash(topic) & 0xFFFF)
    per_hit = [_quizzable(i, h) for i, h in enumerate(hits)]
    pool_items = [_quizzable(i, h) for i, h in enumerate(pool)]
    other_answers = [a for qa, d, st in pool_items + per_hit for _, a in qa + d]
    other_stmts = [a for _, _, st in pool_items + per_hit for _, a in st]
    out: list[QuizQuestion] = []
    used: set[str] = set()

    def words(t: str) -> set[str]:
        return {w.lower() for w in _CW.findall(t)}

    def add(question: str, correct: str, candidates: list[str], h: dict, term: str = "") -> None:
        if correct in used or len(out) >= n:
            return
        cw = words(correct)
        ds = []
        for d in dict.fromkeys(candidates):
            if d == correct or len(words(d) & cw) >= 4 or (term and term.lower() in d.lower()):
                continue  # too close to the right answer, or names the term being asked
            ds.append(d)
        if len(ds) < 3:
            return
        opts = rng.sample(ds, 3) + [correct]
        rng.shuffle(opts)
        m = h["metadata"]
        out.append(QuizQuestion(
            question=question, options=opts, answer=opts.index(correct),
            explanation=f"{correct} {cite(m)}", source=cite(m),
            topic=topic or m.get("topic", "general"),
        ))
        used.add(correct)

    # every candidate question, ranked by how well it matches the requested topic
    cands = []  # (kind, prompt, answer, hit, term)
    for h, (qas, defs, stmts) in zip(hits, per_hit):
        cands += [("qa", q if q.endswith("?") else f"{q.rstrip('.')}?", a, h, "") for q, a in qas]
        cands += [("def", f"According to your notes, what does “{t}” mean?", m, h, t)
                  for t, m in defs]
        label = h["metadata"].get("topic") or topic
        cands += [("stmt", f"Which statement about “{label}” is in your notes?", st, h, "")
                  for _, st in stmts]
    if cands:
        from app.answer import _similarities

        sims = _similarities(topic, [f"{c[4] or c[1]} {c[2]}" for c in cands])
        prior = {"qa": 0.08, "def": 0.05, "stmt": 0.0}  # flashcards make the best questions
        order = sorted(range(len(cands)), key=lambda i: -(sims[i] + prior[cands[i][0]]))
        for i in order:
            kind, prompt, answer, h, term = cands[i]
            add(prompt, answer, other_stmts if kind == "stmt" else other_answers, h, term)
    return out[:n]


# ------------------------------------------------------------------ public
def make_quiz(topic: str, hits: list[dict], pool: list[dict], n: int = 5) -> list[QuizQuestion]:
    """LLM quiz (structured output) -> parse -> top up offline so we always return n."""
    questions: list[QuizQuestion] = []
    res = get_llm().complete(
        QUIZ_SYSTEM,
        [{"role": "user", "content": QUIZ_USER.format(n=n, topic=topic, notes=format_notes(hits))}],
        task="quiz", max_tokens=4000, json_schema=QUIZ_SCHEMA,
        offline=lambda: json.dumps({"questions": [q.model_dump() for q in
                                                   offline_quiz(topic, hits, pool, n)]}),
    )
    try:
        questions = parse_quiz_json(res.text)
    except QuizParseError:
        questions = []
    if len(questions) < n:
        seen = {q.question for q in questions}
        questions += [q for q in offline_quiz(topic, hits, pool, n * 2) if q.question not in seen]
    return questions[:n]
