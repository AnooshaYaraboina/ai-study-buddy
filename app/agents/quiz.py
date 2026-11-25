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
_TERM_LINE = re.compile(r"^[A-Z(][\w /&+()'’\-·]{1,48}$")


def _definition_pairs(hit: dict) -> list[tuple[str, str]]:
    """Find 'Term' line followed by a definition sentence (handbook style)."""
    lines = [ln.strip() for ln in hit["text"].split("\n")]
    pairs = []
    for i in range(len(lines) - 1):
        term, nxt = lines[i], lines[i + 1]
        if (_TERM_LINE.match(term) and len(term.split()) <= 6 and not term.endswith(".")
                and "·" not in term and "·" not in nxt and len(nxt.split()) >= 6):
            definition = nxt
            j = i + 2
            while not definition.rstrip().endswith((".", ")")) and j < len(lines) and len(lines[j].split()) >= 3:
                definition += " " + lines[j]
                j += 1
            if definition.rstrip().endswith("."):
                pairs.append((term.rstrip(":"), definition))
    return pairs


def _statements(hit: dict) -> list[str]:
    return [ln.strip() for ln in hit["text"].split("\n")
            if len(ln.split()) >= 7 and ln.strip().endswith(".") and "·" not in ln]


def offline_quiz(topic: str, hits: list[dict], pool: list[dict], n: int = 5,
                 seed: int | None = None) -> list[QuizQuestion]:
    """Deterministic MCQs built from the notes (used without an API key)."""
    rng = random.Random(seed if seed is not None else hash(topic) & 0xFFFF)
    out: list[QuizQuestion] = []
    used: set[str] = set()
    other_defs = [d for h in pool for _, d in _definition_pairs(h)]
    other_stmts = [s for h in pool for s in _statements(h)]

    def add(question: str, correct: str, distractors: list[str], h: dict) -> None:
        ds = [d for d in dict.fromkeys(distractors) if d.lower() != correct.lower()][:3]
        if len(ds) < 3 or correct in used:
            return
        opts = ds + [correct]
        rng.shuffle(opts)
        m = h["metadata"]
        out.append(QuizQuestion(
            question=question, options=opts, answer=opts.index(correct),
            explanation=f"{correct} {cite(m)}", source=cite(m),
            topic=topic or m.get("topic", "general"),
        ))
        used.add(correct)

    for h in hits:  # 1) definition questions
        for term, definition in _definition_pairs(h):
            if len(out) >= n:
                break
            pool_d = [d for d in other_defs if d != definition]
            rng.shuffle(pool_d)
            add(f"According to your notes, which best describes “{term}”?", definition, pool_d, h)
    for h in hits:  # 2) "which statement is in the notes" questions
        for st in _statements(h):
            if len(out) >= n:
                break
            hit_terms = set(st.lower().split())
            pool_s = [s for s in other_stmts if len(hit_terms & set(s.lower().split())) < 4]
            rng.shuffle(pool_s)
            label = h["metadata"].get("topic") or topic
            add(f"Which statement about “{label}” matches your notes?", st, pool_s, h)
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
