"""Cited RAG answers: retrieve top-5, refuse below threshold, answer with [file p.N]."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from app.config import get_settings
from app.llm import get_llm
from app.observability import tracer
from app.prompts import NOT_IN_NOTES, TUTOR_SYSTEM, TUTOR_USER
from app.retrieval import Retriever, get_retriever

CITE_RE = re.compile(r"\[([^\[\]]+?\.pdf) p\.(\d+)\]")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the and or of to in on for is are was were be been what which who how why
when where does do did can could should would i me my you your it its this that these those
with as at by from about explain tell describe define meaning mean give example vs versus
between difference please again more""".split())


def cite(meta: dict) -> str:
    return f"[{meta['file']} p.{meta['page']}]"


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)
    refused: bool = False
    best_score: float = 0.0
    question: str = ""


def format_notes(hits: list[dict], max_chars: int = 1800) -> str:
    out = []
    for h in hits:
        m = h["metadata"]
        out.append(f'<note source="{cite(m)}" topic="{m.get("topic", "")}">\n'
                   f'{h["text"][:max_chars]}\n</note>')
    return "\n".join(out)


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def extractive_answer(question: str, hits: list[dict], max_sents: int = 4) -> str:
    """Offline tutor: best-matching sentences from the notes, each cited."""
    q = _terms(question)
    scored = []
    for rank, h in enumerate(hits):
        lines = [ln.strip() for ln in h["text"].split("\n")[1:]
                 if len(ln.strip()) > 25 and len(ln.split()) >= 5 and " · " not in ln]
        for ln in lines:
            overlap = len(q & _terms(ln))
            if overlap:
                bonus = 0.5 if ln.rstrip().endswith((".", ")")) else 0.0
                scored.append((overlap + bonus - rank * 0.3, ln, cite(h["metadata"])))
    if not scored:  # fall back to the first lines of the top hit
        h = hits[0]
        lines = [ln for ln in h["text"].split("\n")[1:] if len(ln) > 25][:2]
        scored = [(0, ln, cite(h["metadata"])) for ln in lines]
    scored.sort(key=lambda x: -x[0])
    # lead with the top note's first full sentence: in these notes it is the definition
    lead = next((ln.strip() for ln in hits[0]["text"].split("\n")[1:]
                 if ln.strip().endswith(".") and len(ln.split()) >= 6 and " · " not in ln
                 and not ln.startswith("▸")), None)
    if lead:
        scored.insert(0, (99, lead, cite(hits[0]["metadata"])))
    scored = [x for x in scored if not x[1].startswith("▸")]
    seen, bullets = set(), []
    for _, ln, c in scored:
        if ln in seen:
            continue
        seen.add(ln)
        bullets.append(f"- {ln} {c}")
        if len(bullets) >= max_sents:
            break
    return "From your notes:\n" + "\n".join(bullets)


def enforce_citations(text: str, hits: list[dict]) -> tuple[str, list[str]]:
    """Drop citations to notes we did not retrieve; ensure at least one citation."""
    allowed = {cite(h["metadata"]) for h in hits}
    found = []
    for m in CITE_RE.finditer(text):
        c = f"[{m.group(1)} p.{m.group(2)}]"
        if c not in allowed:
            text = text.replace(c, "")
        elif c not in found:
            found.append(c)
    if not found and NOT_IN_NOTES not in text:
        top = cite(hits[0]["metadata"])
        text = text.rstrip() + f"\n\nSource: {top}"
        found = [top]
    return text, found


def refusal(hits: list[dict]) -> str:
    hint = ""
    if hits and hits[0].get("score", 0) >= 0.05:
        hint = f"\nThe closest topic I do have is “{hits[0]['metadata'].get('topic', '')}”."
    return NOT_IN_NOTES + hint


class Tutor:
    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever or get_retriever()
        self.settings = get_settings()

    def retrieve(self, question: str, owner: str | None = None, **filters) -> list[dict]:
        return self.retriever.search(question, k=self.settings.top_k, owner=owner, **filters)

    def _prompt(self, question: str, hits: list[dict], weak: list[str], summary: str) -> list[dict]:
        return [{"role": "user", "content": TUTOR_USER.format(
            notes=format_notes(hits), weak_topics=", ".join(weak) or "none",
            summary=summary or "none", question=question)}]

    def is_answerable(self, hits: list[dict]) -> bool:
        return bool(hits) and hits[0]["score"] >= self.settings.min_score

    def answer(self, question: str, owner: str | None = None, weak: list[str] | None = None,
               summary: str = "", **filters) -> Answer:
        return self._collect(self.answer_stream(question, owner, weak, summary, **filters))

    def answer_stream(self, question: str, owner: str | None = None,
                      weak: list[str] | None = None, summary: str = "",
                      **filters) -> Iterator[str | Answer]:
        """Yields text deltas, then a final ``Answer`` object."""
        with tracer().start_as_current_span("rag.answer") as span:
            hits = self.retrieve(question, owner=owner, **filters)
            best = hits[0]["score"] if hits else 0.0
            span.set_attributes({"rag.best_score": best, "rag.hits": len(hits)})
            if not self.is_answerable(hits):
                text = refusal(hits)
                yield text
                yield Answer(text, [], hits, True, best, question)
                return
            parts: list[str] = []
            for delta in get_llm().stream(
                TUTOR_SYSTEM, self._prompt(question, hits, weak or [], summary),
                task="tutor", max_tokens=1500,
                offline=lambda: extractive_answer(question, hits),
            ):
                parts.append(delta)
                yield delta
            raw = "".join(parts)
            text, cites = enforce_citations(raw, hits)
            if text != raw:  # tell the client what was appended / removed
                yield {"replace": text}  # type: ignore[misc]
            refused = NOT_IN_NOTES in text
            span.set_attributes({"rag.citations": len(cites), "rag.refused": refused})
            yield Answer(text, cites, hits, refused, best, question)

    @staticmethod
    def _collect(gen: Iterator) -> Answer:
        final = None
        for item in gen:
            if isinstance(item, Answer):
                final = item
        assert final is not None
        return final
