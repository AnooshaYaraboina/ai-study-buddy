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
    sources: list[dict] = field(default_factory=list)


def format_notes(hits: list[dict], max_chars: int = 1800) -> str:
    out = []
    for h in hits:
        m = h["metadata"]
        out.append(f'<note source="{cite(m)}" topic="{m.get("topic", "")}">\n'
                   f'{h["text"][:max_chars]}\n</note>')
    return "\n".join(out)


def extractive_answer(question: str, hits: list[dict]) -> str:
    """No-LLM tutor answer: a readable answer composed from the notes (see app/answer.py)."""
    from app.answer import compose_answer

    return compose_answer(question, hits)


def sources_for(citations: list[str], hits: list[dict]) -> list[dict]:
    """The retrieved passage behind each citation, so the student can read it in the chat."""
    by_cite: dict[str, dict] = {}
    for h in hits:  # first (highest-ranked) passage wins for a page
        by_cite.setdefault(cite(h["metadata"]), h)
    out = []
    for c in citations:
        h = by_cite.get(c)
        if h:
            m = h["metadata"]
            out.append({"citation": c, "file": m["file"], "page": m["page"],
                        "topic": m.get("topic", ""), "day": m.get("day"), "text": h["text"]})
    return out


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
        """Two-signal refusal gate (thresholds tuned on the golden + off-topic probes)."""
        if not hits:
            return False
        s = self.settings
        best = hits[0]["score"]
        best_vec = max((h.get("vector_score") or 0.0) for h in hits)
        return best >= s.min_score or (best >= s.min_score_soft and best_vec >= s.min_vector)

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
            yield Answer(text, cites, hits, refused, best, question,
                         [] if refused else sources_for(cites, hits))

    @staticmethod
    def _collect(gen: Iterator) -> Answer:
        final = None
        for item in gen:
            if isinstance(item, Answer):
                final = item
        assert final is not None
        return final
