"""Retrieval over the course_notes collection."""

from __future__ import annotations

from functools import lru_cache

from app.observability import tracer
from app.vector_store import VectorStore, get_store


class Retriever:
    """Vector-only retriever (score = cosine similarity)."""

    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or get_store()

    def search(self, query: str, k: int = 5, owner: str | None = None,
               day: int | None = None, topic: str | None = None) -> list[dict]:
        with tracer().start_as_current_span("retrieve.vector") as span:
            hits = self.store.query(query, k=k, owner=owner, day=day, topic=topic)
            for h in hits:
                h["score"] = h["vector_score"]
            span.set_attribute("retrieve.n", len(hits))
            return hits


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
