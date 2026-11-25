"""Hybrid search: BM25 keyword + vector, fused with RRF, then a cross-encoder reranker.

score returned on each hit = sigmoid(cross-encoder logit) in [0, 1]; the tutor
refuses when the best score is below MIN_SCORE.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.observability import tracer
from app.vector_store import PUBLIC, VectorStore, get_store

log = logging.getLogger(__name__)
_TOK = re.compile(r"[a-z0-9@]+")
_STOP = set("""a an the and or of to in on for is are was were be what which who how why when
where does do did can i me my you your it its this that with as at by from""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


@lru_cache(maxsize=2)
def _cross_encoder(name: str):
    from sentence_transformers import CrossEncoder

    log.info("loading reranker %s", name)
    return CrossEncoder(name, device="cpu")


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().rerank_model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        logits = _cross_encoder(self.model_name).predict(
            [(query, t[:2000]) for t in texts], show_progress_bar=False
        )
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


class BM25Index:
    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._docs: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._count = -1

    def _ensure(self) -> None:
        n = self.store.count()
        if n == self._count and self._bm25 is not None:
            return
        with self._lock:
            # all owners: filtered per query below
            got = self.store.col.get(include=["documents", "metadatas"])
            self._docs = [
                {"id": i, "text": d, "metadata": m}
                for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
            ]
            self._bm25 = BM25Okapi([tokenize(d["text"]) for d in self._docs]) if self._docs else None
            self._count = n

    def invalidate(self) -> None:
        self._count = -1

    def search(self, query: str, k: int, owner: str | None, day: int | None,
               topic: str | None) -> list[dict]:
        self._ensure()
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        owners = {PUBLIC, owner} if owner else {PUBLIC}
        out = []
        for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
            if scores[i] <= 0:
                break
            m = self._docs[i]["metadata"]
            if m.get("owner", PUBLIC) not in owners:
                continue
            if day and m.get("day") != int(day):
                continue
            if topic and m.get("topic") != topic.lower():
                continue
            out.append({**self._docs[i], "bm25_score": float(scores[i])})
            if len(out) >= k:
                break
        return out


def rrf(*ranked_lists: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal-rank fusion over lists of hits keyed by id."""
    fused: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, h in enumerate(lst):
            e = fused.setdefault(h["id"], {**h, "rrf": 0.0})
            e.update({k2: v for k2, v in h.items() if k2 not in e})
            e["rrf"] += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda h: -h["rrf"])


class HybridRetriever:
    def __init__(self, store: VectorStore | None = None, reranker: Reranker | None = None,
                 n_candidates: int = 20, use_rerank: bool = True) -> None:
        self.store = store or get_store()
        self.bm25 = BM25Index(self.store)
        self.reranker = reranker or Reranker()
        self.n_candidates = n_candidates
        self.use_rerank = use_rerank

    def invalidate(self) -> None:
        self.bm25.invalidate()

    def search(self, query: str, k: int = 5, owner: str | None = None,
               day: int | None = None, topic: str | None = None) -> list[dict]:
        with tracer().start_as_current_span("retrieve.hybrid") as span:
            vec = self.store.query(query, k=self.n_candidates, owner=owner, day=day, topic=topic)
            kw = self.bm25.search(query, self.n_candidates, owner, day, topic)
            fused = rrf(vec, kw)[: self.n_candidates]
            if self.use_rerank and fused:
                with tracer().start_as_current_span("retrieve.rerank"):
                    for h, s in zip(fused, self.reranker.score(query, [h["text"] for h in fused])):
                        h["rerank_score"] = s
                        h["score"] = s
                fused.sort(key=lambda h: -h["score"])
            else:
                for h in fused:
                    h["score"] = h.get("vector_score", 0.0)
            span.set_attributes({
                "retrieve.vector_n": len(vec), "retrieve.bm25_n": len(kw),
                "retrieve.best": fused[0]["score"] if fused else 0.0,
            })
            return fused[:k]
