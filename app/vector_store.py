"""Chroma collection ``course_notes``.

- Local persistent client by default, HTTP client when CHROMA_HOST is set
  (docker compose runs Chroma as its own service).
- Upsert by deterministic chunk ID, so re-running ingest never duplicates.
- Metadata filters: day, topic, file, owner.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

import chromadb

from app.config import get_settings
from app.ingest.chunker import Chunk
from app.ingest.embedder import Embedder

log = logging.getLogger(__name__)

PUBLIC = "public"


class ReindexRequired(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client():
    s = get_settings()
    if s.chroma_host:  # docker compose: Chroma runs as its own service
        last: Exception | None = None
        for attempt in range(30):
            try:
                client = chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)
                client.heartbeat()
                return client
            except Exception as e:  # server still starting
                last = e
                log.info("waiting for chroma at %s:%s (%d)", s.chroma_host, s.chroma_port, attempt)
                time.sleep(2)
        raise RuntimeError(f"cannot reach chroma at {s.chroma_host}:{s.chroma_port}") from last
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(s.chroma_dir))


class VectorStore:
    def __init__(self, name: str | None = None, embedder: Embedder | None = None) -> None:
        s = get_settings()
        self.embedder = embedder or Embedder()
        self.col = _client().get_or_create_collection(
            name or s.collection,
            metadata={"hnsw:space": "cosine", "embed_model": self.embedder.model_name},
        )
        stored = (self.col.metadata or {}).get("embed_model")
        if stored and stored != self.embedder.model_name and self.col.count():
            raise ReindexRequired(
                f"collection embedded with {stored}, current model is "
                f"{self.embedder.model_name}; run `python -m app.ingest --reset`"
            )

    # ------------------------------------------------------------ writes
    def upsert(self, chunks: list[Chunk], owner: str = PUBLIC, source: str = "course") -> int:
        if not chunks:
            return 0
        vecs = self.embedder.embed([c.text for c in chunks])
        metas = [
            {**c.metadata, "embed_model": self.embedder.model_name, "owner": owner, "source": source}
            for c in chunks
        ]
        for i in range(0, len(chunks), 256):
            self.col.upsert(
                ids=[c.id for c in chunks[i : i + 256]],
                documents=[c.text for c in chunks[i : i + 256]],
                embeddings=vecs[i : i + 256],
                metadatas=metas[i : i + 256],
            )
        return len(chunks)

    def delete_stale(self, file: str, keep_ids: set[str], owner: str = PUBLIC) -> int:
        got = self.col.get(where={"$and": [{"file": file}, {"owner": owner}]}, include=[])
        stale = [i for i in got["ids"] if i not in keep_ids]
        if stale:
            self.col.delete(ids=stale)
        return len(stale)

    def reset(self) -> None:
        name = self.col.name
        _client().delete_collection(name)
        self.col = _client().get_or_create_collection(
            name, metadata={"hnsw:space": "cosine", "embed_model": self.embedder.model_name}
        )

    # ------------------------------------------------------------ reads
    @staticmethod
    def build_where(
        owner: str | None = None, day: int | None = None, topic: str | None = None
    ) -> dict[str, Any] | None:
        conds: list[dict[str, Any]] = []
        conds.append({"owner": {"$in": [PUBLIC, owner]}} if owner else {"owner": PUBLIC})
        if day:
            conds.append({"day": int(day)})
        if topic:
            conds.append({"topic": topic.lower()})
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def query(
        self,
        text: str,
        k: int = 20,
        owner: str | None = None,
        day: int | None = None,
        topic: str | None = None,
    ) -> list[dict]:
        if not self.col.count():
            return []
        res = self.col.query(
            query_embeddings=[self.embedder.embed_query(text)],
            n_results=min(k, self.col.count()),
            where=self.build_where(owner, day, topic),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": i, "text": d, "metadata": m, "vector_score": 1.0 - float(dist)}
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def all_docs(self, owner: str | None = None) -> list[dict]:
        got = self.col.get(where=self.build_where(owner), include=["documents", "metadatas"])
        return [
            {"id": i, "text": d, "metadata": m}
            for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
        ]

    def count(self) -> int:
        return self.col.count()


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore()
