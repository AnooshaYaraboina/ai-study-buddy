"""One embedding model for every chunk and every query.

The model name is stored on every vector (metadata ``embed_model``) so that a
re-index with a different model is detectable and safe.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    log.info("loading embedding model %s", name)
    return SentenceTransformer(name, device="cpu")


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().embed_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = _model(self.model_name).encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
