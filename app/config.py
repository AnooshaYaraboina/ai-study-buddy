"""Central settings, read once from the environment (.env supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _tokens(raw: str) -> dict[str, str]:
    """Parse "student:token,student2:token2" into {token: student}."""
    out: dict[str, str] = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        student, _, token = pair.partition(":")
        if student and token:
            out[token] = student
    return out


@dataclass(frozen=True)
class Settings:
    # LLM
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-sonnet-5-5"))
    llm_effort: str = field(default_factory=lambda: _env("LLM_EFFORT", "low"))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "60")))
    llm_max_retries: int = field(default_factory=lambda: int(_env("LLM_MAX_RETRIES", "2")))
    llm_offline_flag: bool = field(default_factory=lambda: _env("LLM_OFFLINE", "0") == "1")

    # Retrieval
    embed_model: str = field(
        default_factory=lambda: _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    rerank_model: str = field(
        default_factory=lambda: _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    chroma_host: str = field(default_factory=lambda: _env("CHROMA_HOST"))
    chroma_port: int = field(default_factory=lambda: int(_env("CHROMA_PORT", "8000")))
    chroma_dir: Path = field(default_factory=lambda: ROOT / _env("CHROMA_DIR", "data/chroma"))
    collection: str = field(default_factory=lambda: _env("COLLECTION", "course_notes"))
    min_score: float = field(default_factory=lambda: float(_env("MIN_SCORE", "0.25")))
    top_k: int = 5
    chunk_tokens: int = 500
    chunk_overlap: int = 60

    # App
    pdf_dir: Path = field(default_factory=lambda: ROOT / _env("PDF_DIR", "data/pdfs"))
    upload_dir: Path = field(default_factory=lambda: ROOT / _env("UPLOAD_DIR", "data/uploads"))
    db_path: Path = field(default_factory=lambda: ROOT / _env("DB_PATH", "data/study_buddy.db"))
    log_dir: Path = field(default_factory=lambda: ROOT / _env("LOG_DIR", "logs"))
    api_url: str = field(default_factory=lambda: _env("API_URL", "http://localhost:8080"))
    api_tokens: dict[str, str] = field(
        default_factory=lambda: _tokens(_env("API_TOKENS", "alice:alice-token,bob:bob-token"))
    )
    rate_limit_per_min: int = field(default_factory=lambda: int(_env("RATE_LIMIT_PER_MIN", "30")))
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "10")))
    max_pdf_pages: int = field(default_factory=lambda: int(_env("MAX_PDF_PAGES", "200")))
    max_message_chars: int = 2000

    # Observability
    langfuse_public_key: str = field(default_factory=lambda: _env("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str = field(default_factory=lambda: _env("LANGFUSE_SECRET_KEY"))
    langfuse_host: str = field(
        default_factory=lambda: _env("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    otlp_endpoint: str = field(default_factory=lambda: _env("OTEL_EXPORTER_OTLP_ENDPOINT"))

    @property
    def llm_offline(self) -> bool:
        """Offline stub when forced, or when there is no API key at all."""
        return self.llm_offline_flag or not self.anthropic_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
