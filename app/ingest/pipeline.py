"""parse -> sanitize -> chunk -> embed -> upsert into Chroma."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.ingest.chunker import chunk_pages
from app.ingest.pdf_loader import load_pdf
from app.vector_store import PUBLIC, VectorStore, get_store

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    file: str
    pages: int
    chunks: int
    stale_removed: int
    dropped_lines: int


def ingest_file(
    path: Path, store: VectorStore | None = None, owner: str = PUBLIC, source: str = "course"
) -> IngestReport:
    s = get_settings()
    store = store or get_store()
    pages = load_pdf(Path(path), max_pages=s.max_pdf_pages, untrusted=source != "course")
    chunks = chunk_pages(pages, max_tokens=s.chunk_tokens, overlap=s.chunk_overlap)
    if owner != PUBLIC:  # uploaded notes: namespace IDs per student
        for c in chunks:
            c.id = f"{owner}:{c.id}"
    store.upsert(chunks, owner=owner, source=source)
    stale = store.delete_stale(Path(path).name, {c.id for c in chunks})
    dropped = sum(len(p.dropped) for p in pages)
    rep = IngestReport(Path(path).name, len(pages), len(chunks), stale, dropped)
    log.info("ingested %s", rep)
    return rep


def ingest_dir(pdf_dir: Path | None = None, reset: bool = False) -> list[IngestReport]:
    s = get_settings()
    store = get_store()
    if reset:
        store.reset()
    files = sorted(Path(pdf_dir or s.pdf_dir).glob("*.pdf"))
    if not files:
        log.warning("no PDFs found in %s", pdf_dir or s.pdf_dir)
    return [ingest_file(f, store) for f in files]
