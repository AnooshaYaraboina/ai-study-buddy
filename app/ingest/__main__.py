"""CLI: python -m app.ingest [--dir data/pdfs] [--reset]"""

from __future__ import annotations

import argparse
import logging

from app.ingest.pipeline import ingest_dir
from app.vector_store import get_store


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest course PDFs into Chroma")
    ap.add_argument("--dir", default=None, help="folder of PDFs (default: PDF_DIR)")
    ap.add_argument("--reset", action="store_true", help="drop and rebuild the collection")
    ap.add_argument("--if-empty", action="store_true", help="only ingest when collection is empty")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.if_empty and get_store().count():
        print(f"collection already has {get_store().count()} chunks; skipping")
        return
    reports = ingest_dir(args.dir, reset=args.reset)
    for r in reports:
        print(f"{r.file:60s} pages={r.pages:3d} chunks={r.chunks:3d} "
              f"stale_removed={r.stale_removed} dropped_lines={r.dropped_lines}")
    print(f"total chunks in collection: {get_store().count()}")


if __name__ == "__main__":
    main()
