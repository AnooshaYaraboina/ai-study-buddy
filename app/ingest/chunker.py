"""Heading-aware chunker.

Rules
- Chunks never cross a page boundary (citations are ``[file p.N]``).
- Target ~``max_tokens`` tokens; a new heading starts a new chunk when the
  current one is already substantial.
- ``list`` and ``table`` blocks are atomic: never cut in half. Only ``para``
  blocks may be split (at sentence boundaries) when they alone exceed the limit.
- A small overlap (tail of the previous chunk's prose) is carried into the
  next chunk on the same page. Lists/tables are never used as partial overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingest.pdf_loader import Block, Page

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (word pieces + punctuation)."""
    return len(_TOKEN_RE.findall(text))


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_para(text: str, max_tokens: int) -> list[str]:
    parts, cur = [], ""
    for sent in _SENT_RE.split(text):
        cand = f"{cur} {sent}".strip()
        if cur and count_tokens(cand) > max_tokens:
            parts.append(cur)
            cur = sent
        else:
            cur = cand
    if cur:
        parts.append(cur)
    return parts


def _tail(text: str, n_tokens: int) -> str:
    words = text.split()
    out: list[str] = []
    for w in reversed(words):
        if count_tokens(" ".join([w, *out])) > n_tokens:
            break
        out.insert(0, w)
    return " ".join(out)


def _units(blocks: list[Block], max_tokens: int) -> list[Block]:
    """Explode over-long paragraphs; keep list/table/heading blocks whole."""
    out: list[Block] = []
    for b in blocks:
        if b.kind == "para" and count_tokens(b.text) > max_tokens:
            out.extend(Block(p, "para") for p in _split_para(b.text, max_tokens))
        else:
            out.append(b)
    return out


def chunk_page(
    page: Page,
    max_tokens: int = 500,
    overlap: int = 60,
    min_tokens_before_heading_break: int = 150,
    id_prefix: str | None = None,
) -> list[Chunk]:
    prefix = id_prefix or re.sub(r"\W+", "_", page.file.rsplit(".", 1)[0])
    units = _units(page.blocks, max_tokens)

    chunks: list[tuple[str, list[Block]]] = []  # (heading, blocks)
    heading = ""
    cur: list[Block] = []
    cur_heading = ""

    def size(bs: list[Block]) -> int:
        return sum(count_tokens(b.text) for b in bs)

    def flush() -> None:
        nonlocal cur
        if any(b.kind != "heading" for b in cur) or cur:
            chunks.append((cur_heading, cur))
        cur = []

    for b in units:
        if b.kind == "heading":
            if cur and size(cur) >= min_tokens_before_heading_break:
                flush()
            heading = b.text
            if not cur:
                cur_heading = heading
            cur.append(b)
            continue
        if cur and size(cur) + count_tokens(b.text) > max_tokens:
            prev = cur
            flush()
            cur_heading = heading
            # overlap: tail of the previous chunk's last prose block
            last_para = next((x for x in reversed(prev) if x.kind == "para"), None)
            if overlap and last_para is not None and prev[-1] is last_para:
                cur.append(Block(_tail(last_para.text, overlap), "overlap"))
        if not cur:
            cur_heading = heading
        cur.append(b)
    if cur:
        flush()

    out: list[Chunk] = []
    for i, (head, blocks) in enumerate(chunks):
        body = "\n".join(b.text for b in blocks).strip()
        if count_tokens(body) < 8:  # skip near-empty fragments
            continue
        title = page.topic.title() if page.topic else page.file
        text = f"{title} — {head}\n{body}" if head and head not in body[:200] else f"{title}\n{body}"
        out.append(
            Chunk(
                id=f"{prefix}-p{page.page}-c{i}",
                text=text,
                metadata={
                    "file": page.file,
                    "page": page.page,
                    "day": page.day,
                    "topic": page.topic,
                    "heading": head or page.topic,
                    "chunk_index": i,
                    "n_tokens": count_tokens(text),
                    "has_table": any(b.kind == "table" for b in blocks),
                    "has_list": any(b.kind == "list" for b in blocks),
                },
            )
        )
    return out


def chunk_pages(pages: list[Page], max_tokens: int = 500, overlap: int = 60) -> list[Chunk]:
    out: list[Chunk] = []
    for p in pages:
        out.extend(chunk_page(p, max_tokens=max_tokens, overlap=overlap))
    return out
