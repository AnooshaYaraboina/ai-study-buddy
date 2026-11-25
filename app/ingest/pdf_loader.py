"""Load a PDF page by page into structured blocks with metadata.

Each page becomes a ``Page`` with metadata (file, page, day, topic) and a list
of ``Block`` objects (heading / para / list / table). The chunker consumes
blocks, so it can keep a table or list together and stay heading-aware.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import pymupdf

from app.ingest.sanitize import clean_text, is_injection

log = logging.getLogger(__name__)

LIST_RE = re.compile(r"^(\s*([•\-–\*·▪●○]|\d{1,2}[.)]?|[a-hA-H][.)]|Q:|A:)\s+\S|\s*\d{1,2}\s*$)")
DAY_IN_HEADER = re.compile(r"\bday\s*(\d{1,2})\b", re.IGNORECASE)
DAYS_IN_NAME = re.compile(r"Days?_(\d{1,2})(?:_(\d{1,2}))?", re.IGNORECASE)


@dataclass
class Block:
    text: str
    kind: str  # heading | para | list | table


@dataclass
class Page:
    file: str
    page: int  # 1-based, as printed in citations
    day: int  # 0 = not tied to a single day
    topic: str
    blocks: list[Block] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


class PDFTooLarge(ValueError):
    pass


def _file_days(name: str) -> tuple[int, int]:
    m = DAYS_IN_NAME.search(name)
    if not m:
        return 0, 0
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    return a, b


def _parse_ranges(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def _manifest_days(path: Path) -> dict[int, int]:
    """Optional data/pdfs/manifest.json: {"file.pdf": {"days": {"24": "1-11"}}}."""
    mf = path.parent / "manifest.json"
    if not mf.exists():
        return {}
    try:
        entry = json.loads(mf.read_text()).get(path.name, {})
    except (json.JSONDecodeError, OSError):
        return {}
    return {p: int(day) for day, spec in entry.get("days", {}).items() for p in _parse_ranges(spec)}


def _visible(span: dict, page_rect: pymupdf.Rect) -> bool:
    """A human-visible span: readable size, not transparent, on the page."""
    if span.get("size", 10) < 4:
        return False
    if span.get("alpha", 255) == 0:
        return False
    x0, y0, x1, y1 = span["bbox"]
    return pymupdf.Rect(x0, y0, x1, y1).intersects(page_rect)


def _page_lines(page: pymupdf.Page) -> tuple[list[tuple[str, float, tuple, int]], int]:
    """Return visible lines as (text, size, bbox, block_no) and hidden-span count."""
    rect = page.rect
    out, hidden = [], 0
    data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
    for bno, block in enumerate(data["blocks"]):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            vis = [s for s in spans if _visible(s, rect)]
            hidden += len(spans) - len(vis)
            if not vis:
                continue
            parts = [vis[0]["text"]]
            for prev, cur in zip(vis, vis[1:]):
                gap = cur["bbox"][0] - prev["bbox"][2]
                if gap > 0.15 * cur["size"] and not prev["text"].endswith(" ") \
                        and not cur["text"].startswith(" "):
                    parts.append(" ")
                parts.append(cur["text"])
            text = clean_text("".join(parts))
            if text:
                size = max(s["size"] for s in vis)
                out.append((text, size, tuple(line["bbox"]), bno))
    return out, hidden


def _table_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    try:
        return [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
    except Exception:  # table finder is best-effort
        return []


def _topic(header_lines: list[str], file_stem: str) -> str:
    """Best-effort page topic from the running header."""
    for ln in header_lines[:6]:
        # visual notes: "AI security · injection"
        if "·" in ln and len(ln) < 70:
            tail = ln.split("·")[-1].strip()
            tail = re.sub(r"\s*\d+$", "", tail)
            if tail and not tail.lower().startswith("day") and not tail.isdigit():
                return tail.lower()
    return file_stem.split("_")[-1].lower()


def load_pdf(path: Path, max_pages: int = 500, untrusted: bool = False) -> list[Page]:
    """Parse a PDF. ``untrusted`` (uploads) also drops instruction-like lines."""
    path = Path(path)
    doc = pymupdf.open(path)
    if doc.page_count > max_pages:
        raise PDFTooLarge(f"{path.name}: {doc.page_count} pages > limit {max_pages}")

    first_day, last_day = _file_days(path.stem)
    overrides = _manifest_days(path)
    multi_day_book = last_day - first_day > 1  # e.g. a Days 01-21 handbook
    raw_pages = []
    for page in doc:
        lines, hidden = _page_lines(page)
        raw_pages.append((page, lines, hidden, _table_rects(page)))

    # Running headers / nav strips: identical lines on >= 40% of pages.
    counts = Counter()
    for _, lines, _, _ in raw_pages:
        counts.update({t for t, *_ in lines})
    n = max(1, len(raw_pages))
    boiler = {t for t, c in counts.items() if n >= 5 and c / n >= 0.4}

    pages: list[Page] = []
    current_day = first_day
    for page, lines, hidden, tables in raw_pages:
        pno = page.number + 1
        texts = [t for t, *_ in lines]

        # Day tracking: header says "day 21" / "DAY 15 / CORE CONCEPTS".
        for t in texts[:5]:
            m = DAY_IN_HEADER.search(t)
            if m and (first_day == 0 or first_day <= int(m.group(1)) <= max(last_day, 21)):
                current_day = int(m.group(1))
                break
        header_day = None
        for t in texts[:5]:
            m = DAY_IN_HEADER.search(t)
            if m:
                header_day = int(m.group(1))
                break
        if pno in overrides:
            day = overrides[pno]
        elif multi_day_book:
            day = header_day if header_day and first_day <= header_day <= last_day else 0
        else:
            day = current_day if first_day else 0

        topic = _topic(texts, path.stem)
        # Handbook pages: "DAY 15 / CORE CONCEPTS" then the title line.
        for i, t in enumerate(texts[:8]):
            if re.match(r"^DAY\s*\d+\s*/", t) and i + 2 < len(texts):
                cand = texts[i + 2] if texts[i + 1].isdigit() or texts[i + 1].startswith("DAY") else texts[i + 1]
                if re.match(r"^DAY\s*\d+\s*/", cand) and i + 3 < len(texts):
                    cand = texts[i + 3]
                topic = re.sub(r"^Apply & explain:\s*", "", cand).strip().lower()
                break

        sizes = [s for _, s, *_ in lines] or [10]
        body = median(sizes)
        pg = Page(file=path.name, page=pno, day=day, topic=topic)
        if hidden:
            pg.dropped.append(f"<{hidden} hidden spans>")

        groups: dict[int, list[tuple[str, float, tuple]]] = {}
        for t, s, bbox, bno in lines:
            if t in boiler or t.isdigit() and len(t) <= 3 and s <= body:
                continue
            if untrusted and is_injection(t):
                pg.dropped.append(t)
                continue
            groups.setdefault(bno, []).append((t, s, bbox))

        for items in groups.values():
            text = "\n".join(t for t, *_ in items)
            first, size, bbox = items[0]
            in_table = any(r.contains(pymupdf.Rect(bbox)) for r in tables)
            if in_table:
                kind = "table"
            elif first.startswith("▸") or (size >= body * 1.18 and len(text) < 90):
                kind = "heading"
            elif sum(bool(LIST_RE.match(t)) for t, *_ in items) >= max(1, len(items) // 2):
                kind = "list"
            else:
                kind = "para"
            if kind == "heading":
                text = text.lstrip("▸ ").replace("\n", " ")
            pg.blocks.append(Block(text=text, kind=kind))

        # merge consecutive table blocks into one atomic table
        merged: list[Block] = []
        for b in pg.blocks:
            if merged and b.kind == "table" and merged[-1].kind == "table":
                merged[-1].text += "\n" + b.text
            else:
                merged.append(b)
        pg.blocks = merged
        if pg.dropped:
            log.warning("%s p.%d: dropped %s", path.name, pno, pg.dropped)
        pages.append(pg)
    return pages
