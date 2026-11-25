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

from app.ingest.layout import RawLine, assemble
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


def _luma(rgb: tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _span_rgb(color: int) -> tuple[float, float, float]:
    return ((color >> 16) & 255) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255


def _background(bbox: pymupdf.Rect, fills: list[tuple[pymupdf.Rect, tuple]]) -> tuple:
    """Colour of the top-most filled shape under the span (page is white otherwise)."""
    bg = (1.0, 1.0, 1.0)
    for rect, fill in fills:  # drawing order: later shapes are on top
        if rect.contains(bbox) or rect.intersect(bbox).get_area() > 0.6 * bbox.get_area():
            bg = fill
    return bg


def _visible(span: dict, page_rect: pymupdf.Rect,
             fills: list[tuple[pymupdf.Rect, tuple]] | None = None) -> bool:
    """A human-visible span: readable size, opaque, on the page, contrasting colour."""
    if span.get("size", 10) < 4:
        return False
    if span.get("alpha", 255) == 0:
        return False
    bbox = pymupdf.Rect(span["bbox"])
    if not bbox.intersects(page_rect):
        return False
    if fills is not None:
        fg = _luma(_span_rgb(span.get("color", 0)))
        bg = _luma(_background(bbox, fills))
        if abs(fg - bg) < 0.08:  # text colour ~= background colour
            return False
    return True


def _page_lines(page: pymupdf.Page, contrast_check: bool = False) -> tuple[list[RawLine], int]:
    """Return visible lines (reading order) and the hidden-span count.

    ``contrast_check`` (untrusted uploads) also drops text whose colour matches the
    background under it, e.g. white-on-white instructions.
    """
    rect = page.rect
    fills: list | None = [] if contrast_check else None
    try:
        for d in (page.get_drawings() if contrast_check else []):
            if d.get("fill") is not None and d.get("fill_opacity", 1) > 0.5:
                fills.append((pymupdf.Rect(d["rect"]), tuple(d["fill"][:3])))
    except Exception:  # drawings are best-effort
        fills = [] if contrast_check else None
    out: list[RawLine] = []
    hidden = 0
    data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE, sort=True)
    for bno, block in enumerate(data["blocks"]):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            vis = [s for s in spans if _visible(s, rect, fills)]
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
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in vis)
            out.append(RawLine(text, x0, y0, x1, y1, vis[0]["size"], vis[-1]["size"], bold, bno,
                               len(out), horizontal=abs(line.get("dir", (1, 0))[1]) < 0.1))
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
        lines, hidden = _page_lines(page, contrast_check=untrusted)
        raw_pages.append((page, lines, hidden, _table_rects(page)))

    # Running headers / nav strips / strap lines repeated across the document
    counts = Counter()
    for _, lines, _, _ in raw_pages:
        counts.update({r.text for r in lines})
    n = max(1, len(raw_pages))
    # (>= 25% of pages and >= 3 pages: e.g. the handbook's "Understand why each building
    # block matters..." strap line, nav strips, "Four things to remember")
    boiler = {t for t, c in counts.items() if n >= 5 and c >= 3 and c / n >= 0.25}

    pages: list[Page] = []
    current_day = first_day
    for page, lines, hidden, tables in raw_pages:
        pno = page.number + 1
        texts = [r.text for r in lines]

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

        kept = [r for r in lines
                if r.text not in boiler and not (r.text.isdigit() and len(r.text) <= 3)]
        body = median([r.size_first for r in kept] or [10])
        pg = Page(file=path.name, page=pno, day=day, topic=topic)
        if hidden:
            pg.dropped.append(f"<{hidden} hidden spans>")

        groups: list[tuple[str, list]] = []  # (kind-hint, logical lines)
        for ln in assemble(kept, body):
            if untrusted and is_injection(ln.text):
                pg.dropped.append(ln.text)
                continue
            sentence = ln.text.endswith((".", "?", "!")) and len(ln.text.split()) >= 6
            is_heading = ln.text.startswith("▸") or (
                not ln.labeled and not sentence and ln.size >= body * 1.18 and len(ln.text) < 90)
            if is_heading:
                groups.append(("heading", [ln]))
            elif groups and groups[-1][0] == "body" and groups[-1][1][-1].block == ln.block:
                groups[-1][1].append(ln)
            else:
                groups.append(("body", [ln]))

        for hint, items in groups:
            text = "\n".join(ln.text for ln in items)
            if hint == "heading":
                kind, text = "heading", text.lstrip("▸ ").replace("\n", " ")
            elif any(r.contains(pymupdf.Rect(items[0].bbox)) for r in tables):
                kind = "table"
            elif sum(bool(LIST_RE.match(ln.text)) for ln in items) >= max(1, (len(items) + 1) // 2):
                kind = "list"
            else:
                kind = "para"
            pg.blocks.append(Block(text=text, kind=kind))

        # merge consecutive table / list blocks into one atomic block
        merged: list[Block] = []
        for b in pg.blocks:
            if merged and b.kind in ("table", "list") and merged[-1].kind == b.kind:
                merged[-1].text += "\n" + b.text
            else:
                merged.append(b)
        pg.blocks = merged
        if pg.dropped:
            (log.warning if untrusted else log.debug)("%s p.%d: dropped %s", path.name, pno,
                                                      pg.dropped)
        pages.append(pg)
    return pages
