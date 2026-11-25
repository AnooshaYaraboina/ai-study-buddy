"""Rebuild readable lines from PDF layout.

PDF text comes out as *visual* lines: sentences wrap, centred captions are split
into separate blocks, a bold term sits on the same row as its definition, and a
small label sits above its description. This module turns raw lines into
*logical* lines, in reading order:

- wrapped lines are joined
  ``"...autoregressive models" + "generate one token at a time."``
- a short label is attached to its text
  ``"ENTRY" + "hostile content arrives: ..."`` -> ``"ENTRY: hostile content arrives: ..."``
  ``"Token / tokenizer" | "A token is ..."``   -> ``"Token / tokenizer: A token is ..."``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TERMINAL = (".", "!", "?", "…")
_JOINERS = {
    "and", "or", "of", "to", "the", "a", "an", "in", "for", "with", "on", "by", "from", "as",
    "at", "into", "that", "is", "are", "be", "not", "than", "via", "per", "its", "their",
    "your", "when", "while", "if", "so", "but", "whether", "which", "who", "can", "will",
}
# a line starting like this begins a new item; it never continues the previous line
_NEW_ITEM = re.compile(r"^(▸|Q:|A:|•|·|[-–]\s|\d{1,2}[.)]\s|\d{1,2}\s+[A-Z]|\(?[a-h]\)\s)")


@dataclass
class RawLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size_first: float
    size_last: float
    bold: bool
    block: int
    order: int
    horizontal: bool = True

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class LogicalLine:
    text: str
    lines: list[RawLine] = field(default_factory=list)
    labeled: bool = False

    @property
    def head(self) -> RawLine:
        return self.lines[0]

    @property
    def order(self) -> int:
        return self.head.order

    @property
    def block(self) -> int:
        return self.head.block

    @property
    def size(self) -> float:
        return self.head.size_first

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (min(r.x0 for r in self.lines), min(r.y0 for r in self.lines),
                max(r.x1 for r in self.lines), max(r.y1 for r in self.lines))


def _join(a: str, b: str) -> str:
    if re.search(r"[A-Za-z]-$", a) and b[:1].islower():  # hyphen at a wrap: "lower-" + "trust"
        return a + b
    return f"{a} {b}"


def _continues(a: RawLine, b: RawLine) -> bool:
    """Is ``b`` the wrapped continuation of ``a``?"""
    ta, tb = a.text, b.text
    if not (a.horizontal and b.horizontal) or ta.startswith("▸") or _NEW_ITEM.match(tb):
        return False
    if tb.isdigit() or ta.endswith(_TERMINAL) or (ta.endswith(":") and not tb[:1].islower()):
        return False
    if abs(a.size_last - b.size_first) > 0.6 or a.bold != b.bold:
        return False
    dy = b.y0 - a.y0
    if not 0.6 * a.size_last <= dy <= 2.0 * a.size_last:
        return False
    if not (abs(a.x0 - b.x0) <= 4 or abs(a.cx - b.cx) <= 8):
        return False
    words = ta.split()
    last = words[-1].lower().strip(",;") if words else ""
    return (tb[:1].islower() or tb[:1] in "(\"'“‘→" or ta.endswith((",", "-", "–", "+", "/", "&"))
            or last in _JOINERS)


def _chains(raw: list[RawLine]) -> list[LogicalLine]:
    """Greedy one-to-one continuation matching (closest pairs first)."""
    pairs = []
    for a in raw:
        for b in raw:
            if b is not a and b.y0 > a.y0 and _continues(a, b):
                pairs.append(((b.y0 - a.y0) + 0.5 * min(abs(a.x0 - b.x0), abs(a.cx - b.cx)),
                              a.order, b.order))
    nxt: dict[int, int] = {}
    prv: dict[int, int] = {}
    for _, ia, ib in sorted(pairs):
        if ia not in nxt and ib not in prv:
            nxt[ia], prv[ib] = ib, ia
    by_order = {r.order: r for r in raw}
    out = []
    for r in raw:
        if r.order in prv:
            continue
        chain = [r]
        while chain[-1].order in nxt:
            chain.append(by_order[nxt[chain[-1].order]])
        text = chain[0].text
        for c in chain[1:]:
            text = _join(text, c.text)
        out.append(LogicalLine(text=text, lines=chain))
    return out


def _label_like(a: LogicalLine, body_size: float) -> bool:
    t = a.text
    n = len(t.split())
    return (1 <= n <= 6 and len(t) <= 48 and len(a.lines) <= 2 and not a.labeled
            and not t.endswith(_TERMINAL) and not t.startswith(("▸", "Q:", "A:", "→", "•"))
            and not t.isdigit() and " · " not in t and a.lines[-1].size_last <= 1.45 * body_size)


def _merge_labels(lines: list[LogicalLine], body_size: float) -> list[LogicalLine]:
    """Attach short labels to their text: same row first (term | definition), then below."""
    consumed: set[int] = set()
    for mode in ("row", "below"):
        for a in lines:
            if id(a) in consumed or not _label_like(a, body_size):
                continue
            ah, at = a.head, a.lines[-1]
            best, best_cost = None, float("inf")
            for b in lines:
                if b is a or id(b) in consumed or b.labeled or b.text.startswith(("▸", "Q:")):
                    continue
                bh = b.head
                if mode == "row":  # "Term | definition" or "key | value" on one row
                    ok = abs(bh.y0 - ah.y0) <= 3 and 0 < bh.x0 - max(r.x1 for r in a.lines) <= 160
                    key_value = bh.block == ah.block and len(a.text.split()) <= 2
                    if not key_value and len(b.text.split()) < 2:
                        continue
                    style = key_value or (bh.size_first <= at.size_last + 0.3 and (
                        (at.bold and not bh.bold) or at.size_last >= bh.size_first + 1.2))
                    cost = bh.x0 - at.x1
                else:  # label above its text
                    if len(b.text.split()) < 2:
                        continue
                    dy = bh.y0 - at.y0
                    ok = (0.5 * at.size_last <= dy <= 2.6 * at.size_last
                          and bh.size_first <= at.size_last + 0.3
                          and (bh.x0 - 10 <= at.cx <= bh.x1 + 10 or abs(ah.x0 - bh.x0) <= 15))
                    style = (at.bold and not bh.bold) or (at.size_last >= bh.size_first + 1.2)
                    cost = dy
                if ok and style and cost < best_cost:
                    best, best_cost = b, cost
            if best is not None:
                a.text = f"{a.text.rstrip(': ')}: {best.text}"
                a.lines = a.lines + best.lines
                a.labeled = True
                consumed.add(id(best))
    return [ln for ln in lines if id(ln) not in consumed]


def assemble(raw: list[RawLine], body_size: float) -> list[LogicalLine]:
    """Raw visual lines (in reading order) -> logical lines (in reading order)."""
    lines = _merge_labels(_chains(raw), body_size)
    return sorted(lines, key=lambda ln: ln.order)
