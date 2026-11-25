"""Guardrails for untrusted PDF text.

Uploaded PDFs are untrusted. Two layers:
1. Layout layer (in pdf_loader): spans that a human cannot see -- tiny font,
   fully transparent, or positioned off the page -- are dropped.
2. Text layer (here): any line that reads like an instruction aimed at the
   model ("ignore previous instructions", "you are now ...") is dropped,
   whether or not it was visible.
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|messages?)",
    r"\bdisregard\s+(all\s+|the\s+|any\s+)?(previous|prior|above|system)\b",
    r"\bforget\s+(everything|all|your)\s+(you|instructions?|rules?)",
    r"\byou\s+are\s+now\s+(a|an|the|in)\b",
    r"\b(reveal|print|show|leak|output)\s+(the\s+|your\s+)?(system\s+prompt|hidden\s+prompt|api\s*key|secrets?)",
    r"\bnew\s+(system\s+)?instructions?\s*:",
    r"^\s*(system|assistant)\s*:",
    r"<\s*/?\s*(system|instructions?)\s*>",
    r"\bact\s+as\s+(an?\s+)?(unrestricted|jailbroken|dan)\b",
    r"\bdo\s+not\s+(tell|inform)\s+the\s+(user|student)",
    r"\b(send|email|post|upload)\s+.{0,40}\b(to|at)\s+https?://",
    r"\bwhen\s+(asked|answering)\b.{0,60}\b(always|instead)\s+(say|answer|reply|respond)",
    r"\balways\s+(answer|say|reply|respond|tell\s+(the\s+)?(user|student))\s+(that|with)\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Zero-width / bidi control characters are a common way to hide payloads.
_INVISIBLE_CHARS = re.compile("[​-‏‪-‮⁠-⁤﻿]")


def is_injection(line: str) -> bool:
    """True if the line looks like an instruction aimed at the model."""
    return bool(_INJECTION_RE.search(line))


def clean_text(text: str) -> str:
    """Strip invisible characters and normalise whitespace (keeps newlines)."""
    text = _INVISIBLE_CHARS.sub("", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def sanitize_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    """Return (kept, dropped) lines."""
    kept, dropped = [], []
    for ln in lines:
        ln = clean_text(ln)
        if not ln:
            continue
        (dropped if is_injection(ln) else kept).append(ln)
    return kept, dropped
