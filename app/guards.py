"""Security guardrails used by the FastAPI backend.

- auth: bearer token -> student_id (each student sees only their own data)
- rate limit: sliding window per student
- size limits: message length, upload size, PDF magic bytes, page count
- uploaded PDFs are ingested as *untrusted*: hidden spans and instruction-like
  lines are dropped (see app/ingest/sanitize.py), and chunks are owner-scoped.
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, status

from app.config import get_settings

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class GuardError(HTTPException):
    pass


# ------------------------------------------------------------------ auth
def current_student(authorization: str = Header(default="")) -> str:
    """FastAPI dependency: 'Authorization: Bearer <token>' -> student_id."""
    scheme, _, token = authorization.partition(" ")
    student = get_settings().api_tokens.get(token.strip()) if scheme.lower() == "bearer" else None
    if not student:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    return student


# ------------------------------------------------------------------ rate limit
class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                retry = int(60 - (now - q[0])) + 1
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                                    f"rate limit: {self.per_minute}/min",
                                    headers={"Retry-After": str(retry)})
            q.append(now)


# ------------------------------------------------------------------ inputs
def clean_message(text: str) -> str:
    text = _CTRL.sub("", text or "").strip()
    if not text:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "empty message")
    limit = get_settings().max_message_chars
    if len(text) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"message longer than {limit} characters")
    return text


def clean_session(session_id: str | None) -> str:
    sid = (session_id or "default").strip()
    if not _SESSION_RE.match(sid):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid session_id")
    return sid


def check_upload(filename: str, data: bytes) -> str:
    s = get_settings()
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"file larger than {s.max_upload_mb} MB")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "only PDF files are accepted")
    name = _SAFE_NAME.sub("_", (filename or "upload.pdf").rsplit("/", 1)[-1])[:100]
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def public_quiz(quiz: list[dict] | None) -> list[dict] | None:
    """Strip answers/explanations before a quiz leaves the backend."""
    if not quiz:
        return None
    return [{"question": q["question"], "options": q["options"], "topic": q.get("topic", "")}
            for q in quiz]
