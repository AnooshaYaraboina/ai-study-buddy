"""FastAPI backend: auth · guards · rate limit · SSE streaming · traces.

    uvicorn app.api.main:app --port 8080
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app import db
from app.agents.graph import run_turn, stream_turn
from app.config import get_settings
from app.guards import (
    RateLimiter,
    check_upload,
    clean_message,
    clean_session,
    current_student,
    public_quiz,
)
from app.ingest.pipeline import ingest_dir, ingest_file
from app.llm import get_llm
from app.observability import setup_tracing, tracer
from app.vector_store import get_store

log = logging.getLogger("api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
settings = get_settings()
limiter = RateLimiter(settings.rate_limit_per_min)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing("study-buddy-api")
    if os.getenv("INGEST_ON_START", "1") == "1" and get_store().count() == 0:
        log.info("collection empty -> ingesting %s", settings.pdf_dir)
        await run_in_threadpool(ingest_dir)
    # warm models so the first student request is fast
    from app.retrieval import get_retriever

    await run_in_threadpool(get_retriever().search, "warm up", 1)
    log.info("ready: %d chunks, llm=%s offline=%s", get_store().count(),
             get_llm().model, get_llm().offline)
    yield


app = FastAPI(title="AI Study Buddy API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    t0 = time.perf_counter()
    with tracer().start_as_current_span(f"{request.method} {request.url.path}") as span:
        span.set_attributes({"http.method": request.method, "http.route": request.url.path,
                             "request.id": rid})
        response = await call_next(request)
        span.set_attributes({"http.status_code": response.status_code,
                             "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
        ctx = span.get_span_context()
        response.headers["x-request-id"] = rid
        response.headers["x-trace-id"] = format(ctx.trace_id, "032x")
        return response


def guarded(student: str = Depends(current_student)) -> str:
    limiter.check(student)
    trace.get_current_span().set_attribute("student.id", student)
    return student


class ChatIn(BaseModel):
    message: str = Field(max_length=10_000)
    session_id: str | None = None


def _out(state: dict) -> dict:
    return {
        "route": state.get("route"),
        "reply": state.get("reply", ""),
        "citations": state.get("citations", []),
        "sources": state.get("sources", []),
        "refused": state.get("refused", False),
        "quiz": public_quiz(state.get("quiz")),
        "grade": state.get("grade"),
        "weak_topics": state.get("weak_topics", []),
        "standalone": state.get("standalone"),
    }


@app.get("/health")
def health() -> dict:
    llm = get_llm()
    return {"status": "ok", "chunks": get_store().count(), "llm_provider": llm.provider,
            "llm_model": llm.model, "llm_offline": llm.offline}


@app.post("/chat")
def chat(body: ChatIn, student: str = Depends(guarded)) -> dict:
    state = run_turn(student, clean_session(body.session_id), clean_message(body.message))
    return _out(state)


@app.post("/chat/stream")
def chat_stream(body: ChatIn, student: str = Depends(guarded)) -> StreamingResponse:
    session = clean_session(body.session_id)
    message = clean_message(body.message)
    parent = trace.set_span_in_context(trace.get_current_span())

    def events():
        with tracer().start_as_current_span("chat.stream", context=parent):
            try:
                for ev in stream_turn(student, session, message):
                    if ev["type"] == "final":
                        ev = {"type": "final", **_out(ev["state"])}
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            except Exception as e:  # surface errors to the client instead of a broken stream
                log.exception("stream failed")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)[:300]})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/upload")
async def upload(file: UploadFile = File(...), student: str = Depends(guarded)) -> dict:
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    name = check_upload(file.filename or "upload.pdf", data)
    dest = settings.upload_dir / student
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / name
    path.write_bytes(data)
    try:
        rep = await run_in_threadpool(ingest_file, path, None, student, "upload")
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    from app.retrieval import get_retriever

    getattr(get_retriever(), "invalidate", lambda: None)()
    return {"file": rep.file, "pages": rep.pages, "chunks": rep.chunks,
            "dropped_hidden_or_injected": rep.dropped_lines, "visible_to": student}


@app.get("/progress")
def progress(student: str = Depends(guarded)) -> dict:
    return db.progress(student)


@app.get("/flashcards")
def flashcards(student: str = Depends(guarded)) -> list[dict]:
    return db.list_flashcards(student)


@app.get("/usage")
def usage(student: str = Depends(guarded)) -> dict:
    """LLM token + cost totals for this API process."""
    return get_llm().totals


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):  # pragma: no cover
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "internal error"}, status_code=500)
