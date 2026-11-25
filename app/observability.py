"""Tracing: OpenTelemetry spans for every request, exported to

- Langfuse (OTLP endpoint ``{LANGFUSE_HOST}/api/public/otel``) when keys are set,
- any OTLP collector when OTEL_EXPORTER_OTLP_ENDPOINT is set,
- always: a local JSONL file ``logs/traces.jsonl`` (handy for tests + demos).
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from typing import Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from app.config import get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()
_configured = False


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with _lock, open(self.path, "a", encoding="utf-8") as f:
            for s in spans:
                f.write(json.dumps({
                    "name": s.name,
                    "trace_id": format(s.context.trace_id, "032x"),
                    "span_id": format(s.context.span_id, "016x"),
                    "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
                    "start": s.start_time,
                    "duration_ms": round((s.end_time - s.start_time) / 1e6, 2),
                    "status": s.status.status_code.name,
                    "attributes": dict(s.attributes or {}),
                }, default=str) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass


def setup_tracing(service: str = "study-buddy-api") -> None:
    global _configured
    if _configured:
        return
    s = get_settings()
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(s.log_dir / "traces.jsonl")))
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        if s.langfuse_public_key and s.langfuse_secret_key:
            auth = base64.b64encode(
                f"{s.langfuse_public_key}:{s.langfuse_secret_key}".encode()
            ).decode()
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                endpoint=f"{s.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
                headers={"Authorization": f"Basic {auth}"},
            )))
            log.info("tracing -> Langfuse %s", s.langfuse_host)
        if s.otlp_endpoint:
            provider.add_span_processor(BatchSpanProcessor(
                OTLPSpanExporter(endpoint=f"{s.otlp_endpoint.rstrip('/')}/v1/traces")
            ))
            log.info("tracing -> OTLP %s", s.otlp_endpoint)
    except Exception as e:  # exporters are optional
        log.warning("OTLP exporter unavailable: %s", e)
    trace.set_tracer_provider(provider)
    _configured = True


def tracer() -> trace.Tracer:
    return trace.get_tracer("study-buddy")
