"""One LLM client wrapper (Anthropic Claude).

- streaming answers (``stream``) and one-shot calls (``complete``)
- timeout + ``LLM_MAX_RETRIES`` (default 2) retries with exponential backoff
- token + cost logged for every request (logs/llm_usage.jsonl + span attrs)
- server-side refusal fallback opted in
- offline mode (no key / LLM_OFFLINE=1): each caller passes a deterministic
  ``offline`` fallback so the whole app, tests and CI run without a key.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import anthropic

from app.config import get_settings
from app.observability import tracer

log = logging.getLogger(__name__)

# $ per 1M tokens (input, output)
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-2": (8.0, 40.0),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
RETRYABLE = (
    anthropic.APIConnectionError,  # includes APITimeoutError
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

_usage_lock = threading.Lock()


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    offline: bool = False
    stop_reason: str | None = None
    raw: Any = field(default=None, repr=False)


def cost_of(model: str, in_tok: int, out_tok: int, cache_read: int = 0) -> float:
    pin, pout = PRICES.get(model, (2.0, 10.0))
    return round((in_tok * pin + cache_read * pin * 0.1 + out_tok * pout) / 1e6, 6)


class LLM:
    def __init__(self) -> None:
        s = get_settings()
        self.settings = s
        self.model = s.llm_model
        self.offline = s.llm_offline
        self.totals = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        self._client = None
        if not self.offline:
            # we own the retry loop (explicit backoff + logging), so SDK retries = 0
            self._client = anthropic.Anthropic(
                api_key=s.anthropic_api_key, timeout=s.llm_timeout_s, max_retries=0
            )

    # ------------------------------------------------------------ helpers
    def _params(self, system: str, messages: list[dict], max_tokens: int,
                json_schema: dict | None, effort: str | None) -> dict:
        output_config: dict[str, Any] = {"effort": effort or self.settings.llm_effort}
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        params: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config=output_config,
        )
        if self.model in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5",
                          "claude-fable-5-2", "claude-fable-5-1"):
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def _log_usage(self, task: str, res: LLMResult, latency_ms: float, attempt: int) -> None:
        rec = {
            "ts": time.time(), "task": task, "model": res.model, "offline": res.offline,
            "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
            "cost_usd": res.cost_usd, "latency_ms": round(latency_ms, 1), "attempts": attempt,
            "stop_reason": res.stop_reason,
        }
        with _usage_lock:
            self.totals["requests"] += 1
            self.totals["input_tokens"] += res.input_tokens
            self.totals["output_tokens"] += res.output_tokens
            self.totals["cost_usd"] = round(self.totals["cost_usd"] + res.cost_usd, 6)
            path = self.settings.log_dir / "llm_usage.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        log.info("llm %s tokens=%d/%d cost=$%.5f %.0fms", task, res.input_tokens,
                 res.output_tokens, res.cost_usd, latency_ms)

    def _result_from(self, msg: Any) -> LLMResult:
        if msg.stop_reason == "refusal":
            log.warning("LLM refusal: %s", getattr(msg, "stop_details", None))
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        u = msg.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        return LLMResult(
            text=text,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cost_usd=cost_of(msg.model, u.input_tokens, u.output_tokens, cache_read),
            model=msg.model,
            stop_reason=msg.stop_reason,
            raw=msg,
        )

    def _backoff(self, attempt: int, err: Exception) -> None:
        delay = min(8.0, 0.8 * 2 ** (attempt - 1)) + random.uniform(0, 0.3)
        log.warning("LLM call failed (%s); retry %d in %.1fs", type(err).__name__, attempt, delay)
        time.sleep(delay)

    # ------------------------------------------------------------ public
    def complete(
        self,
        system: str,
        messages: list[dict],
        *,
        task: str,
        offline: Callable[[], str],
        max_tokens: int = 4000,
        json_schema: dict | None = None,
        effort: str | None = None,
    ) -> LLMResult:
        with tracer().start_as_current_span(f"llm.{task}") as span:
            t0 = time.perf_counter()
            if self.offline:
                res = LLMResult(text=offline(), model="offline-stub", offline=True)
                self._log_usage(task, res, (time.perf_counter() - t0) * 1000, 1)
                span.set_attributes({"llm.offline": True})
                return res
            params = self._params(system, messages, max_tokens, json_schema, effort)
            for attempt in range(1, self.settings.llm_max_retries + 2):
                try:
                    with self._client.beta.messages.stream(**params) as stream:
                        msg = stream.get_final_message()
                    res = self._result_from(msg)
                    break
                except RETRYABLE as e:
                    if attempt > self.settings.llm_max_retries:
                        span.record_exception(e)
                        raise
                    self._backoff(attempt, e)
            self._log_usage(task, res, (time.perf_counter() - t0) * 1000, attempt)
            span.set_attributes({
                "llm.model": res.model, "llm.input_tokens": res.input_tokens,
                "llm.output_tokens": res.output_tokens, "llm.cost_usd": res.cost_usd,
                "llm.attempts": attempt,
            })
            return res

    def stream(
        self,
        system: str,
        messages: list[dict],
        *,
        task: str,
        offline: Callable[[], str],
        max_tokens: int = 4000,
        effort: str | None = None,
        on_done: Callable[[LLMResult], None] | None = None,
    ) -> Iterator[str]:
        """Yield text deltas. Retries only if nothing was yielded yet."""
        t0 = time.perf_counter()
        if self.offline:
            text = offline()
            for i in range(0, len(text), 24):
                yield text[i : i + 24]
            res = LLMResult(text=text, model="offline-stub", offline=True)
            self._log_usage(task, res, (time.perf_counter() - t0) * 1000, 1)
            if on_done:
                on_done(res)
            return
        params = self._params(system, messages, max_tokens, None, effort)
        with tracer().start_as_current_span(f"llm.{task}.stream") as span:
            for attempt in range(1, self.settings.llm_max_retries + 2):
                yielded = False
                try:
                    with self._client.beta.messages.stream(**params) as stream:
                        for delta in stream.text_stream:
                            yielded = True
                            yield delta
                        msg = stream.get_final_message()
                    res = self._result_from(msg)
                    break
                except RETRYABLE as e:
                    if yielded or attempt > self.settings.llm_max_retries:
                        span.record_exception(e)
                        raise
                    self._backoff(attempt, e)
            self._log_usage(task, res, (time.perf_counter() - t0) * 1000, attempt)
            span.set_attributes({
                "llm.model": res.model, "llm.input_tokens": res.input_tokens,
                "llm.output_tokens": res.output_tokens, "llm.cost_usd": res.cost_usd,
            })
            if on_done:
                on_done(res)


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


def reset_llm() -> None:
    global _llm
    _llm = None
