"""Prometheus metrics for the Laya Decision Studio.

Metrics live on the default registry and are updated in place by the app:

* ``laya_studio_http_requests_total`` — every HTTP request, by method,
  endpoint template and status code;
* ``laya_studio_http_request_duration_seconds`` — whole-request wall time;
* ``laya_studio_inference_seconds`` — forward-pass wall time behind the
  asyncio gate, by checkpoint;
* ``laya_studio_inference_input_tokens_total`` ·
  ``laya_studio_inference_output_tokens_total`` — token counters by
  checkpoint (no generation: output is usually 0);
* ``laya_studio_decisions_total`` — decision results, by the checkpoint
  that actually answered (``routing.model``, not what was asked for);
* ``laya_studio_inference_in_flight`` — forward passes currently running.

Endpoint labels are the fixed API paths, ``unmatched`` (unknown ``/api/``
and ``/v1/`` paths) or ``static`` — never raw paths, so scrapes and 404s
cannot blow up label cardinality. Set ``LAYA_METRICS=0`` to stop updates
and serve 404 from ``/metrics``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

from prometheus_client import Counter, Gauge, Histogram

_OFF_VALUES = ("0", "false", "no", "off")

# Wall-time buckets: inference is ~tens of ms on CPU and the synchronous
# wait is capped at 45 s, so the last bucket is the documented bound.
_SECONDS_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 45.0)

HTTP_REQUESTS = Counter(
    "laya_studio_http_requests_total",
    "HTTP requests answered, by method, endpoint template and status code.",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_SECONDS = Histogram(
    "laya_studio_http_request_duration_seconds",
    "Whole-request wall time, by method and endpoint template.",
    ["method", "endpoint"],
    buckets=_SECONDS_BUCKETS,
)

INFERENCE_SECONDS = Histogram(
    "laya_studio_inference_seconds",
    "Forward-pass wall time behind the asyncio gate, by checkpoint.",
    ["checkpoint"],
    buckets=_SECONDS_BUCKETS,
)

INPUT_TOKENS = Counter(
    "laya_studio_inference_input_tokens_total",
    "Input tokens fed to the checkpoints, by checkpoint.",
    ["checkpoint"],
)

OUTPUT_TOKENS = Counter(
    "laya_studio_inference_output_tokens_total",
    "Output tokens produced by the checkpoints, by checkpoint "
    "(no generation: usually 0).",
    ["checkpoint"],
)

DECISIONS = Counter(
    "laya_studio_decisions_total",
    "Decision results produced, by the checkpoint that actually answered.",
    ["checkpoint"],
)

INFERENCE_IN_FLIGHT = Gauge(
    "laya_studio_inference_in_flight",
    "Forward passes currently running (the asyncio gate allows one).",
)


def enabled() -> bool:
    """Metrics are on unless ``LAYA_METRICS`` disables them."""
    return os.environ.get("LAYA_METRICS", "1").strip().lower() not in _OFF_VALUES


def inference_started() -> None:
    if enabled():
        INFERENCE_IN_FLIGHT.inc()


def inference_finished() -> None:
    if enabled():
        INFERENCE_IN_FLIGHT.dec()


def record_http(method: str, endpoint: str, status: int, seconds: float) -> None:
    """One answered request: a count plus its wall-time observation."""
    if not enabled():
        return
    HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    HTTP_REQUEST_SECONDS.labels(method=method, endpoint=endpoint).observe(seconds)


def record_single(result: Dict[str, Any], infer_seconds: float) -> None:
    """One single decision: pass duration plus its one result."""
    if not enabled():
        return
    _record_pass([result], infer_seconds)


def record_batch(results: Iterable[Dict[str, Any]], infer_seconds: float) -> None:
    """One batch decision. Tokens and decisions are exact per result; the
    pass duration is observed once per distinct checkpoint the batch
    answered, so a cross-checkpoint batch counts once per checkpoint."""
    if not enabled():
        return
    _record_pass(list(results), infer_seconds)


def _record_pass(results: List[Dict[str, Any]], infer_seconds: float) -> None:
    for checkpoint in sorted({_result_checkpoint(r) for r in results}):
        INFERENCE_SECONDS.labels(checkpoint=checkpoint).observe(infer_seconds)
    for result in results:
        _record_result(result)


def _record_result(result: Dict[str, Any]) -> None:
    checkpoint = _result_checkpoint(result)
    usage = result.get("usage") or {}
    DECISIONS.labels(checkpoint=checkpoint).inc()
    INPUT_TOKENS.labels(checkpoint=checkpoint).inc(_tokens(usage.get("input_tokens")))
    OUTPUT_TOKENS.labels(checkpoint=checkpoint).inc(_tokens(usage.get("output_tokens")))


def _result_checkpoint(result: Dict[str, Any]) -> str:
    """The checkpoint that actually answered, falling back to ``unknown``."""
    routing = result.get("routing") or {}
    model = routing.get("model") or result.get("model")
    return str(model) if model else "unknown"


def _tokens(value: Any) -> int:
    """Clamped to >= 0: a malformed usage never makes a counter go backwards."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
