"""Prometheus metrics, when ``prometheus_client`` is installed; no-ops otherwise.

Counters are updated where the work happens (queue, generation jobs,
inference slots, rate limiter, the request-id middleware), so they cover
every entry point: REST, ``/v1``, MCP.  ``GET /metrics`` (admin key) renders
them, and ``VOICEBOX_METRICS_PORT`` starts an unauthenticated exporter for
scrapers that cannot send a bearer; bind that port to a private network only.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        start_http_server,
    )
    from prometheus_client.core import GaugeMetricFamily

    ENABLED = True
except ImportError:  # pragma: no cover - exercised only where the package is absent
    ENABLED = False


class _Noop:
    def labels(self, *_args, **_kwargs) -> _Noop:
        return self

    def inc(self, *_args, **_kwargs) -> None:
        pass

    def observe(self, *_args, **_kwargs) -> None:
        pass

    def set(self, *_args, **_kwargs) -> None:
        pass


_SECONDS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300)


def _counter(name: str, doc: str, labels: tuple[str, ...] = ()):
    return Counter(name, doc, labels) if ENABLED else _Noop()


def _gauge(name: str, doc: str, labels: tuple[str, ...] = ()):
    return Gauge(name, doc, labels) if ENABLED else _Noop()


def _histogram(name: str, doc: str, labels: tuple[str, ...] = (), buckets=_SECONDS):
    return Histogram(name, doc, labels, buckets=buckets) if ENABLED else _Noop()


HTTP_REQUESTS = _counter(
    "voicebox_http_requests_total", "HTTP requests by route and status", ("method", "route", "status")
)
HTTP_SECONDS = _histogram("voicebox_http_request_seconds", "HTTP request duration", ("method", "route"))
QUEUE_PENDING = _gauge("voicebox_queue_pending_jobs", "Generation jobs queued or running")
QUEUE_WAIT_SECONDS = _histogram("voicebox_queue_wait_seconds", "Time a job waited in the queue before starting")
GENERATION_SECONDS = _histogram("voicebox_generation_seconds", "Synthesis job duration", ("engine", "kind"))
GENERATIONS = _counter("voicebox_generations_total", "Synthesis jobs by outcome", ("engine", "kind", "status"))
TTS_CHARACTERS = _counter("voicebox_tts_characters_total", "Characters synthesized", ("engine",))
STREAM_FIRST_CHUNK_SECONDS = _histogram(
    "voicebox_stream_first_chunk_seconds", "Enqueue to first streamed chunk", ("engine",)
)
SLOT_WAIT_SECONDS = _histogram("voicebox_inference_slot_wait_seconds", "Wait for the Whisper/LLM slot", ("slot",))
MODEL_LOAD_SECONDS = _histogram(
    "voicebox_model_load_seconds",
    "Model load time (download included)",
    ("model",),
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800),
)
RATE_LIMITED = _counter("voicebox_rate_limited_total", "Requests refused by a limit", ("dimension",))


def observe_http(method: str, route: str, status: int, seconds: float) -> None:
    HTTP_REQUESTS.labels(method, route, str(status)).inc()
    HTTP_SECONDS.labels(method, route).observe(seconds)


class _GpuCollector:
    """GPU memory at scrape time, without importing torch until then."""

    def collect(self):
        try:
            import torch  # lazy: heavy import
        except Exception:
            return
        if not torch.cuda.is_available():
            return
        family = GaugeMetricFamily("voicebox_gpu_memory_bytes", "CUDA memory by kind", labels=["kind"])
        family.add_metric(["allocated"], float(torch.cuda.memory_allocated()))
        family.add_metric(["reserved"], float(torch.cuda.memory_reserved()))
        yield family


if ENABLED:
    REGISTRY.register(_GpuCollector())


def render() -> tuple[bytes, str]:
    """``(body, content_type)`` for ``GET /metrics``; raises ``RuntimeError`` when the package is absent."""
    if not ENABLED:
        raise RuntimeError("prometheus_client is not installed")
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def start_exporter(port: int, addr: str = "0.0.0.0") -> bool:
    """Serve the registry on its own port (no auth); returns False when metrics are unavailable."""
    if not ENABLED:
        logger.warning("VOICEBOX_METRICS_PORT is set but prometheus_client is not installed")
        return False
    start_http_server(port, addr=addr)
    logger.info("Metrics exporter listening on %s:%d", addr, port)
    return True
