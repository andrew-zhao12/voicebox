"""Process lifecycle state: draining and readiness.

Shared by the lifespan (``app.py``), the generation queue, the readiness
route and the long-lived SSE loops.  Pure Python, so it stays importable in
the torch-free tests and the CLI tools.

Draining starts on the first SIGTERM/SIGINT (``install_signal_hooks`` wraps
the handlers uvicorn installed) or when the lifespan exits.  While draining,
``GET /health/ready`` answers 503, ``task_queue.ensure_capacity`` refuses new
jobs and the SSE loops end, so uvicorn's graceful shutdown finishes quickly
and the lifespan exit can wait for the jobs that are already running.

Readiness tracks the models named in ``VOICEBOX_PRELOAD_MODELS``: the server
is ready once every one of them is resident, the queue worker is alive and
no drain has started.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
from collections.abc import Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

DRAIN_TIMEOUT_ENV = "VOICEBOX_DRAIN_TIMEOUT_S"
DEFAULT_DRAIN_TIMEOUT_S = 30.0

_draining = False
_drain_reason: str | None = None
# signal number -> (our wrapper, the handler it replaced)
_hooked_signals: dict[int, tuple[Callable, Callable]] = {}

_preload_pending: dict[str, str] = {}
_preload_failed: dict[str, str] = {}
_preload_ready: list[str] = []


def drain_timeout_s(environ: Mapping[str, str] = os.environ) -> float:
    """Seconds the shutdown waits for queued and running jobs before cancelling them."""
    raw = environ.get(DRAIN_TIMEOUT_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_DRAIN_TIMEOUT_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("%s=%r is not a number; using %.0f", DRAIN_TIMEOUT_ENV, raw, DEFAULT_DRAIN_TIMEOUT_S)
        return DEFAULT_DRAIN_TIMEOUT_S


def begin_drain(reason: str) -> bool:
    """Start draining; returns False when a drain had already begun."""
    global _draining, _drain_reason
    if _draining:
        return False
    _draining = True
    _drain_reason = reason
    logger.info("Draining (%s): refusing new jobs, finishing the ones in flight", reason)
    return True


def is_draining() -> bool:
    return _draining


def drain_reason() -> str | None:
    return _drain_reason


def install_signal_hooks(signals: Iterable[int] = (signal.SIGTERM, signal.SIGINT)) -> list[str]:
    """Wrap the handlers already installed for *signals* so the first one also begins draining.

    uvicorn installs its own handlers before the lifespan starts and closes
    the listening socket as soon as one fires; this hook only adds the drain
    flag in front of that, so readiness flips and SSE loops end at once.
    Signals without a Python-level handler (``SIG_DFL``/``SIG_IGN``) are left
    alone.  Returns the names of the signals hooked.
    """
    if threading.current_thread() is not threading.main_thread():
        return []
    hooked: list[str] = []
    for sig in signals:
        if sig in _hooked_signals:
            continue
        previous = signal.getsignal(sig)
        if not callable(previous):
            continue

        def handler(signum: int, frame, _previous: Callable = previous) -> None:
            begin_drain(f"signal {signal.Signals(signum).name}")
            _previous(signum, frame)

        signal.signal(sig, handler)
        _hooked_signals[sig] = (handler, previous)
        hooked.append(signal.Signals(sig).name)
    return hooked


def restore_signal_hooks() -> None:
    """Put back the handlers ``install_signal_hooks`` replaced (when still ours)."""
    if threading.current_thread() is not threading.main_thread():
        return
    for sig, (handler, previous) in list(_hooked_signals.items()):
        if signal.getsignal(sig) is handler:
            signal.signal(sig, previous)
        _hooked_signals.pop(sig, None)


async def wait_for_idle(pending: Callable[[], int], timeout_s: float, *, poll_s: float = 0.25) -> bool:
    """Wait until ``pending()`` is zero; False when *timeout_s* elapses first."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while pending() > 0:
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(min(poll_s, max(0.0, deadline - loop.time())))
    return True


def preload_expect(names: Iterable[str]) -> None:
    """Register models that must be resident before the server reports ready."""
    for name in names:
        if name in _preload_ready:
            continue
        _preload_failed.pop(name, None)
        _preload_pending[name] = "queued"


def preload_phase(name: str, phase: str) -> None:
    _preload_pending[name] = phase


def preload_ready(name: str) -> None:
    _preload_pending.pop(name, None)
    _preload_failed.pop(name, None)
    if name not in _preload_ready:
        _preload_ready.append(name)


def preload_failed(name: str, error: str) -> None:
    _preload_pending.pop(name, None)
    _preload_failed[name] = error


def preload_state() -> dict:
    return {
        "ready": list(_preload_ready),
        "pending": dict(_preload_pending),
        "failed": dict(_preload_failed),
    }


def readiness(worker_alive: bool) -> tuple[bool, dict]:
    """``(ready, body)`` for ``GET /health/ready``.

    The body is public, so failed preloads are listed by name only; the
    error text is in the server log.
    """
    ready = worker_alive and not _draining and not _preload_pending and not _preload_failed
    body = {
        "ready": ready,
        "draining": _draining,
        "worker": worker_alive,
        "models": {
            "ready": list(_preload_ready),
            "pending": dict(_preload_pending),
            "failed": sorted(_preload_failed),
        },
    }
    return ready, body


def reset() -> None:
    """Forget every flag (tests, and forced re-initialisation)."""
    global _draining, _drain_reason
    restore_signal_hooks()
    _draining = False
    _drain_reason = None
    _preload_pending.clear()
    _preload_failed.clear()
    _preload_ready.clear()
