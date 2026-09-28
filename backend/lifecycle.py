"""Process lifecycle state: draining and readiness.

Shared by the lifespan (``app.py``), the generation queue, the readiness
route and the long-lived SSE loops.  Pure Python, so it stays importable in
the torch-free tests and the CLI tools.

Draining starts on the first SIGTERM/SIGINT (``install_signal_hooks`` wraps
the handlers uvicorn installed) or when the lifespan exits.  While draining,
``GET /health/ready`` answers 503, ``task_queue.ensure_capacity`` refuses new
jobs and the SSE loops end, so uvicorn's graceful shutdown finishes quickly
and the lifespan exit can wait for the jobs that are already running.

With ``VOICEBOX_SHUTDOWN_DELAY_S`` set, the first signal starts a "stopping"
phase instead: readiness answers 503 at once but requests are still accepted
and served for that many seconds, so a load balancer polling
``/health/ready`` stops routing here before uvicorn closes the socket.  Only
then does the drain above begin.  A second signal skips the rest of the delay.

Readiness tracks the models named in ``VOICEBOX_PRELOAD_MODELS`` and the
startup steps registered with ``step_begin`` (the voice seed, the GPU check):
the server is ready once every model is resident, every step finished, the
queue worker is alive and no drain has started.
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
SHUTDOWN_DELAY_ENV = "VOICEBOX_SHUTDOWN_DELAY_S"
MAX_SHUTDOWN_DELAY_S = 120.0

_draining = False
_drain_reason: str | None = None
# Between the first signal and the drain when VOICEBOX_SHUTDOWN_DELAY_S is set.
_stopping = False
# Set once uvicorn's own handler has been called, so the delayed call never
# runs it a second time (uvicorn treats a second call as "force exit").
_exit_forwarded = False
# signal number -> (our wrapper, the handler it replaced)
_hooked_signals: dict[int, tuple[Callable, Callable]] = {}

_preload_pending: dict[str, str] = {}
_preload_failed: dict[str, str] = {}
_preload_ready: list[str] = []

# Startup steps other than model loads (seeded voices, the GPU check).
_step_pending: dict[str, str] = {}
_step_failed: dict[str, str] = {}
_step_done: list[str] = []


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


def shutdown_delay_s(environ: Mapping[str, str] = os.environ) -> float:
    """Seconds between the first SIGTERM and closing the listener (0 = stop at once, the default)."""
    raw = environ.get(SHUTDOWN_DELAY_ENV)
    if raw is None or not raw.strip():
        return 0.0
    try:
        return min(MAX_SHUTDOWN_DELAY_S, max(0.0, float(raw)))
    except ValueError:
        logger.warning("%s=%r is not a number; stopping without a delay", SHUTDOWN_DELAY_ENV, raw)
        return 0.0


def begin_stopping(reason: str, delay_s: float) -> bool:
    """Fail readiness but keep serving; returns False when stopping or draining had already begun."""
    global _stopping
    if _stopping or _draining:
        return False
    _stopping = True
    logger.info("Stopping in %.0f s (%s): readiness answers 503, requests are still served", delay_s, reason)
    return True


def is_stopping() -> bool:
    """True from the first signal on (the delay phase and the drain)."""
    return _stopping or _draining


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


def _forward_exit(previous: Callable, signum: int, frame, reason: str) -> None:
    """Begin the drain and hand the signal to uvicorn's handler, once."""
    global _exit_forwarded
    begin_drain(reason)
    if _exit_forwarded:
        return
    _exit_forwarded = True
    previous(signum, frame)


def install_signal_hooks(
    signals: Iterable[int] = (signal.SIGTERM, signal.SIGINT),
    *,
    delay_s: float | None = None,
) -> list[str]:
    """Wrap the handlers already installed for *signals* so the first one also begins draining.

    uvicorn installs its own handlers before the lifespan starts and closes
    the listening socket as soon as one fires; this hook adds the drain flag
    in front of that, so readiness flips and SSE loops end at once.  With a
    shutdown delay (``VOICEBOX_SHUTDOWN_DELAY_S``, or *delay_s*) and a running
    event loop, the first signal only fails readiness and uvicorn's handler
    runs *delay_s* seconds later; a second signal forwards at once.  Signals
    without a Python-level handler (``SIG_DFL``/``SIG_IGN``) are left alone.
    Returns the names of the signals hooked.
    """
    if threading.current_thread() is not threading.main_thread():
        return []
    delay = shutdown_delay_s() if delay_s is None else max(0.0, delay_s)
    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    hooked: list[str] = []
    for sig in signals:
        if sig in _hooked_signals:
            continue
        previous = signal.getsignal(sig)
        if not callable(previous):
            continue

        def handler(signum: int, frame, _previous: Callable = previous) -> None:
            reason = f"signal {signal.Signals(signum).name}"
            if delay > 0 and loop is not None and not loop.is_closed() and begin_stopping(reason, delay):
                # Signal handlers run between bytecodes of the loop's thread;
                # call_soon_threadsafe is the signal-safe way into the loop.
                loop.call_soon_threadsafe(loop.call_later, delay, _forward_exit, _previous, signum, None, reason)
                return
            _forward_exit(_previous, signum, frame, reason)

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


def step_begin(name: str, phase: str = "running") -> None:
    """Register a startup step that must finish before the server reports ready."""
    _step_failed.pop(name, None)
    if name in _step_done:
        _step_done.remove(name)
    _step_pending[name] = phase


def step_done(name: str) -> None:
    _step_pending.pop(name, None)
    _step_failed.pop(name, None)
    if name not in _step_done:
        _step_done.append(name)


def step_failed(name: str, error: str) -> None:
    _step_pending.pop(name, None)
    _step_failed[name] = error


def steps_state() -> dict:
    return {"done": list(_step_done), "pending": dict(_step_pending), "failed": dict(_step_failed)}


def readiness(worker_alive: bool) -> tuple[bool, dict]:
    """``(ready, body)`` for ``GET /health/ready``.

    The body is public, so failed preloads and steps are listed by name
    only; the error text is in the server log.
    """
    ready = (
        worker_alive
        and not _draining
        and not _stopping
        and not _preload_pending
        and not _preload_failed
        and not _step_pending
        and not _step_failed
    )
    body = {
        "ready": ready,
        "draining": _draining,
        "stopping": _stopping or _draining,
        "worker": worker_alive,
        "models": {
            "ready": list(_preload_ready),
            "pending": dict(_preload_pending),
            "failed": sorted(_preload_failed),
        },
        "startup": {
            "done": list(_step_done),
            "pending": dict(_step_pending),
            "failed": sorted(_step_failed),
        },
    }
    return ready, body


def reset() -> None:
    """Forget every flag (tests, and forced re-initialisation)."""
    global _draining, _drain_reason, _stopping, _exit_forwarded
    restore_signal_hooks()
    _draining = False
    _drain_reason = None
    _stopping = False
    _exit_forwarded = False
    _preload_pending.clear()
    _preload_failed.clear()
    _preload_ready.clear()
    _step_pending.clear()
    _step_failed.clear()
    _step_done.clear()
