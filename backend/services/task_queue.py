"""
Serial generation queue — ensures only one TTS inference runs at a time
to avoid GPU contention.
"""

import asyncio
import contextlib
import logging
import traceback
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Literal

from .. import lifecycle

logger = logging.getLogger(__name__)

# Keep references to fire-and-forget background tasks to prevent GC
_background_tasks: set = set()

# Job ids of streaming generations (``POST /generate/stream``). They share the
# queue with regular generations but have no ``generations`` row.
STREAM_JOB_PREFIX = "stream-"
# Job ids of startup preloads (``services/preload.py``); no ``generations`` row either.
PRELOAD_JOB_PREFIX = "preload-"


class QueueFullError(Exception):
    """Raised by ``enqueue_generation`` when a pending cap is reached or the server is draining."""

    def __init__(self, reason: Literal["global", "owner", "draining"], retry_after_s: int = 5) -> None:
        if reason == "draining":
            message = f"Server is shutting down; retry in {retry_after_s} s"
        else:
            message = f"Generation queue is full ({reason} limit); retry in {retry_after_s} s"
        super().__init__(message)
        self.reason = reason
        self.retry_after_s = retry_after_s


@dataclass
class GenerationJob:
    """Queued generation work plus the generation ID it belongs to."""

    generation_id: str
    coro: Coroutine
    owner: str | None = None


DEFAULT_MAX_DEPTH = 32

# Generation queue — serializes TTS inference to avoid GPU contention
_generation_queue: asyncio.Queue = None  # type: ignore  # initialized at startup
_generation_worker_task: asyncio.Task | None = None
_queued_generation_ids: set[str] = set()
_running_generation_tasks: dict[str, asyncio.Task] = {}
_cancelled_generation_ids: set[str] = set()
_max_depth: int = DEFAULT_MAX_DEPTH
# Pending (queued + running) jobs per owner (API key id), for per-key caps.
_pending_by_owner: dict[str, int] = {}
_job_owner: dict[str, str] = {}


def create_background_task(coro) -> asyncio.Task:
    """Create a background task and prevent it from being garbage collected."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _generation_worker():
    """Worker that processes generation tasks one at a time."""
    while True:
        job = await _generation_queue.get()
        try:
            if job.generation_id in _cancelled_generation_ids:
                _cancelled_generation_ids.discard(job.generation_id)
                job.coro.close()
                continue

            task = asyncio.create_task(job.coro)
            _running_generation_tasks[job.generation_id] = task
            _queued_generation_ids.discard(job.generation_id)
            try:
                await task
            except asyncio.CancelledError:
                # A cancelled job must not take the worker down with it, but a
                # cancellation aimed at the worker itself (shutdown, test
                # teardown) has to win even when the job was cancelled too.
                current = asyncio.current_task()
                if not task.cancelled() or (current is not None and current.cancelling()):
                    raise
        except Exception:
            traceback.print_exc()
            await _force_fail_if_active(
                job.generation_id,
                "Worker exited without writing terminal status",
            )
        finally:
            _running_generation_tasks.pop(job.generation_id, None)
            _queued_generation_ids.discard(job.generation_id)
            _release(job.generation_id)
            _generation_queue.task_done()


async def _force_fail_if_active(generation_id: str, error: str) -> None:
    """Best-effort recovery — flip an active row to failed if the worker
    bailed before writing a terminal status. Catches the case where the gen
    coroutine's own status-write raised (e.g. SQLite lock contention)."""
    if generation_id.startswith((STREAM_JOB_PREFIX, PRELOAD_JOB_PREFIX)):
        # Streaming and preload jobs have no DB row; they report failures themselves.
        return
    try:
        from ..database import Generation as DBGeneration, get_db
        from . import history

        db = next(get_db())
        try:
            gen = db.query(DBGeneration).filter_by(id=generation_id).first()
            if gen is None:
                return
            if (gen.status or "completed") not in ("loading_model", "generating"):
                return
            await history.update_generation_status(
                generation_id=generation_id,
                status="failed",
                db=db,
                error=error,
            )
        finally:
            db.close()
    except Exception:
        traceback.print_exc()


def pending_count() -> int:
    """Jobs queued or running right now."""
    return len(_queued_generation_ids) + len(_running_generation_tasks)


def ensure_capacity(owner: str | None = None, max_pending: int | None = None) -> None:
    """Raise ``QueueFullError`` when the global depth or the owner's cap is reached."""
    if _generation_queue is None:
        raise RuntimeError("Generation queue has not been initialized")
    if lifecycle.is_draining():
        raise QueueFullError("draining", retry_after_s=10)
    if pending_count() >= _max_depth:
        raise QueueFullError("global")
    if owner is not None and max_pending is not None and _pending_by_owner.get(owner, 0) >= max_pending:
        raise QueueFullError("owner")


def enqueue_generation(generation_id: str, coro, *, owner: str | None = None, max_pending: int | None = None):
    """Add a generation coroutine to the serial queue.

    ``owner`` (an API key id) and ``max_pending`` enforce a per-caller cap on
    top of the global depth; a rejected coroutine is closed so it never warns
    about being un-awaited.
    """
    try:
        ensure_capacity(owner, max_pending)
    except (QueueFullError, RuntimeError):
        coro.close()
        raise

    _queued_generation_ids.add(generation_id)
    if owner is not None:
        _job_owner[generation_id] = owner
        _pending_by_owner[owner] = _pending_by_owner.get(owner, 0) + 1
    _generation_queue.put_nowait(GenerationJob(generation_id=generation_id, coro=coro, owner=owner))


def _release(generation_id: str) -> None:
    """Give the owner's pending slot back; safe to call more than once."""
    owner = _job_owner.pop(generation_id, None)
    if owner is None:
        return
    remaining = _pending_by_owner.get(owner, 0) - 1
    if remaining > 0:
        _pending_by_owner[owner] = remaining
    else:
        _pending_by_owner.pop(owner, None)


def cancel_generation(generation_id: str) -> Literal["queued", "running"] | None:
    """Cancel a queued or running generation if it is still active."""
    running_task = _running_generation_tasks.get(generation_id)
    if running_task is not None:
        running_task.cancel()
        return "running"

    if generation_id in _queued_generation_ids:
        _queued_generation_ids.discard(generation_id)
        _cancelled_generation_ids.add(generation_id)
        _release(generation_id)
        return "queued"

    return None


def worker_running() -> bool:
    """Whether the queue worker exists and is still alive."""
    return _generation_worker_task is not None and not _generation_worker_task.done()


async def shutdown(drain_timeout_s: float) -> bool:
    """Drain, then stop the worker.

    Refuses new jobs at once, waits up to *drain_timeout_s* for the pending
    ones, then cancels whatever is still running and closes the coroutines
    still queued.  Returns True when every job finished on its own.
    """
    lifecycle.begin_drain("shutdown")
    if _generation_queue is None:
        return True
    drained = await lifecycle.wait_for_idle(pending_count, drain_timeout_s)
    if not drained:
        logger.warning(
            "Drain timeout (%.0f s) reached with %d job(s) pending; cancelling them",
            drain_timeout_s,
            pending_count(),
        )
    tasks = list(_running_generation_tasks.values())
    worker = _generation_worker_task
    for task in tasks:
        task.cancel()
    if worker is not None and not worker.done():
        worker.cancel()
    for task in [*tasks, worker]:
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    while not _generation_queue.empty():
        job = _generation_queue.get_nowait()
        job.coro.close()
        _queued_generation_ids.discard(job.generation_id)
        _release(job.generation_id)
        _generation_queue.task_done()
    return drained


def init_queue(force: bool = False, *, max_depth: int | None = None):
    """Initialize the generation queue and start the worker.

    Must be called once during application startup (inside a running event loop).
    """
    global _generation_queue, _generation_worker_task, _max_depth
    global _queued_generation_ids, _running_generation_tasks, _cancelled_generation_ids
    global _pending_by_owner, _job_owner

    # Reset fully so a forced re-init (tests, restarts) never inherits a cap.
    _max_depth = DEFAULT_MAX_DEPTH if max_depth is None else max(1, max_depth)

    if _generation_worker_task is not None and not _generation_worker_task.done():
        if not force:
            return
        _generation_worker_task.cancel()
        for task in list(_running_generation_tasks.values()):
            task.cancel()

    _generation_queue = asyncio.Queue()
    _queued_generation_ids = set()
    _running_generation_tasks = {}
    _cancelled_generation_ids = set()
    _pending_by_owner = {}
    _job_owner = {}
    _generation_worker_task = create_background_task(_generation_worker())
