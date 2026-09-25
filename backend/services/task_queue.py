"""
Serial generation queue — ensures only one TTS inference runs at a time
to avoid GPU contention.
"""

import asyncio
import traceback
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Literal

# Keep references to fire-and-forget background tasks to prevent GC
_background_tasks: set = set()

# Job ids of streaming generations (``POST /generate/stream``). They share the
# queue with regular generations but have no ``generations`` row.
STREAM_JOB_PREFIX = "stream-"


class QueueFullError(Exception):
    """Raised by ``enqueue_generation`` when the global or the caller's pending cap is reached."""

    def __init__(self, reason: Literal["global", "owner"], retry_after_s: int = 5) -> None:
        super().__init__(f"Generation queue is full ({reason} limit); retry in {retry_after_s} s")
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
    if generation_id.startswith(STREAM_JOB_PREFIX):
        # Streaming jobs have no DB row; they report failures to their consumer.
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
