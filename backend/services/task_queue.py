"""Generation queue: one worker per lane.

By default there is a single lane (``all``) with one worker, so TTS
inference is strictly serial, which is what one GPU wants.  With
``VOICEBOX_GENERATION_WORKERS=2`` (``init_queue(workers=2)``) the work splits
into a ``cpu`` and a ``gpu`` lane, each with its own FIFO and worker, so a
CPU engine such as Kokoro synthesizes while a GPU engine does; jobs in the
same lane run one at a time unless ``VOICEBOX_ENGINE_CONCURRENCY``
(``init_queue(engine_concurrency=...)``) lets several jobs of one engine
overlap: the lane's worker still takes jobs strictly in order, and the head
job starts only when the lane is empty or already running its engine below
that engine's limit, so a different engine never overtakes.  Seeded jobs
are *exclusive*: ``torch.manual_seed`` is process-wide, so an exclusive job
waits until everything is idle and blocks new starts until it finishes.
Model loading is serialized separately by ``services/generation.prepare_engine``.
"""

import asyncio
import contextlib
import logging
import os
import time
import traceback
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter
from typing import Literal

from .. import lifecycle
from ..observability import metrics

logger = logging.getLogger(__name__)

# Keep references to fire-and-forget background tasks to prevent GC
_background_tasks: set = set()

# Job ids of streaming generations (``POST /generate/stream``). They share the
# queue with regular generations but have no ``generations`` row.
STREAM_JOB_PREFIX = "stream-"
# Job ids of startup preloads (``services/preload.py``); no ``generations`` row either.
PRELOAD_JOB_PREFIX = "preload-"

LANE_ALL = "all"
LANE_CPU = "cpu"
LANE_GPU = "gpu"
WORKERS_ENV = "VOICEBOX_GENERATION_WORKERS"
MAX_WORKERS = 2
ENGINE_CONCURRENCY_ENV = "VOICEBOX_ENGINE_CONCURRENCY"
MAX_ENGINE_CONCURRENCY = 4
# Engines whose ``generate()`` keeps no per-call state on the shared model
# object (reviewed 2026-09-29).  Chatterbox and Chatterbox Turbo are not on
# the list: their ``generate(audio_prompt_path=...)`` stores the reference
# voice on the model (``prepare_conditionals`` -> ``self.conds``), so two
# concurrent calls would swap voices.
CONCURRENCY_SAFE_ENGINES = frozenset({"kokoro", "qwen", "qwen_custom_voice", "luxtts", "tada"})


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
    enqueued_at: float = field(default_factory=perf_counter)
    lane: str = LANE_ALL
    engine: str | None = None
    exclusive: bool = False


@dataclass
class _Lane:
    name: str
    queue: asyncio.Queue
    worker: asyncio.Task | None = None
    # Jobs in flight on this lane (id -> task), the engine they share and
    # whether the one running is exclusive; ``changed`` wakes the worker
    # whenever a job finishes so the head of the queue can be reconsidered.
    running: dict[str, asyncio.Task] = field(default_factory=dict)
    running_engine: str | None = None
    running_exclusive: bool = False
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)


class _ExclusiveGate:
    """Readers/writer gate: an exclusive job runs alone, everything else may overlap."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active = 0
        self._exclusive = False

    async def enter(self, exclusive: bool) -> None:
        async with self._condition:
            if exclusive:
                await self._condition.wait_for(lambda: self._active == 0 and not self._exclusive)
                self._exclusive = True
            else:
                await self._condition.wait_for(lambda: not self._exclusive)
            self._active += 1

    async def leave(self, exclusive: bool) -> None:
        async with self._condition:
            self._active -= 1
            if exclusive:
                self._exclusive = False
            self._condition.notify_all()


DEFAULT_MAX_DEPTH = 32

_lanes: dict[str, _Lane] = {}
_gate: _ExclusiveGate | None = None
# The first lane's worker; kept for tests and ``worker_running``.
_generation_worker_task: asyncio.Task | None = None
_queued_generation_ids: set[str] = set()
_running_generation_tasks: dict[str, asyncio.Task] = {}
_cancelled_generation_ids: set[str] = set()
_max_depth: int = DEFAULT_MAX_DEPTH
# Engine -> how many of its jobs may run at once in a lane (absent = 1).
_engine_limits: dict[str, int] = {}
# Pending (queued + running) jobs per owner (API key id), for per-key caps.
_pending_by_owner: dict[str, int] = {}
_job_owner: dict[str, str] = {}
# Engine per queued or running job, and when each engine last finished a job.
_job_engine: dict[str, str] = {}
_engine_last_used: dict[str, float] = {}


def configured_workers(environ: Mapping[str, str] = os.environ) -> int:
    """``VOICEBOX_GENERATION_WORKERS``: 1 (serial, default) or 2 (a cpu and a gpu lane)."""
    raw = environ.get(WORKERS_ENV)
    if raw is None or not raw.strip():
        return 1
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using 1", WORKERS_ENV, raw)
        return 1
    if value > MAX_WORKERS:
        logger.warning("%s=%d: only the cpu and gpu lanes exist; using %d", WORKERS_ENV, value, MAX_WORKERS)
    return max(1, min(value, MAX_WORKERS))


def configured_engine_concurrency(environ: Mapping[str, str] = os.environ) -> dict[str, int]:
    """``VOICEBOX_ENGINE_CONCURRENCY="kokoro=2,qwen=2"``: jobs of one engine that may run at once.

    Only reviewed engines (``CONCURRENCY_SAFE_ENGINES``) may exceed 1, at most
    ``MAX_ENGINE_CONCURRENCY``; anything else is logged and kept serial.
    """
    limits: dict[str, int] = {}
    raw = environ.get(ENGINE_CONCURRENCY_ENV, "")
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, _sep, value = item.partition("=")
        name = name.strip().lower()
        try:
            count = int(value.strip())
        except ValueError:
            logger.warning("%s: %r is not engine=count; ignoring it", ENGINE_CONCURRENCY_ENV, item)
            continue
        if name not in CONCURRENCY_SAFE_ENGINES:
            logger.warning(
                "%s: %s is not reviewed for concurrent generation (safe: %s); keeping it serial",
                ENGINE_CONCURRENCY_ENV,
                name,
                ", ".join(sorted(CONCURRENCY_SAFE_ENGINES)),
            )
            continue
        if count > MAX_ENGINE_CONCURRENCY:
            logger.warning("%s: %s=%d capped at %d", ENGINE_CONCURRENCY_ENV, name, count, MAX_ENGINE_CONCURRENCY)
            count = MAX_ENGINE_CONCURRENCY
        if count > 1:
            limits[name] = count
    return limits


def engine_limit(engine: str | None) -> int:
    """How many jobs of *engine* may run at once in one lane (1 unless configured)."""
    return _engine_limits.get(engine, 1) if engine else 1


def engine_limits() -> dict[str, int]:
    return dict(_engine_limits)


def create_background_task(coro) -> asyncio.Task:
    """Create a background task and prevent it from being garbage collected."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def lane_names() -> list[str]:
    return list(_lanes)


def parallel() -> bool:
    """Whether more than one lane exists."""
    return len(_lanes) > 1


def default_lane() -> str:
    """Where a job without a lane goes: the only lane, or the gpu lane (the safe assumption)."""
    return LANE_GPU if parallel() else LANE_ALL


def resolve_lane(lane: str | None) -> str:
    if not parallel():
        return LANE_ALL
    return lane if lane in _lanes else default_lane()


def _can_start(lane: _Lane, job: GenerationJob, limit: int) -> bool:
    """Whether the head job may start now: an empty lane, or the same engine below its limit."""
    if not lane.running:
        return True
    if job.exclusive or lane.running_exclusive:
        return False
    return job.engine is not None and job.engine == lane.running_engine and len(lane.running) < limit


def _skip_cancelled(job: GenerationJob) -> bool:
    if job.generation_id not in _cancelled_generation_ids:
        return False
    _cancelled_generation_ids.discard(job.generation_id)
    job.coro.close()
    return True


async def _lane_worker(lane: _Lane):
    """Dispatch the lane's jobs in order; same-engine jobs overlap up to the engine's limit (default 1)."""
    held: GenerationJob | None = None  # taken off the queue, not dispatched yet
    try:
        while True:
            job = await lane.queue.get()
            held = job
            try:
                if _skip_cancelled(job):
                    continue
                limit = engine_limit(job.engine)
                async with lane.changed:
                    await lane.changed.wait_for(partial(_can_start, lane, job, limit))
                if _skip_cancelled(job):  # cancelled while it waited for a slot
                    continue

                metrics.QUEUE_WAIT_SECONDS.observe(perf_counter() - job.enqueued_at)
                entered = False
                if _gate is not None:
                    await _gate.enter(job.exclusive)
                    entered = True
                task = asyncio.create_task(_run_job(lane, job, entered))
                lane.running[job.generation_id] = task
                lane.running_engine = job.engine
                lane.running_exclusive = job.exclusive
                _running_generation_tasks[job.generation_id] = task
                _queued_generation_ids.discard(job.generation_id)
                metrics.QUEUE_RUNNING.labels(lane.name).set(len(lane.running))
            finally:
                held = None
                lane.queue.task_done()
    except asyncio.CancelledError:
        # Stopping the worker (shutdown, re-init) stops what it dispatched and
        # drops the job it was holding for a slot, as shutdown() drops queued ones.
        for task in list(lane.running.values()):
            task.cancel()
        if held is not None:
            held.coro.close()
            _queued_generation_ids.discard(held.generation_id)
            _release(held.generation_id)
            metrics.QUEUE_PENDING.set(pending_count())
        raise


async def _run_job(lane: _Lane, job: GenerationJob, entered: bool) -> None:
    """Await one job and do the lane's bookkeeping when it ends, however it ends."""
    try:
        await job.coro
    except asyncio.CancelledError:
        raise
    except Exception:
        traceback.print_exc()
        await _force_fail_if_active(
            job.generation_id,
            "Worker exited without writing terminal status",
        )
    finally:
        if entered and _gate is not None:
            await _gate.leave(job.exclusive)
        lane.running.pop(job.generation_id, None)
        if not lane.running:
            lane.running_engine = None
            lane.running_exclusive = False
        _running_generation_tasks.pop(job.generation_id, None)
        _queued_generation_ids.discard(job.generation_id)
        _release(job.generation_id)
        metrics.QUEUE_RUNNING.labels(lane.name).set(len(lane.running))
        metrics.QUEUE_PENDING.set(pending_count())
        async with lane.changed:
            lane.changed.notify_all()


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
    """Jobs queued or running right now, across all lanes."""
    return len(_queued_generation_ids) + len(_running_generation_tasks)


def engine_in_use(engine: str) -> int:
    """Queued or running jobs for *engine* (so an unload can be refused)."""
    return sum(1 for job_engine in _job_engine.values() if job_engine == engine)


def engines_last_used() -> dict[str, float]:
    """``time.monotonic()`` of each engine's last finished job."""
    return dict(_engine_last_used)


def ensure_capacity(owner: str | None = None, max_pending: int | None = None) -> None:
    """Raise ``QueueFullError`` when the global depth or the owner's cap is reached."""
    if not _lanes:
        raise RuntimeError("Generation queue has not been initialized")
    if lifecycle.is_draining():
        raise QueueFullError("draining", retry_after_s=10)
    if pending_count() >= _max_depth:
        raise QueueFullError("global")
    if owner is not None and max_pending is not None and _pending_by_owner.get(owner, 0) >= max_pending:
        raise QueueFullError("owner")


def enqueue_generation(
    generation_id: str,
    coro,
    *,
    owner: str | None = None,
    max_pending: int | None = None,
    lane: str | None = None,
    engine: str | None = None,
    exclusive: bool = False,
):
    """Add a generation coroutine to its lane's queue.

    ``owner`` (an API key id) and ``max_pending`` enforce a per-caller cap on
    top of the global depth; ``lane`` picks the worker (ignored with a single
    lane), ``engine`` is recorded for ``engine_in_use``, and ``exclusive``
    jobs (seeded synthesis) run with nothing else in flight.  A rejected
    coroutine is closed so it never warns about being un-awaited.
    """
    try:
        ensure_capacity(owner, max_pending)
    except (QueueFullError, RuntimeError):
        coro.close()
        raise

    lane_name = resolve_lane(lane)
    _queued_generation_ids.add(generation_id)
    if owner is not None:
        _job_owner[generation_id] = owner
        _pending_by_owner[owner] = _pending_by_owner.get(owner, 0) + 1
    if engine is not None:
        _job_engine[generation_id] = engine
    _lanes[lane_name].queue.put_nowait(
        GenerationJob(
            generation_id=generation_id,
            coro=coro,
            owner=owner,
            lane=lane_name,
            engine=engine,
            exclusive=exclusive,
        )
    )
    metrics.QUEUE_PENDING.set(pending_count())


def _release(generation_id: str) -> None:
    """Give the owner's pending slot back and record the engine's last use; safe to call more than once."""
    engine = _job_engine.pop(generation_id, None)
    if engine is not None:
        _engine_last_used[engine] = time.monotonic()
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
        metrics.QUEUE_PENDING.set(pending_count())
        return "queued"

    return None


def worker_running() -> bool:
    """Whether every lane's worker exists and is still alive."""
    return bool(_lanes) and all(lane.worker is not None and not lane.worker.done() for lane in _lanes.values())


async def shutdown(drain_timeout_s: float) -> bool:
    """Drain, then stop the workers.

    Refuses new jobs at once, waits up to *drain_timeout_s* for the pending
    ones, then cancels whatever is still running and closes the coroutines
    still queued.  Returns True when every job finished on its own.
    """
    lifecycle.begin_drain("shutdown")
    if not _lanes:
        return True
    drained = await lifecycle.wait_for_idle(pending_count, drain_timeout_s)
    if not drained:
        logger.warning(
            "Drain timeout (%.0f s) reached with %d job(s) pending; cancelling them",
            drain_timeout_s,
            pending_count(),
        )
    tasks = list(_running_generation_tasks.values())
    workers = [lane.worker for lane in _lanes.values() if lane.worker is not None]
    for task in tasks:
        task.cancel()
    for worker in workers:
        if not worker.done():
            worker.cancel()
    for task in [*tasks, *workers]:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    for lane in _lanes.values():
        while not lane.queue.empty():
            job = lane.queue.get_nowait()
            job.coro.close()
            _queued_generation_ids.discard(job.generation_id)
            _release(job.generation_id)
            lane.queue.task_done()
    return drained


def init_queue(
    force: bool = False,
    *,
    max_depth: int | None = None,
    workers: int = 1,
    engine_concurrency: Mapping[str, int] | None = None,
):
    """Create the lanes and start their workers.

    Must be called once during application startup (inside a running event
    loop).  ``workers=1`` keeps everything serial in one lane; ``workers=2``
    creates the ``gpu`` and ``cpu`` lanes.  ``engine_concurrency`` maps an
    engine to how many of its jobs may run at once within a lane.
    """
    global _lanes, _gate, _generation_worker_task, _max_depth, _engine_limits
    global _queued_generation_ids, _running_generation_tasks, _cancelled_generation_ids
    global _pending_by_owner, _job_owner, _job_engine, _engine_last_used

    # Reset fully so a forced re-init (tests, restarts) never inherits a cap.
    _max_depth = DEFAULT_MAX_DEPTH if max_depth is None else max(1, max_depth)
    _engine_limits = {name: max(1, int(count)) for name, count in (engine_concurrency or {}).items()}

    if worker_running():
        if not force:
            return
        for lane in _lanes.values():
            if lane.worker is not None:
                lane.worker.cancel()
        for task in list(_running_generation_tasks.values()):
            task.cancel()

    names = [LANE_ALL] if workers <= 1 else [LANE_GPU, LANE_CPU]
    _lanes = {name: _Lane(name=name, queue=asyncio.Queue()) for name in names}
    _gate = _ExclusiveGate() if len(names) > 1 else None
    for name in names:
        metrics.QUEUE_RUNNING.labels(name).set(0)
    _queued_generation_ids = set()
    _running_generation_tasks = {}
    _cancelled_generation_ids = set()
    _pending_by_owner = {}
    _job_owner = {}
    _job_engine = {}
    _engine_last_used = {}
    for lane in _lanes.values():
        lane.worker = create_background_task(_lane_worker(lane))
    _generation_worker_task = _lanes[names[0]].worker
