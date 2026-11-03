"""One-at-a-time guards for Whisper and the LLM, which run outside the generation queue.

Both models are process-wide singletons that unload and reload themselves, so
two concurrent calls corrupt each other.  A slot admits one caller, lets a
few more wait, and turns everyone else away with a 429.  Model calls go
through ``run()``, which keeps the slot until the call has really ended: the
work happens in a worker thread that cancelling the caller cannot stop.
There is deliberately no ``async with slot.acquire()``: it released the slot
when its block was cancelled, so the next call overlapped that thread.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from time import perf_counter

from ..observability import metrics

logger = logging.getLogger(__name__)


class InferenceBusyError(Exception):
    def __init__(self, name: str, retry_after_s: int = 5) -> None:
        super().__init__(f"{name} is busy; retry in {retry_after_s} s")
        self.name = name
        self.retry_after_s = retry_after_s


class InferenceSlot:
    def __init__(self, name: str, *, max_waiters: int = 4, wait_timeout_s: float = 60.0) -> None:
        self.name = name
        self.max_waiters = max_waiters
        self.wait_timeout_s = wait_timeout_s
        self._semaphore = asyncio.Semaphore(1)
        self._waiters = 0
        # The task ``run()`` started.  The event loop keeps only weak references
        # to tasks, and once its caller is cancelled nothing else refers to it.
        self._inflight: asyncio.Future[object] | None = None
        # ``time.monotonic()`` of the last release; None until first used.
        self.last_used: float | None = None

    @property
    def busy(self) -> bool:
        return self._semaphore.locked()

    async def run[**P, T](self, fn: Callable[P, Awaitable[T]], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Await ``fn(*args, **kwargs)`` in the slot and return its result.

        ``fn`` is called once the slot is taken, and its awaitable runs as a
        task of its own that releases the slot when it ends.  Cancelling the
        caller only stops the caller waiting: Whisper and the LLM run in
        worker threads that cannot be interrupted, so the slot stays taken
        until the call is really over and the next caller cannot overlap it.

        Raises:
            InferenceBusyError: Too many callers are already waiting, or the
                wait outlasted ``wait_timeout_s``.
        """
        await self._take()
        try:
            inflight = asyncio.ensure_future(fn(*args, **kwargs))
        except Exception:
            self._release()
            raise
        self._inflight = inflight
        inflight.add_done_callback(self._finished)
        try:
            return await asyncio.shield(inflight)
        except asyncio.CancelledError:
            if not inflight.done():
                logger.debug("%s caller cancelled; the slot stays taken until its call ends", self.name)
                inflight.add_done_callback(self._log_abandoned)
            raise

    async def _take(self) -> None:
        if self._semaphore.locked() and self._waiters >= self.max_waiters:
            metrics.RATE_LIMITED.labels(f"slot:{self.name}").inc()
            raise InferenceBusyError(self.name)
        self._waiters += 1
        started = perf_counter()
        try:
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=self.wait_timeout_s)
            except TimeoutError:
                metrics.RATE_LIMITED.labels(f"slot:{self.name}").inc()
                raise InferenceBusyError(self.name) from None
        finally:
            self._waiters -= 1
            metrics.SLOT_WAIT_SECONDS.labels(self.name).observe(perf_counter() - started)

    def _release(self) -> None:
        self.last_used = time.monotonic()
        self._semaphore.release()

    def _finished(self, inflight: asyncio.Future[object]) -> None:
        self._inflight = None
        self._release()

    def _log_abandoned(self, inflight: asyncio.Future[object]) -> None:
        # ``shield`` stops watching the task once its caller is cancelled, so
        # read the exception here or asyncio reports it as never retrieved.
        if not inflight.cancelled() and (error := inflight.exception()) is not None:
            logger.debug("%s call failed after its caller was cancelled", self.name, exc_info=error)


whisper_slot = InferenceSlot("whisper")
llm_slot = InferenceSlot("llm")
