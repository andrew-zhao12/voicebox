"""One-at-a-time guards for Whisper and the LLM, which run outside the generation queue.

Both models are process-wide singletons that unload and reload themselves, so
two concurrent calls corrupt each other.  A slot admits one caller, lets a
few more wait, and turns everyone else away with a 429.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from ..observability import metrics


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

    @property
    def busy(self) -> bool:
        return self._semaphore.locked()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
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
        try:
            yield
        finally:
            self._semaphore.release()


whisper_slot = InferenceSlot("whisper")
llm_slot = InferenceSlot("llm")
