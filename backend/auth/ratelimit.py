"""In-memory token buckets keyed by API key or client IP.

The backend is one process by construction (in-process generation queue,
model singletons), so a local dict is the right store.  Capacity equals the
per-minute limit and refills continuously at ``limit / 60`` per second.
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import HTTPException

from .principal import Principal

logger = logging.getLogger(__name__)

AUTH_FAILURES_PER_MIN = 30
PUBLIC_PER_MIN = 300


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int | None
    remaining: int
    reset_s: int
    retry_after_s: int


ALLOW_UNLIMITED = Decision(allowed=True, limit=None, remaining=0, reset_s=0, retry_after_s=0)


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimited(HTTPException):
    """429 raised by route-level helpers; carries the ``Retry-After`` and ``RateLimit-*`` headers."""

    def __init__(self, decision: Decision, dimension: str) -> None:
        super().__init__(
            status_code=429,
            detail=f"Rate limit exceeded for {dimension}; retry in {decision.retry_after_s} s",
            headers=RateLimiter.headers_for(decision),
        )
        self.decision = decision
        self.dimension = dimension


class RateLimiter:
    def __init__(
        self,
        *,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
        idle_s: float = 600.0,
        max_buckets: int = 10_000,
    ) -> None:
        self.enabled = enabled
        self._clock = clock
        self._idle_s = idle_s
        self._max_buckets = max_buckets
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._charges = 0
        self._oversize_logged: set[str] = set()

    def charge(self, bucket_key: str, *, limit: int | None, cost: float = 1.0) -> Decision:
        """Take ``cost`` tokens from ``bucket_key`` whose capacity is ``limit`` per minute."""
        if not self.enabled or limit is None:
            return ALLOW_UNLIMITED
        if limit <= 0:
            return Decision(allowed=False, limit=limit, remaining=0, reset_s=60, retry_after_s=60)

        capacity = float(limit)
        refill = capacity / 60.0
        now = self._clock()
        bucket = self._buckets.get(bucket_key)
        if bucket is None:
            bucket = _Bucket(tokens=capacity, updated=now)
            self._buckets[bucket_key] = bucket
        else:
            bucket.tokens = min(capacity, bucket.tokens + (now - bucket.updated) * refill)
            bucket.updated = now
            self._buckets.move_to_end(bucket_key)

        if cost > capacity:
            if bucket_key not in self._oversize_logged:
                self._oversize_logged.add(bucket_key)
                logger.warning("Rate-limit cost %.0f exceeds the capacity %d of %s", cost, limit, bucket_key)
            allowed = False
            retry_after = math.ceil(cost / refill)
        elif bucket.tokens >= cost:
            bucket.tokens -= cost
            allowed = True
            retry_after = 0
        else:
            allowed = False
            retry_after = max(1, math.ceil((cost - bucket.tokens) / refill))

        self._charges += 1
        if self._charges % 1000 == 0 or len(self._buckets) > self._max_buckets:
            self.sweep()

        return Decision(
            allowed=allowed,
            limit=limit,
            remaining=max(0, math.floor(bucket.tokens)),
            reset_s=max(0, math.ceil((capacity - bucket.tokens) / refill)),
            retry_after_s=retry_after,
        )

    def charge_principal(self, principal: Principal, dimension: str, cost: float = 1.0) -> Decision:
        limit = getattr(principal.limits, dimension)
        return self.charge(f"key:{principal.key_id}:{dimension}", limit=limit, cost=cost)

    def charge_or_raise(self, principal: Principal, dimension: str, cost: float = 1.0) -> Decision:
        decision = self.charge_principal(principal, dimension, cost)
        if not decision.allowed:
            raise RateLimited(decision, dimension)
        return decision

    def note_auth_failure(self, ip: str, limit: int = AUTH_FAILURES_PER_MIN) -> Decision:
        return self.charge(f"ip:{ip}:auth_failures", limit=limit)

    def charge_public(self, ip: str, limit: int = PUBLIC_PER_MIN) -> Decision:
        return self.charge(f"ip:{ip}:public", limit=limit)

    @staticmethod
    def headers_for(decision: Decision) -> dict[str, str]:
        if decision.limit is None:
            return {}
        headers = {
            "RateLimit-Limit": str(decision.limit),
            "RateLimit-Remaining": str(decision.remaining),
            "RateLimit-Reset": str(decision.reset_s),
        }
        if not decision.allowed:
            headers["Retry-After"] = str(max(1, decision.retry_after_s))
        return headers

    def sweep(self) -> None:
        """Drop buckets idle for longer than ``idle_s`` and enforce the size cap (LRU)."""
        now = self._clock()
        stale = [key for key, bucket in self._buckets.items() if now - bucket.updated > self._idle_s]
        for key in stale:
            del self._buckets[key]
        while len(self._buckets) > self._max_buckets:
            self._buckets.popitem(last=False)
