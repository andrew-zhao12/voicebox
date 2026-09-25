"""Who is calling: the principal the auth middleware resolves for each request."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Literal

from fastapi import Depends, HTTPException, Request

Role = Literal["admin", "client", "anonymous"]
Via = Literal["header", "token", "none"]

MIB = 1024 * 1024
GIB = 1024 * MIB

LIMIT_FIELDS = ("requests", "inference", "uploads_bytes", "tts_chars", "max_pending_jobs")


@dataclass(frozen=True)
class KeyLimits:
    """Per-minute budgets for one key (``max_pending_jobs`` is a count); ``None`` means unlimited."""

    requests: int | None
    inference: int | None
    uploads_bytes: int | None
    tts_chars: int | None
    max_pending_jobs: int | None

    @classmethod
    def defaults_for(cls, role: str) -> KeyLimits:
        if role == "admin":
            return cls(requests=1200, inference=120, uploads_bytes=2 * GIB, tts_chars=600_000, max_pending_jobs=None)
        if role == "client":
            return cls(requests=300, inference=30, uploads_bytes=256 * MIB, tts_chars=60_000, max_pending_jobs=4)
        return cls(requests=0, inference=0, uploads_bytes=0, tts_chars=0, max_pending_jobs=0)

    def merged(self, overrides: Mapping[str, int | None] | None) -> KeyLimits:
        if not overrides:
            return self
        values = {name: overrides[name] for name in LIMIT_FIELDS if name in overrides}
        return replace(self, **values)

    def as_dict(self) -> dict[str, int | None]:
        return {name: getattr(self, name) for name in LIMIT_FIELDS}


@dataclass(frozen=True)
class Principal:
    key_id: str
    role: Role
    via: Via
    limits: KeyLimits

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def authenticated(self) -> bool:
        return self.via != "none"


ANONYMOUS = Principal(key_id="anonymous", role="anonymous", via="none", limits=KeyLimits.defaults_for("anonymous"))

principal_var: ContextVar[Principal | None] = ContextVar("voicebox_principal", default=None)


def principal_from_scope(scope: Mapping[str, Any]) -> Principal:
    """The principal the middleware stored on the ASGI scope, else ``ANONYMOUS``."""
    state = scope.get("state")
    principal = state.get("principal") if isinstance(state, dict) else None
    return principal if isinstance(principal, Principal) else ANONYMOUS


def current_principal(request: Request) -> Principal:
    """FastAPI dependency: the caller the auth middleware resolved."""
    return principal_from_scope(request.scope)


def require_admin(principal: Principal = Depends(current_principal)) -> Principal:
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="Admin key required")
    return principal


def get_principal() -> Principal:
    """The principal of the in-flight request for code without a ``Request``.

    Fails closed: raising here turns a missing middleware into a 500 instead
    of silently treating the caller as an admin.
    """
    principal = principal_var.get()
    if principal is None:
        raise RuntimeError("No principal in context; the auth middleware did not run")
    return principal


@contextmanager
def principal_scope(principal: Principal) -> Iterator[None]:
    """Run a block as ``principal`` (MCP tools, tests)."""
    token = principal_var.set(principal)
    try:
        yield
    finally:
        principal_var.reset(token)
