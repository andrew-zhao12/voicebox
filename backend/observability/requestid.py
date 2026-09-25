"""A request id on every request and response, plus HTTP metrics.

``X-Request-Id`` is echoed when the client sends a well-formed one and
generated otherwise.  It rides on ``scope["state"]["request_id"]``, on the
``request_id_var`` ContextVar (read by the JSON log formatter and the 500
handler, whose ``error_id`` is the same value) and on the response header.
The middleware sits outermost so even the auth middleware's 401s carry it.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar
from time import perf_counter

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import metrics

HEADER = "x-request-id"
_VALID = re.compile(r"^[A-Za-z0-9._:/-]{1,64}$")

request_id_var: ContextVar[str | None] = ContextVar("voicebox_request_id", default=None)


def current_request_id() -> str | None:
    return request_id_var.get()


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def route_label(scope: Scope) -> str:
    """The matched route template for metric labels; ``unrouted`` when a middleware answered."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unrouted"


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(HEADER)
        request_id = incoming if incoming and _VALID.match(incoming) else new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)
        status = 500
        started = perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                MutableHeaders(scope=message).append(HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)
            metrics.observe_http(str(scope.get("method", "GET")), route_label(scope), status, perf_counter() - started)
