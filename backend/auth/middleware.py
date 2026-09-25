"""Pure ASGI middlewares: authentication, rate limiting, security headers and body caps.

They are plain ASGI callables rather than ``BaseHTTPMiddleware`` because the
API streams SSE and PCM audio, and they answer with responses of their own
because they sit outside FastAPI's ``ExceptionMiddleware``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

from starlette.datastructures import Headers
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..api_errors import error_body
from ..observability import metrics
from . import policy
from .principal import ANONYMOUS, Principal, principal_from_scope, principal_var

if TYPE_CHECKING:
    from .install import SecurityRuntime
    from .settings import SecuritySettings

logger = logging.getLogger(__name__)

WWW_AUTHENTICATE = 'Bearer realm="voicebox"'
WWW_AUTHENTICATE_INVALID = 'Bearer realm="voicebox", error="invalid_token"'
HTML_TYPES = ("text/html", "application/xhtml+xml")
MINIMAL_HEALTH = {"status": "healthy", "service": "voicebox"}
SAFE_METHODS = ("GET", "HEAD")


def client_ip(scope: Scope) -> str:
    client = scope.get("client")
    return client[0] if client and client[0] else "unknown"


def _json(scope: Scope, status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    """An error body: ``{"detail": ...}``, or the OpenAI envelope under ``/v1/``."""
    return JSONResponse(error_body(str(scope.get("path", "/")), status, detail), status_code=status, headers=headers)


def _prefers_html(accept: str | None) -> bool:
    if not accept:
        return False
    first = accept.split(",", 1)[0].split(";", 1)[0].strip().lower()
    return first in HTML_TYPES


def _bearer_credential(headers: Headers) -> tuple[bool, str]:
    """``(header_present, credential)``; the credential is empty for a malformed header."""
    raw = headers.get("authorization")
    if raw is None:
        return False, ""
    scheme, _, credential = raw.strip().partition(" ")
    if scheme.lower() != "bearer":
        return True, ""
    return True, credential.strip()


def _query_param(scope: Scope, name: str) -> str | None:
    raw = scope.get("query_string") or b""
    if not raw:
        return None
    values = parse_qs(raw.decode("latin-1"), keep_blank_values=True).get(name)
    return values[0] if values else None


def _append_headers(message: Message, extra: dict[str, str]) -> None:
    """Add response headers that are not already present."""
    raw = list(message.get("headers") or [])
    present = {key.lower() for key, _ in raw}
    for key, value in extra.items():
        lower = key.lower().encode("latin-1")
        if lower not in present:
            raw.append((lower, value.encode("latin-1")))
    message["headers"] = raw


def _header_value(message: Message, name: str) -> str | None:
    wanted = name.lower().encode("latin-1")
    for key, value in message.get("headers") or []:
        if key.lower() == wanted:
            return value.decode("latin-1")
    return None


class AuthMiddleware:
    """Resolve the caller from the bearer header or a media token, or reject."""

    def __init__(self, app: ASGIApp, *, runtime: SecurityRuntime) -> None:
        self.app = app
        self.runtime = runtime

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return

        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", "/"))
        headers = Headers(scope=scope)

        principal, failure = self._authenticate(method, path, headers, scope)
        if principal is None:
            await self._handle_anonymous(scope, receive, send, method, path, headers, failure)
            return

        docs_enabled = not self.runtime.settings.disable_docs
        if not policy.allows(principal.role, method, path, docs_enabled=docs_enabled):
            self.runtime.limiter.note_auth_failure(client_ip(scope))
            await _json(scope, 403, "Admin key required")(scope, receive, send)
            return

        await self._run_as(principal, scope, receive, send)

    async def _run_as(self, principal: Principal, scope: Scope, receive: Receive, send: Send) -> None:
        scope.setdefault("state", {})["principal"] = principal
        token = principal_var.set(principal)
        try:
            await self.app(scope, receive, send)
        finally:
            principal_var.reset(token)

    def _authenticate(
        self, method: str, path: str, headers: Headers, scope: Scope
    ) -> tuple[Principal | None, str | None]:
        """``(principal, failure)`` where failure is ``None``, ``missing``, ``invalid`` or ``token``."""
        present, credential = _bearer_credential(headers)
        if present:
            record = self.runtime.keystore.lookup(credential) if credential else None
            if record is None:
                return None, "invalid"
            return Principal(key_id=record.id, role=record.role, via="header", limits=record.key_limits()), None

        if method in SAFE_METHODS and policy.token_allowed(method, path):
            token = _query_param(scope, "token")
            if token is not None:
                claims = self.runtime.tokens.verify(token)
                record = self.runtime.keystore.get(claims.key_id) if claims else None
                if claims is None or record is None:
                    return None, "token"
                return Principal(key_id=record.id, role=record.role, via="token", limits=record.key_limits()), None

        return None, "missing"

    async def _handle_anonymous(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        method: str,
        path: str,
        headers: Headers,
        failure: str | None,
    ) -> None:
        settings = self.runtime.settings
        limiter = self.runtime.limiter
        ip = client_ip(scope)

        if policy.is_public(method, path, docs_enabled=not settings.disable_docs):
            if method in SAFE_METHODS and path == "/health":
                await JSONResponse(MINIMAL_HEALTH)(scope, receive, send)
                return
            decision = limiter.charge_public(ip)
            if not decision.allowed:
                metrics.RATE_LIMITED.labels("public").inc()
                await _json(scope, 429, "Too many requests", limiter.headers_for(decision))(scope, receive, send)
                return
            await self._run_as(ANONYMOUS, scope, receive, send)
            return

        if method in SAFE_METHODS and settings.frontend_dir is not None:
            spa = self._spa_response(path, headers, settings.frontend_dir)
            if spa == "passthrough":
                await self._run_as(ANONYMOUS, scope, receive, send)
                return
            if spa is not None:
                await spa(scope, receive, send)
                return

        token_path = method in SAFE_METHODS and policy.token_allowed(method, path)
        if failure in ("missing", "invalid") and not token_path:
            decision = limiter.note_auth_failure(ip)
            if not decision.allowed:
                metrics.RATE_LIMITED.labels("auth_failures").inc()
                await _json(scope, 429, "Too many failed authentication attempts", limiter.headers_for(decision))(
                    scope, receive, send
                )
                return

        if failure == "invalid":
            detail, challenge = "Invalid API key", WWW_AUTHENTICATE_INVALID
        elif failure == "token":
            detail, challenge = "Invalid or expired media token", WWW_AUTHENTICATE_INVALID
        else:
            detail, challenge = "Authentication required", WWW_AUTHENTICATE
        await _json(scope, 401, detail, {"WWW-Authenticate": challenge})(scope, receive, send)

    @staticmethod
    def _spa_response(path: str, headers: Headers, frontend_dir: Path) -> Response | str | None:
        """Static file, ``index.html`` for browser navigations, or ``"passthrough"`` for ``/assets``."""
        if path.startswith("/assets/"):
            return "passthrough"
        if path not in ("", "/"):
            root = frontend_dir.resolve()
            candidate = (root / path.lstrip("/")).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return FileResponse(candidate)
        if _prefers_html(headers.get("accept")):
            index = frontend_dir / "index.html"
            if index.is_file():
                return FileResponse(index, media_type="text/html")
        return None


class RateLimitMiddleware:
    """Charge the per-key buckets and decorate responses with ``RateLimit-*`` headers."""

    def __init__(self, app: ASGIApp, *, runtime: SecurityRuntime) -> None:
        self.app = app
        self.runtime = runtime

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limiter = self.runtime.limiter
        principal = principal_from_scope(scope)
        if not principal.authenticated or not limiter.enabled:
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", "/"))
        headers = Headers(scope=scope)

        decision = limiter.charge_principal(principal, "requests")
        if not decision.allowed:
            await self._reject(scope, receive, send, "requests", decision)
            return
        if policy.is_inference(method, path):
            inference = limiter.charge_principal(principal, "inference")
            if not inference.allowed:
                await self._reject(scope, receive, send, "inference", inference)
                return
        if policy.is_upload(method, path):
            length = _content_length(headers)
            if length:
                uploads = limiter.charge_principal(principal, "uploads_bytes", cost=length)
                if not uploads.allowed:
                    await self._reject(scope, receive, send, "uploads", uploads)
                    return
            elif "chunked" in headers.get("transfer-encoding", "").lower():
                receive = self._counting_receive(receive, principal)

        scope.setdefault("state", {})["ratelimit"] = decision

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                current = scope["state"].get("ratelimit", decision)
                _append_headers(message, limiter.headers_for(current))
            await send(message)

        await self.app(scope, receive, send_with_headers)

    def _counting_receive(self, receive: Receive, principal: Principal) -> Receive:
        limiter = self.runtime.limiter
        seen = 0

        async def counting() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body") or b"")
                if not message.get("more_body") and seen:
                    limiter.charge_principal(principal, "uploads_bytes", cost=seen)
            return message

        return counting

    async def _reject(self, scope: Scope, receive: Receive, send: Send, dimension: str, decision) -> None:
        detail = f"Rate limit exceeded for {dimension}; retry in {max(1, decision.retry_after_s)} s"
        metrics.RATE_LIMITED.labels(dimension).inc()
        await _json(scope, 429, detail, self.runtime.limiter.headers_for(decision))(scope, receive, send)


def _content_length(headers: Headers) -> int:
    try:
        return max(0, int(headers.get("content-length", "0")))
    except ValueError:
        return 0


class SecurityHeadersMiddleware:
    """Defensive response headers; no CSP because the desktop webview loads media cross-origin."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                extra = {
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "Referrer-Policy": "no-referrer",
                }
                content_type = _header_value(message, "content-type") or ""
                if content_type.startswith("application/json"):
                    extra["Cache-Control"] = "no-store"
                _append_headers(message, extra)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodyLimitMiddleware:
    """Per-route request body caps on top of Starlette's ``RequestBodyLimitMiddleware``."""

    def __init__(self, app: ASGIApp, *, settings: SecuritySettings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        cap = policy.body_limit_for(
            str(scope.get("method", "GET")),
            str(scope.get("path", "/")),
            headers.get("content-type", ""),
            default=self.settings.body_limit_default,
            multipart_default=self.settings.body_limit_multipart,
        )
        await RequestBodyLimitMiddleware(self.app, max_body_size=cap)(scope, receive, send)
