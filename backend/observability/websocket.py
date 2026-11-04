"""WebSocket lifetime metrics with fixed labels and content-free correlation logs."""

import asyncio
import logging
from time import perf_counter

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import metrics

logger = logging.getLogger(__name__)

# Never use client close text, paths, model names or identifiers as metric labels.
CLOSE_REASONS = frozenset(
    {
        "client_closed",
        "peer_disconnected",
        "handshake_rejected",
        "idle_timeout",
        "session_too_long",
        "server_shutting_down",
        "server_error",
        "slow_consumer",
        "session_limit_reached",
        "rate_limit_exceeded",
        "message_too_large",
        "unsupported_intent",
        "model_not_found",
        "model_not_downloaded",
        "runtime_unavailable",
        "policy_violation",
        "transport_error",
        "cancelled",
        "other",
    }
)


def close_reason(value: str | None, *, fallback: str = "other") -> str:
    """Map server reasons to a finite vocabulary, never retaining arbitrary peer text."""
    return value if value in CLOSE_REASONS else fallback


def socket_kind(path: str) -> str:
    """Collapse routes to a bounded service label, including unrouted handshakes."""
    if path == "/v1/audio/speech/stream":
        return "speech"
    if path in {"/v1/realtime", "/v1/realtime/transcription"}:
        return "transcription"
    return "other"


async def observe_websocket(app: ASGIApp, scope: Scope, receive: Receive, send: Send) -> None:
    """Observe transport lifecycle without reading or retaining audio/text payloads."""
    kind = socket_kind(scope.get("path", ""))
    accepted_at = None
    reason = None

    async def observed_receive() -> Message:
        nonlocal reason
        message = await receive()
        if message["type"] == "websocket.disconnect" and reason is None:
            reason = "peer_disconnected"
        return message

    async def observed_send(message: Message) -> None:
        nonlocal accepted_at, reason
        if message["type"] == "websocket.close":
            fallback = "policy_violation" if message.get("code") == 1008 else "other"
            reason = close_reason(message.get("reason"), fallback=fallback)
            if accepted_at is None:
                reason = "handshake_rejected"
        elif message["type"] == "websocket.http.response.start":
            reason = "handshake_rejected"
        await send(message)
        if message["type"] == "websocket.accept" and accepted_at is None:
            accepted_at = perf_counter()
            metrics.WEBSOCKET_SESSIONS.labels(kind).inc()
            metrics.WEBSOCKET_ACCEPTED.labels(kind).inc()
            logger.info("WebSocket accepted", extra={"event": "websocket.accepted", "socket_kind": kind})

    try:
        await app(scope, observed_receive, observed_send)
    except asyncio.CancelledError:
        reason = reason or "cancelled"
        raise
    except Exception:
        reason = reason or "server_error"
        raise
    finally:
        reason = reason or ("transport_error" if accepted_at is not None else "handshake_rejected")
        if accepted_at is not None:
            metrics.WEBSOCKET_SESSIONS.labels(kind).dec()
            metrics.WEBSOCKET_SECONDS.labels(kind).observe(perf_counter() - accepted_at)
        metrics.WEBSOCKET_CLOSED.labels(kind, reason).inc()
        logger.info(
            "WebSocket ended",
            extra={
                "event": "websocket.closed",
                "socket_kind": kind,
                "reason": reason,
                "session_id": scope.get("state", {}).get("session_id"),
            },
        )
