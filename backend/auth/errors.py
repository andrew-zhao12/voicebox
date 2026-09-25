"""Generic 500 for unexpected exceptions so tracebacks and paths stay in the log."""

from __future__ import annotations

import logging
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse

from ..api_errors import error_body

logger = logging.getLogger(__name__)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # The request id doubles as the error id, so a client-reported id finds the log line.
    error_id = request.scope.get("state", {}).get("request_id") or secrets.token_hex(8)
    logger.error(
        "Unhandled error %s: %s %s",
        error_id,
        request.method,
        request.url.path,
        exc_info=exc,
        extra={"error_id": error_id},
    )
    body = error_body(request.url.path, 500, "Internal server error")
    (body.get("error") or body)["error_id"] = error_id
    # Starlette's ServerErrorMiddleware sends this response outside the
    # request-id middleware, so the header is added here.
    return JSONResponse(body, status_code=500, headers={"X-Request-Id": error_id})
