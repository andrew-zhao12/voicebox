"""Generic 500 for unexpected exceptions so tracebacks and paths stay in the log."""

from __future__ import annotations

import logging
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse

from ..api_errors import error_body

logger = logging.getLogger(__name__)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    error_id = secrets.token_hex(8)
    logger.error("Unhandled error %s: %s %s", error_id, request.method, request.url.path, exc_info=exc)
    body = error_body(request.url.path, 500, "Internal server error")
    (body.get("error") or body)["error_id"] = error_id
    return JSONResponse(body, status_code=500)
