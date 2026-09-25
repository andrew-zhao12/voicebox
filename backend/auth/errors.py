"""Generic 500 for unexpected exceptions so tracebacks and paths stay in the log."""

from __future__ import annotations

import logging
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    error_id = secrets.token_hex(8)
    logger.error("Unhandled error %s: %s %s", error_id, request.method, request.url.path, exc_info=exc)
    return JSONResponse({"detail": "Internal server error", "error_id": error_id}, status_code=500)
