"""Generic 500 for unexpected exceptions so tracebacks and paths stay in the log."""

from __future__ import annotations

import logging
import secrets
import sys
from importlib.util import resolve_name

from fastapi import Request
from fastapi.responses import JSONResponse

from ..api_errors import error_body

logger = logging.getLogger(__name__)

# Looked up in ``sys.modules`` instead of imported: the generation service pulls
# in the audio stack, which the torch-free test app lacks, and nothing can have
# raised its error unless it is loaded.
_GENERATION_MODULE = resolve_name("..services.generation", __package__)


def _is_stream_abort(exc: Exception) -> bool:
    """Whether *exc* is ``services.generation.StreamAbortedError``."""
    aborted = getattr(sys.modules.get(_GENERATION_MODULE), "StreamAbortedError", None)
    return aborted is not None and isinstance(exc, aborted)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # The request id doubles as the error id, so a client-reported id finds the log line.
    error_id = request.scope.get("state", {}).get("request_id") or secrets.token_hex(8)
    if _is_stream_abort(exc):
        # The job has logged the engine's traceback and uvicorn logs this one
        # when it drops the connection, so a third copy would only be noise.
        logger.warning(
            "Streamed response %s aborted: %s %s: %s",
            error_id,
            request.method,
            request.url.path,
            exc,
            extra={"error_id": error_id},
        )
    else:
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
