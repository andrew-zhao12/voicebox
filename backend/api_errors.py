"""OpenAI-style error envelopes for the ``/v1`` routes.

Everything under ``/v1/`` answers errors as
``{"error": {"message", "type", "param", "code"}}``, which is what OpenAI
SDKs parse and surface; every other route keeps FastAPI's ``{"detail": ...}``.
The auth middleware, the generic 500 handler and the ``/v1`` router's own
exception handlers all build their bodies here, so a 401 from the middleware
and a 404 from the router look the same to an OpenAI client.
"""

from __future__ import annotations

import json
from typing import Any

OPENAI_PREFIX = "/v1/"

_TYPES = {401: "authentication_error", 403: "permission_error", 429: "rate_limit_error"}
_CODES = {
    401: "invalid_api_key",
    403: "insufficient_permissions",
    404: "not_found",
    413: "request_too_large",
    429: "rate_limit_exceeded",
    503: "server_draining",
}


def is_openai_path(path: str) -> bool:
    return path == "/v1" or path.startswith(OPENAI_PREFIX)


def openai_error(status: int, message: str, *, code: str | None = None, param: str | None = None) -> dict:
    """The envelope for one error."""
    error_type = "server_error" if status >= 500 else _TYPES.get(status, "invalid_request_error")
    return {"error": {"message": message, "type": error_type, "param": param, "code": code or _CODES.get(status)}}


def error_body(path: str, status: int, detail: Any, *, code: str | None = None, param: str | None = None) -> dict:
    """``{"detail": ...}`` normally, the OpenAI envelope under ``/v1/``."""
    if not is_openai_path(path):
        return {"detail": detail}
    if isinstance(detail, dict) and "error" in detail:
        return detail
    message = detail if isinstance(detail, str) else json.dumps(detail)
    return openai_error(status, message, code=code, param=param)
