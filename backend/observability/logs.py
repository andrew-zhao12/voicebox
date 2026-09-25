"""One-line JSON logs (``VOICEBOX_LOG_FORMAT=json``) carrying the request id and the caller's key id."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from datetime import UTC, datetime

from ..auth.principal import principal_var
from .requestid import request_id_var

ENV_VAR = "VOICEBOX_LOG_FORMAT"
# Loggers uvicorn configures with its own handlers (they do not propagate to the root).
_LOGGER_NAMES = ("", "uvicorn", "uvicorn.error", "uvicorn.access")


class JsonFormatter(logging.Formatter):
    """``{"ts","level","logger","msg",...}`` per record; access-log fields are split out."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        principal = principal_var.get()
        if principal is not None and principal.authenticated:
            payload["key_id"] = principal.key_id
        for attr in ("error_id",):
            value = getattr(record, attr, None)
            if value is not None:
                payload[attr] = value
        args = record.args
        if record.name == "uvicorn.access" and isinstance(args, tuple) and len(args) == 5:
            client, method, path, _version, status = args
            payload.update({"client": client, "method": method, "path": path, "status": status})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def json_logging_enabled(environ: Mapping[str, str] = os.environ) -> bool:
    return environ.get(ENV_VAR, "").strip().lower() == "json"


def apply_json_logging() -> int:
    """Put the JSON formatter on every handler of the root and uvicorn loggers; returns how many."""
    count = 0
    for name in _LOGGER_NAMES:
        for handler in logging.getLogger(name).handlers:
            if not isinstance(handler.formatter, JsonFormatter):
                handler.setFormatter(JsonFormatter())
            count += 1
    return count
