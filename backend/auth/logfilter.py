"""Keep media tokens and OAuth codes out of uvicorn's logs."""

from __future__ import annotations

import logging
import re

_SECRET_QUERY_RE = re.compile(r"([?&](?:token|code|state)=)[^&\s]*")
# uvicorn logs HTTP requests on its access logger and WebSocket handshakes
# ('%s - "WebSocket %s" [accepted]', or 403) on its error logger.
_LOGGERS = ("uvicorn.access", "uvicorn.error")


class RedactQueryTokenFilter(logging.Filter):
    """Rewrite ``token=``/``code=``/``state=`` query values in a record's string arguments.

    uvicorn passes the path with its query string as an argument: access
    records carry ``(client_addr, method, full_path, http_version, status)``,
    WebSocket handshake records ``(client_addr, full_path[, status])``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and any(isinstance(arg, str) and "=" in arg for arg in args):
            record.args = tuple(
                _SECRET_QUERY_RE.sub(r"\1[redacted]", arg) if isinstance(arg, str) else arg for arg in args
            )
        return True


def install_log_redaction() -> None:
    for name in _LOGGERS:
        target = logging.getLogger(name)
        if not any(isinstance(existing, RedactQueryTokenFilter) for existing in target.filters):
            target.addFilter(RedactQueryTokenFilter())
