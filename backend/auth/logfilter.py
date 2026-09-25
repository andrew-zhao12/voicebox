"""Keep media tokens and OAuth codes out of uvicorn's access log."""

from __future__ import annotations

import logging
import re

_SECRET_QUERY_RE = re.compile(r"([?&](?:token|code|state)=)[^&\s]*")


class RedactQueryTokenFilter(logging.Filter):
    """Rewrite ``token=``/``code=``/``state=`` query values in access-log records.

    uvicorn logs ``(client_addr, method, full_path, http_version, status)``;
    the path is the third argument.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) == 5 and isinstance(args[2], str) and "=" in args[2]:
            record.args = (*args[:2], _SECRET_QUERY_RE.sub(r"\1[redacted]", args[2]), *args[3:])
        return True


def install_access_log_redaction() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(existing, RedactQueryTokenFilter) for existing in access_logger.filters):
        access_logger.addFilter(RedactQueryTokenFilter())
