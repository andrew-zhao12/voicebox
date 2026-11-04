"""Correlation fields carried across socket tasks and queued generation work."""

from contextvars import ContextVar

session_id_var: ContextVar[str | None] = ContextVar("voicebox_session_id", default=None)
flush_id_var: ContextVar[str | None] = ContextVar("voicebox_flush_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("voicebox_job_id", default=None)
