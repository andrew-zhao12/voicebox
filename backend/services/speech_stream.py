"""Bounded incremental-text protocol state for the speech WebSocket."""

import re
from dataclasses import dataclass

from pydantic import ValidationError

from .. import models

MAX_TEXT_CHARS = 10_000
MAX_PENDING_FLUSHES = 4
MAX_SESSION_FLUSHES = 1000
FLUSH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,79}\Z")


class ProtocolError(ValueError):
    """A recoverable command error that leaves existing speech intact."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class Flush:
    """One immutable text/settings snapshot, with lifecycle state owned by the connection."""

    flush_id: str
    request: models.OpenAISpeechRequest
    cancelled: bool = False
    terminal: bool = False


class SpeechBuffer:
    """Bound buffered and submitted text while preventing ambiguous flush ID reuse."""

    def __init__(self):
        self.text = ""
        self.settings = models.OpenAISpeechRequest(input=".", voice="af_heart", response_format="pcm").model_dump(
            exclude={"input", "response_format"}
        )
        self.seen: set[str] = set()
        self.pending_chars = 0
        self.pending_count = 0

    def update(self, settings: dict) -> None:
        """Validate a partial configuration update; submitted flushes keep their snapshots."""
        if not isinstance(settings, dict) or set(settings) - set(self.settings):
            raise ProtocolError("invalid_session", "session must contain only documented speech settings")
        try:
            request = models.OpenAISpeechRequest(**(self.settings | settings), input=".", response_format="pcm")
        except ValidationError as exc:
            raise ProtocolError("invalid_session", exc.errors()[0]["msg"]) from exc
        self.settings = request.model_dump(exclude={"input", "response_format"})

    def append(self, text: str) -> None:
        """Append text without synthesizing or charging inference yet."""
        if not isinstance(text, str) or not text:
            raise ProtocolError("invalid_text", "text must be a non-empty string")
        if len(self.text) + self.pending_chars + len(text) > MAX_TEXT_CHARS:
            raise ProtocolError(
                "text_limit_exceeded", f"buffered and pending text is capped at {MAX_TEXT_CHARS} characters"
            )
        self.text += text

    def prepare(self, flush_id: str) -> Flush:
        """Validate a flush without changing the buffer, so a rejected charge loses no text."""
        if not isinstance(flush_id, str) or not FLUSH_ID.fullmatch(flush_id):
            raise ProtocolError(
                "invalid_flush_id", "flush_id must be 1-80 ASCII letters, digits, '.', '_', ':', or '-'"
            )
        if flush_id in self.seen:
            raise ProtocolError("duplicate_flush_id", "flush_id was already used in this session")
        if len(self.seen) >= MAX_SESSION_FLUSHES:
            raise ProtocolError("flush_limit_exceeded", "the session has reached its flush limit; reconnect")
        if self.pending_count >= MAX_PENDING_FLUSHES:
            raise ProtocolError("pending_limit_exceeded", f"at most {MAX_PENDING_FLUSHES} flushes may be pending")
        if not self.text.strip():
            raise ProtocolError("empty_text", "append non-whitespace text before flushing")
        request = models.OpenAISpeechRequest(**self.settings, input=self.text, response_format="pcm")
        return Flush(flush_id, request)

    def commit(self, flush: Flush) -> None:
        """Record an accepted flush after its inference charge succeeds."""
        self.seen.add(flush.flush_id)
        self.pending_chars += len(flush.request.input)
        self.pending_count += 1
        self.text = ""

    def finish(self, flush: Flush) -> None:
        """Release the text budget exactly once when a flush ends."""
        if not flush.terminal:
            self.pending_chars -= len(flush.request.input)
            self.pending_count -= 1
            flush.terminal = True
