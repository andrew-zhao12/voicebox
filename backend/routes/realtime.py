"""Live transcription over WebSocket: ``/v1/realtime/transcription``.

The client streams PCM16 audio and receives partial transcripts while the
speaker talks and a final transcript per utterance.  Event names follow the
OpenAI Realtime transcription API so client code ports easily (SDK
compatibility is not promised); ``services/realtime_stt.py`` holds the
state machine, this module holds the socket: authentication happened in the
auth middleware (bearer header or ``?token=``), the session and key caps are
applied here, and every Whisper call goes through the shared
``routes/transcription.py`` helper, so the model rules (client keys never
trigger downloads) and the Whisper slot are the same as for uploads.

Three tasks serve one connection: the receiver parses client events and
feeds the session, the worker runs Whisper on the work the session produces
(partials are skipped while the slot is busy, finals wait for it), and the
sender is the only writer to the socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .. import lifecycle
from ..auth import charge, get_principal
from ..auth.ratelimit import RateLimited
from ..observability import metrics
from ..services import realtime_stt as rt, transcribe
from ..services.inference_slots import whisper_slot
from .openai_compat import WHISPER_ALIASES
from .transcription import transcribe_file_detailed

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_SESSIONS_ENV = "VOICEBOX_REALTIME_MAX_SESSIONS"
DEFAULT_MAX_SESSIONS = 4
MAX_SESSION_S_ENV = "VOICEBOX_REALTIME_MAX_SESSION_S"
DEFAULT_MAX_SESSION_S = 1800.0
IDLE_TIMEOUT_S = 60.0
MAX_MESSAGE_BYTES = 1024 * 1024
# Close codes: policy violation, try again later, service restart.
CLOSE_POLICY = 1008
CLOSE_TRY_LATER = 1013
CLOSE_RESTART = 1012

_sessions_by_key: dict[str, int] = {}


def max_sessions(environ: Mapping[str, str] = os.environ) -> int:
    raw = environ.get(MAX_SESSIONS_ENV, "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_MAX_SESSIONS
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", MAX_SESSIONS_ENV, raw, DEFAULT_MAX_SESSIONS)
        return DEFAULT_MAX_SESSIONS


def max_session_s(environ: Mapping[str, str] = os.environ) -> float:
    raw = environ.get(MAX_SESSION_S_ENV, "").strip()
    try:
        return max(10.0, float(raw)) if raw else DEFAULT_MAX_SESSION_S
    except ValueError:
        logger.warning("%s=%r is not a number; using %.0f", MAX_SESSION_S_ENV, raw, DEFAULT_MAX_SESSION_S)
        return DEFAULT_MAX_SESSION_S


def open_sessions() -> int:
    return sum(_sessions_by_key.values())


def whisper_size(model: str | None) -> str | None:
    """A registry Whisper size from ``whisper-turbo``-style names; ``None`` means the server's current size."""
    from ..backends import WHISPER_HF_REPOS  # lazy: heavy import

    if not model or model in WHISPER_ALIASES:
        return None
    size = model.removeprefix("whisper-")
    if size in WHISPER_HF_REPOS:
        return size
    raise rt.ProtocolError("model_not_found", f"Unknown model '{model}'; use whisper-1 or whisper-<size>")


def check_model(model: str | None) -> None:
    """Raise ``ProtocolError`` when a client key asks for a Whisper size that is not on disk."""
    size = whisper_size(model)
    if get_principal().is_admin:
        return
    whisper = transcribe.get_whisper_model()
    wanted = size or whisper.model_size
    loaded = whisper.is_loaded() and whisper.model_size == wanted
    if not loaded and not whisper._is_model_cached(wanted):
        raise rt.ProtocolError(
            "model_not_downloaded", f"Whisper model {wanted} is not downloaded; ask an admin to download it first."
        )


async def transcribe_pcm(audio: np.ndarray, language: str | None, model: str | None) -> str:
    """Run Whisper on 16 kHz float32 audio through the shared upload path (temp WAV).

    The segment-level path is used on purpose: Whisper pads short audio to
    30 s and hallucinates over the padding, and ``transcribe_detailed`` drops
    the segments that start after the audio ends.
    """
    import soundfile as sf  # lazy: native library

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await asyncio.to_thread(sf.write, tmp_path, audio, rt.TARGET_RATE, subtype="PCM_16")
        transcript = await transcribe_file_detailed(tmp_path, language, whisper_size(model))
        return transcript.text
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class _Connection:
    """One WebSocket session: receiver, worker and sender tasks around a ``TranscriptionSession``."""

    def __init__(self, websocket: WebSocket, session: rt.TranscriptionSession) -> None:
        self.websocket = websocket
        self.session = session
        self.outbound: asyncio.Queue[dict | None] = asyncio.Queue()
        self.work_ready = asyncio.Event()
        self.close_code: int | None = None

    def emit(self, events: list[dict]) -> None:
        for event in events:
            self.outbound.put_nowait(event)
        if self.session.work:
            self.work_ready.set()

    async def sender(self) -> None:
        while True:
            event = await self.outbound.get()
            if event is None:
                return
            await self.websocket.send_json(event)

    async def receiver(self) -> None:
        while True:
            if lifecycle.is_stopping():
                self.emit(
                    [rt.error_event("server_shutting_down", "the server is restarting; reconnect", kind="server_error")]
                )
                self.close_code = CLOSE_RESTART
                return
            try:
                message = await asyncio.wait_for(self.websocket.receive(), timeout=IDLE_TIMEOUT_S)
            except TimeoutError:
                self.emit([rt.error_event("idle_timeout", f"no message for {IDLE_TIMEOUT_S:.0f} s")])
                self.close_code = 1000
                return
            if message["type"] == "websocket.disconnect":
                return
            text = message.get("text")
            data = message.get("bytes")
            if (text is not None and len(text) > MAX_MESSAGE_BYTES) or (
                data is not None and len(data) > MAX_MESSAGE_BYTES
            ):
                self.emit([rt.error_event("message_too_large", f"messages are capped at {MAX_MESSAGE_BYTES} bytes")])
                self.close_code = CLOSE_POLICY
                return
            try:
                if data is not None:
                    self._append(data)  # a binary frame is raw PCM16, a convenience for native clients
                elif text is not None:
                    self._handle_text(text)
            except rt.ProtocolError as e:
                self.emit([rt.error_event(e.code, str(e))])
            except RateLimited as e:
                self.emit([rt.error_event("rate_limit_exceeded", str(e.detail), kind="rate_limit_error")])
                self.close_code = CLOSE_TRY_LATER
                return

    def _handle_text(self, text: str) -> None:
        try:
            payload = json.loads(text)
        except ValueError as e:
            raise rt.ProtocolError("invalid_json", "messages must be JSON objects") from e
        if not isinstance(payload, dict):
            raise rt.ProtocolError("invalid_json", "messages must be JSON objects")
        kind = payload.get("type")
        if kind in ("transcription_session.update", "session.update"):
            config = rt.SessionConfig.from_update(self.session.config, payload)
            if config.model != self.session.config.model:
                check_model(config.model)
            self.emit(self.session.update(config))
        elif kind == "input_audio_buffer.append":
            audio = payload.get("audio")
            if not isinstance(audio, str):
                raise rt.ProtocolError("invalid_audio", "audio must be base64 PCM16")
            self._append(rt.decode_base64_audio(audio))
        elif kind == "input_audio_buffer.commit":
            self.emit(self.session.commit())
        elif kind == "input_audio_buffer.clear":
            self.emit(self.session.clear())
        else:
            raise rt.ProtocolError("unknown_event", f"unknown event type {kind!r}")

    def _append(self, pcm16: bytes) -> None:
        charge("uploads_bytes", len(pcm16))
        self.emit(self.session.append(pcm16))

    async def worker(self) -> None:
        while True:
            await self.work_ready.wait()
            self.work_ready.clear()
            for work in self.session.take_work():
                if work.kind == "partial":
                    if whisper_slot.busy:
                        continue  # a fresher partial will come; finals never skip
                    with contextlib.suppress(HTTPException):
                        hypothesis = await transcribe_pcm(work.audio, work.language, work.model)
                        self.emit(self.session.partial_text(work.item_id, hypothesis))
                    continue
                try:
                    transcript = await transcribe_pcm(work.audio, work.language, work.model)
                except HTTPException as e:
                    code = "whisper_busy" if e.status_code == 429 else "transcription_failed"
                    detail = e.detail if isinstance(e.detail, str) else json.dumps(e.detail)
                    self.emit(self.session.failed(work.item_id, code, detail))
                    continue
                metrics.REALTIME_UTTERANCES.inc()
                self.emit(self.session.final_text(work.item_id, transcript))


async def _refuse(
    websocket: WebSocket, code: str, message: str, close_code: int, *, kind: str = "invalid_request_error"
):
    await websocket.send_json(rt.error_event(code, message, kind=kind))
    await websocket.close(code=close_code)


@router.websocket("/v1/realtime/transcription")
async def realtime_transcription(websocket: WebSocket) -> None:
    """Stream PCM16 in, get ``transcription.delta`` and ``transcription.completed`` events out."""
    principal = get_principal()
    await websocket.accept()
    if lifecycle.is_stopping():
        await _refuse(
            websocket, "server_shutting_down", "the server is restarting; reconnect", CLOSE_RESTART, kind="server_error"
        )
        return
    per_key = principal.limits.max_realtime_sessions
    if open_sessions() >= max_sessions() or (
        per_key is not None and _sessions_by_key.get(principal.key_id, 0) >= per_key
    ):
        await _refuse(
            websocket, "session_limit_reached", "too many live transcription sessions; retry later", CLOSE_TRY_LATER
        )
        return
    try:
        charge("inference")
        check_model(None)
    except RateLimited as e:
        await _refuse(websocket, "rate_limit_exceeded", str(e.detail), CLOSE_TRY_LATER, kind="rate_limit_error")
        return
    except rt.ProtocolError as e:
        await _refuse(websocket, e.code, str(e), CLOSE_POLICY)
        return

    session = rt.TranscriptionSession(rt.SessionConfig())
    connection = _Connection(websocket, session)
    _sessions_by_key[principal.key_id] = _sessions_by_key.get(principal.key_id, 0) + 1
    metrics.REALTIME_SESSIONS.set(open_sessions())
    connection.emit([{"type": "transcription_session.created", "session": session.config.to_event()}])
    sender = asyncio.create_task(connection.sender())
    worker = asyncio.create_task(connection.worker())
    try:
        async with asyncio.timeout(max_session_s()):
            await connection.receiver()
    except TimeoutError:
        connection.emit([rt.error_event("session_too_long", f"sessions are capped at {max_session_s():.0f} s")])
        connection.close_code = 1000
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Live transcription session failed")
        connection.close_code = 1011
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await worker
        connection.outbound.put_nowait(None)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(sender, timeout=5)
        if connection.close_code is not None:
            with contextlib.suppress(Exception):
                await websocket.close(code=connection.close_code)
        remaining = _sessions_by_key.get(principal.key_id, 0) - 1
        if remaining > 0:
            _sessions_by_key[principal.key_id] = remaining
        else:
            _sessions_by_key.pop(principal.key_id, None)
        metrics.REALTIME_SESSIONS.set(open_sessions())
