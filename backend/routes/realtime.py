"""Live transcription over WebSocket: ``/v1/realtime`` and ``/v1/realtime/transcription``.

The client streams PCM16 or G.711 audio and receives partial transcripts
while the speaker talks and a final transcript per utterance.
``/v1/realtime?intent=transcription`` speaks OpenAI's GA Realtime
transcription protocol, so the OpenAI SDK and LiveKit's
``openai.STT(use_realtime=True)`` connect unmodified;
``/v1/realtime/transcription`` keeps Voicebox's own dialect (the beta
``transcription_session.*`` events).  ``services/realtime_stt.py`` holds the
state machine, this module holds the socket: authentication happened in the
auth middleware (bearer header or ``?token=``), the session and key caps are
applied here, and every Whisper call goes through the shared
``routes/transcription.py`` helper, so the model rules (client keys never
trigger downloads) and the Whisper slot are the same as for uploads.

Each connection runs a reader (the only caller of ``receive()``), the
receiver that parses client events and feeds the session, a worker that
runs Whisper on the work the session produces (partials are skipped while
the slot is busy, finals wait for it), and the sender, the only writer to
the socket.  Before the server ends a session (idle, too long, shutting
down) it commits the open utterance and delivers the pending finals.
OpenAI clients treat any ``error`` event as fatal, so that dialect
announces only the closes no reconnect gets past (1008); the others close
with the error code as the reason, and the client reconnects.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket
from starlette.types import Message

from .. import lifecycle
from ..auth import Principal, charge, get_principal
from ..auth.ratelimit import RateLimited
from ..observability import metrics
from ..services import live_sessions, realtime_stt as rt, transcribe
from ..services.inference_slots import InferenceBusyError, whisper_slot
from ..services.native_stt import NativeTranscriber
from ..utils.subtitles import Transcript, to_verbose_json
from .openai_compat import WHISPER_ALIASES
from .transcription import transcribe_file_detailed

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_SESSIONS_ENV = "VOICEBOX_REALTIME_MAX_SESSIONS"
DEFAULT_MAX_SESSIONS = 4
MAX_SESSION_S_ENV = "VOICEBOX_REALTIME_MAX_SESSION_S"
DEFAULT_MAX_SESSION_S = 1800.0
IDLE_S_ENV = "VOICEBOX_REALTIME_IDLE_S"
DEFAULT_IDLE_S = 60.0
# LiveKit's RecognizeStream starts its retry count over only after an attempt
# outlived conn_options.timeout (10 s by default): shorter idle closes count as
# consecutive failures, and the fourth in a row (max_retry=3) ends the agent's STT.
MIN_IDLE_S = 15.0
MAX_MESSAGE_BYTES = 1024 * 1024
# Keep the socket's backpressure: unread audio must not escape the server's
# receive queue into an unbounded application queue before limits are checked.
MAX_INBOUND_MESSAGES = 32
# How long pending finals may take once the server ends a session.
FLUSH_TIMEOUT_S = 10.0
# A quiet socket wakes this often to notice that the server is stopping.
STOP_POLL_S = 0.5
# Close codes: normal, policy violation, internal error, service restart, try again later.
CLOSE_NORMAL = 1000
CLOSE_POLICY = 1008
CLOSE_ERROR = 1011
CLOSE_RESTART = 1012
CLOSE_TRY_LATER = 1013

_sessions_by_key = live_sessions.counts_for_kind("transcription")


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


def idle_timeout_s(environ: Mapping[str, str] = os.environ) -> float:
    """Seconds without a client message before the server ends the session."""
    raw = environ.get(IDLE_S_ENV, "").strip()
    try:
        return max(MIN_IDLE_S, float(raw)) if raw else DEFAULT_IDLE_S
    except ValueError:
        logger.warning("%s=%r is not a number; using %.0f", IDLE_S_ENV, raw, DEFAULT_IDLE_S)
        return DEFAULT_IDLE_S


def open_sessions() -> int:
    return sum(_sessions_by_key.values())


def whisper_size(model: str | None) -> str | None:
    """A registry Whisper size from ``whisper-turbo``-style names; ``None`` means the server's current size."""
    from ..backends import WHISPER_HF_REPOS, get_stt_model_configs  # lazy: heavy import

    if not model or model in WHISPER_ALIASES:
        return None
    size = model.removeprefix("whisper-")
    if size in WHISPER_HF_REPOS:
        return size
    if any(cfg.model_name == model for cfg in get_stt_model_configs()):
        return model
    raise rt.ProtocolError("model_not_found", f"Unknown model '{model}'; GET /v1/models lists STT model IDs")


def check_model(model: str | None) -> None:
    """Refuse unavailable runtimes and uncached models before accepting client audio."""
    size = whisper_size(model)
    whisper, wanted, cfg = transcribe.resolve_stt(size)
    if hasattr(whisper, "runtime_available") and not whisper.runtime_available():
        raise rt.ProtocolError("runtime_unavailable", f"{cfg.engine} runtime is not installed or configured")
    if get_principal().is_admin:
        return
    loaded = whisper.is_loaded() and whisper.model_size == wanted
    if not loaded and not whisper._is_model_cached(wanted):
        raise rt.ProtocolError(
            "model_not_downloaded", f"{cfg.model_name} is not downloaded; ask an admin to download it first."
        )


async def transcribe_pcm(audio: np.ndarray, language: str | None, model: str | None, **options) -> str | Transcript:
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
        # A live session already snapshots caller defaults at connection time.
        # Do not reintroduce saved hints after an explicit session-level clear.
        transcript = await transcribe_file_detailed(
            tmp_path, language, whisper_size(model), use_saved_settings=False, **options
        )
        return transcript if options.get("word_timestamps") else transcript.text
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@dataclass(frozen=True)
class _End:
    """Why the server ends (or refuses) a session; ``code`` is also the close reason."""

    code: str
    message: str
    close_code: int
    kind: str = "invalid_request_error"
    flush: bool = False  # deliver the pending finals before closing

    def announced(self, dialect: rt.Dialect) -> bool:
        """Whether an ``error`` event precedes the close.

        OpenAI clients (the SDK, LiveKit) treat every ``error`` as fatal and
        reconnect after a bare close, so that dialect announces only policy
        closes, which no reconnect gets past.  Internal failures never are.
        """
        if self.close_code == CLOSE_ERROR:
            return False
        return dialect == "voicebox" or self.close_code == CLOSE_POLICY

    def event(self) -> dict:
        return rt.error_event(self.code, self.message, kind=self.kind)


def _shutting_down() -> _End:
    message = "the server is restarting; reconnect"
    return _End("server_shutting_down", message, CLOSE_RESTART, "server_error", flush=True)


def _flush_timeout_s() -> float:
    """``FLUSH_TIMEOUT_S``, cut to the shutdown drain while the server is stopping."""
    if lifecycle.is_stopping():
        return min(FLUSH_TIMEOUT_S, lifecycle.drain_timeout_s())
    return FLUSH_TIMEOUT_S


def _validate_recognition_options(config: rt.SessionConfig) -> None:
    """Reject unsupported effective options before any audio reaches a recognizer."""
    if config.prompt or config.hotwords or config.word_timestamps:
        try:
            _backend, _size, cfg = transcribe.resolve_stt(config.model)
            transcribe.validate_options(
                cfg, prompt=config.prompt, hotwords=config.hotwords, word_timestamps=config.word_timestamps
            )
        except ValueError as exc:
            raise rt.ProtocolError("unsupported_value", str(exc)) from exc


class _Connection:
    """One WebSocket session: reader, receiver, worker and sender tasks around a ``TranscriptionSession``."""

    def __init__(self, websocket: WebSocket, session: rt.TranscriptionSession) -> None:
        self.websocket = websocket
        self.session = session
        self.inbound: asyncio.Queue[Message] = asyncio.Queue(maxsize=MAX_INBOUND_MESSAGES)
        self.outbound: asyncio.Queue[dict | None] = asyncio.Queue()
        self.work_ready = asyncio.Event()
        self.settled = asyncio.Event()  # set while no final is pending
        self.settled.set()
        self.finishing = False  # the server is ending the session: partials stop
        self.native = NativeTranscriber()
        self._options_checked = False

    def emit(self, events: list[dict]) -> None:
        for event in events:
            self.outbound.put_nowait(event)
        if self.session.work:
            self.work_ready.set()
        if self.session.pending_finals:
            self.settled.clear()
        else:
            self.settled.set()

    async def reader(self) -> None:
        """Move client messages to ``inbound``.

        The only caller of ``receive()``: a timed wait on ``inbound`` cannot
        lose a message, which a cancelled ``receive()`` can on some servers.
        """
        try:
            while True:
                message = await self.websocket.receive()
                await self.inbound.put(message)
                if message["type"] == "websocket.disconnect":
                    return
        except Exception:
            logger.debug("Live transcription socket read failed", exc_info=True)
            await self.inbound.put({"type": "websocket.disconnect", "code": 1006})

    async def sender(self) -> None:
        while True:
            event = await self.outbound.get()
            if event is None:
                return
            await self.websocket.send_json(event)

    async def receiver(self, idle_s: float) -> _End | None:
        """Feed client events to the session; ``None`` when the client leaves, else why the server ends it."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + idle_s
        while True:
            if lifecycle.is_stopping():
                return _shutting_down()
            remaining = deadline - loop.time()
            if remaining <= 0:
                return _End("idle_timeout", f"no message for {idle_s:g} s", CLOSE_NORMAL, flush=True)
            try:
                # Short waits, so a quiet socket notices when the server starts stopping.
                message = await asyncio.wait_for(self.inbound.get(), timeout=min(remaining, STOP_POLL_S))
            except TimeoutError:
                continue
            deadline = loop.time() + idle_s
            if message["type"] == "websocket.disconnect":
                return None
            text = message.get("text")
            data = message.get("bytes")
            if (text is not None and len(text) > MAX_MESSAGE_BYTES) or (
                data is not None and len(data) > MAX_MESSAGE_BYTES
            ):
                return _End("message_too_large", f"messages are capped at {MAX_MESSAGE_BYTES} bytes", CLOSE_POLICY)
            try:
                if data is not None:
                    self._append(data)  # a binary frame is raw audio in the session's format, for native clients
                elif text is not None:
                    self._handle_text(text)
            except rt.ProtocolError as e:
                self.emit([rt.error_event(e.code, str(e), client_event_id=e.event_id)])
            except RateLimited as e:
                return _End("rate_limit_exceeded", str(e.detail), CLOSE_TRY_LATER, "rate_limit_error")

    def _handle_text(self, text: str) -> None:
        try:
            payload = json.loads(text)
        except ValueError as e:
            raise rt.ProtocolError("invalid_json", "messages must be JSON objects") from e
        if not isinstance(payload, dict):
            raise rt.ProtocolError("invalid_json", "messages must be JSON objects")
        try:
            self._dispatch(payload)
        except rt.ProtocolError as e:
            event_id = payload.get("event_id")
            if isinstance(event_id, str):
                e.event_id = event_id
            raise

    def _dispatch(self, payload: dict) -> None:
        kind = payload.get("type")
        if kind in ("transcription_session.update", "session.update"):
            config = rt.SessionConfig.from_update(self.session.config, payload)
            if config.model != self.session.config.model:
                check_model(config.model)
            _validate_recognition_options(config)
            self.emit(self.session.update(config))
            self._options_checked = True
        elif kind == "input_audio_buffer.append":
            audio = payload.get("audio")
            if not isinstance(audio, str):
                raise rt.ProtocolError("invalid_audio", "audio must be base64-encoded")
            self._append(rt.decode_base64_audio(audio))
        elif kind == "input_audio_buffer.commit":
            self.emit(self.session.commit())
        elif kind == "input_audio_buffer.clear":
            self.emit(self.session.clear())
            self.work_ready.set()
        else:
            raise rt.ProtocolError("unknown_event", f"unknown event type {kind!r}")

    def _append(self, data: bytes) -> None:
        if not self._options_checked:
            # Let the client override saved defaults before checking model
            # capabilities, but never buffer audio under unsupported settings.
            _validate_recognition_options(self.session.config)
            self._options_checked = True
        charge("uploads_bytes", len(data))
        self.emit(self.session.append(data))

    async def worker(self) -> None:
        while True:
            await self.work_ready.wait()
            self.work_ready.clear()
            for work in self.session.take_work():
                if work.kind == "final":
                    await self._final(work)
                elif not self.finishing and not whisper_slot.busy:
                    await self._partial(work)  # skipped while busy: a fresher partial will come
            await self.native.prune(self.session.active_item_ids)

    async def _transcribe(self, work):
        if self.native.supports(work.model):
            return await self.native.transcribe(work)
        options = {}
        if work.prompt:
            options["prompt"] = work.prompt
        if work.hotwords:
            options["hotwords"] = work.hotwords
        if work.word_timestamps and work.kind == "final":
            options["word_timestamps"] = True
        return await transcribe_pcm(work.audio, work.language, work.model, **options)

    async def _partial(self, work: rt.Work) -> None:
        try:
            hypothesis = await self._transcribe(work)
        except (HTTPException, InferenceBusyError):
            return
        except Exception:
            logger.exception("Live transcription partial failed")
            return
        self.emit(self.session.partial_text(work.item_id, hypothesis))

    async def _final(self, work: rt.Work) -> None:
        try:
            transcript = await self._transcribe(work)
        except InferenceBusyError as e:
            self.emit(self.session.failed(work.item_id, "whisper_busy", str(e)))
            return
        except HTTPException as e:
            code = "whisper_busy" if e.status_code == 429 else "transcription_failed"
            detail = e.detail if isinstance(e.detail, str) else json.dumps(e.detail)
            self.emit(self.session.failed(work.item_id, code, detail))
            return
        except Exception:
            # A worker that died here would leave the session deaf; the utterance fails instead.
            logger.exception("Live transcription failed")
            self.emit(self.session.failed(work.item_id, "transcription_failed", "Transcription failed"))
            return
        metrics.REALTIME_UTTERANCES.inc()
        if isinstance(transcript, Transcript):
            words = to_verbose_json(transcript, include_words=True)["words"] if work.word_timestamps else None
            self.emit(self.session.final_text(work.item_id, transcript.text, words=words))
        else:
            self.emit(self.session.final_text(work.item_id, transcript))

    async def finish(self, timeout_s: float) -> bool:
        """Commit the open utterance and deliver the pending finals; False when the client left meanwhile.

        Partials stop and client messages are dropped from here on.  Finals
        still pending after *timeout_s* are reported as failed.
        """
        self.finishing = True
        with contextlib.suppress(rt.ProtocolError):  # nothing to commit
            self.emit(self.session.commit())
        if self.session.pending_finals:
            settled = asyncio.ensure_future(self.settled.wait())
            gone = asyncio.ensure_future(self._until_disconnect())
            await asyncio.wait((settled, gone), timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
            left = gone.done()
            for task in (settled, gone):
                task.cancel()
            await asyncio.gather(settled, gone, return_exceptions=True)
            if left:
                return False
        self.emit(self.session.abandon("session_closed", "the session ended before this utterance was transcribed"))
        return True

    async def _until_disconnect(self) -> None:
        """Discard client messages until the client disconnects."""
        while (await self.inbound.get())["type"] != "websocket.disconnect":
            pass


def _admit(principal: Principal, model: str | None) -> _End | None:
    """Why a new session is refused, or ``None`` once it is admitted and charged one ``inference``."""
    if lifecycle.is_stopping():
        return _shutting_down()
    per_key = principal.limits.max_realtime_sessions
    if open_sessions() >= max_sessions() or (
        per_key is not None and live_sessions.open_for_key(principal.key_id) >= per_key
    ):
        return _End("session_limit_reached", "too many live transcription sessions; retry later", CLOSE_TRY_LATER)
    try:
        charge("inference")
        check_model(model)
    except RateLimited as e:
        return _End("rate_limit_exceeded", str(e.detail), CLOSE_TRY_LATER, "rate_limit_error")
    except rt.ProtocolError as e:
        return _End(e.code, str(e), CLOSE_POLICY)
    return None


async def _refuse(websocket: WebSocket, dialect: rt.Dialect, end: _End) -> None:
    if end.announced(dialect):
        await websocket.send_json(end.event())
    await websocket.close(code=end.close_code, reason=end.code)


async def _serve(websocket: WebSocket, dialect: rt.Dialect) -> None:
    """Run one session until the client leaves or the server ends it; ``?model=`` seeds its model."""
    principal = get_principal()
    model = websocket.query_params.get("model") or None
    await websocket.accept()
    refusal = _admit(principal, model)
    if refusal is not None:
        await _refuse(websocket, dialect, refusal)
        return

    from ..services.speech_settings import resolve_transcription_hints

    prompt, hotwords = resolve_transcription_hints(None, None, principal=principal)
    session = rt.TranscriptionSession(
        rt.SessionConfig(model=model, prompt=prompt, hotwords=tuple(hotwords)), dialect=dialect
    )
    connection = _Connection(websocket, session)
    live_sessions.reserve(
        "transcription", principal.key_id, per_key=principal.limits.max_realtime_sessions, max_global=max_sessions()
    )
    metrics.REALTIME_SESSIONS.set(open_sessions())
    connection.emit(session.created())
    reader = asyncio.create_task(connection.reader())
    sender = asyncio.create_task(connection.sender())
    worker = asyncio.create_task(connection.worker())
    end: _End | None = None
    try:
        cap_s = max_session_s()
        try:
            async with asyncio.timeout(cap_s):
                end = await connection.receiver(idle_timeout_s())
        except TimeoutError:
            end = _End("session_too_long", f"sessions are capped at {cap_s:g} s", CLOSE_NORMAL, flush=True)
        if end is not None and end.flush and not await connection.finish(_flush_timeout_s()):
            end = None  # the client left while the finals were delivered
    except Exception:
        logger.exception("Live transcription session failed")
        end = _End("server_error", "the session failed", CLOSE_ERROR, "server_error")
    finally:
        for task in (reader, worker):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await connection.native.prune(set())
        if end is not None and end.announced(dialect):
            connection.emit([end.event()])
        connection.outbound.put_nowait(None)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(sender, timeout=5)
        if end is not None:
            with contextlib.suppress(Exception):
                await websocket.close(code=end.close_code, reason=end.code)
        live_sessions.release("transcription", principal.key_id)
        metrics.REALTIME_SESSIONS.set(open_sessions())


@router.websocket("/v1/realtime")
async def realtime_openai(websocket: WebSocket) -> None:
    """OpenAI's GA Realtime transcription protocol: ``?intent=transcription[&model=...]``.

    Conversation sessions are not served: any other intent is refused.
    """
    if websocket.query_params.get("intent") != "transcription":
        await websocket.accept()
        refusal = _End("unsupported_intent", "only ?intent=transcription sessions are served", CLOSE_POLICY)
        await _refuse(websocket, "openai", refusal)
        return
    await _serve(websocket, "openai")


@router.websocket("/v1/realtime/transcription")
async def realtime_transcription(websocket: WebSocket) -> None:
    """Stream audio in, get ``transcription.delta`` and ``transcription.completed`` events out (Voicebox dialect)."""
    await _serve(websocket, "voicebox")
