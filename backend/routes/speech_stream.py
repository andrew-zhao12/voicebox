"""Incremental text -> queued speech over ``WS /v1/audio/speech/stream``."""

import asyncio
import base64
import contextlib
import json
import logging
import math
import os
import uuid
from collections import deque
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, WebSocket

from .. import lifecycle, models
from ..auth import Principal, charge, get_principal
from ..auth.ratelimit import RateLimited
from ..database import get_db
from ..services import generation as generation_service, live_sessions, profiles
from ..services.speech_stream import (
    MAX_PENDING_FLUSHES,
    MAX_SESSION_FLUSHES,
    MAX_TEXT_CHARS,
    Flush,
    ProtocolError,
    SpeechBuffer,
)
from ..utils import encode

logger = logging.getLogger(__name__)
router = APIRouter()

KIND = "speech"
MAX_MESSAGE_BYTES = 64 * 1024
MAX_INBOUND_MESSAGES = 16
MAX_OUTBOUND_MESSAGES = 16
PCM_CHUNK_BYTES = 4800
SEND_TIMEOUT_S = 5.0
STOP_POLL_S = 0.25


def _setting(name: str, default: float, minimum: float) -> float:
    try:
        value = float(os.getenv(name, default))
        return max(minimum, value) if math.isfinite(value) else default
    except ValueError:
        logger.warning("Invalid %s; using %s", name, default)
        return default


def max_sessions() -> int:
    """Maximum speech sockets in this server process."""
    return int(_setting("VOICEBOX_SPEECH_MAX_SESSIONS", 4, 1))


def idle_timeout_s() -> float:
    """Maximum quiet time when no flush is pending."""
    return _setting("VOICEBOX_SPEECH_IDLE_S", 60, 1)


def max_session_s() -> float:
    """Maximum connection lifetime, including synthesis."""
    return _setting("VOICEBOX_SPEECH_MAX_SESSION_S", 1800, 1)


@dataclass(frozen=True)
class _End:
    reason: str
    close_code: int = 1000


def error_event(code: str, message: str, *, flush_id: str | None = None, retry_after_s: int | None = None) -> dict:
    """Create an explicit protocol error, optionally identifying one failed flush."""
    event = {"type": "error", "error": {"code": code, "message": message}}
    if flush_id is not None:
        event["flush_id"] = flush_id
    if retry_after_s is not None:
        event["error"]["retry_after_s"] = retry_after_s
    return event


class _Connection:
    """Keep control reads responsive while one ordered synthesis worker emits bounded audio."""

    def __init__(self, websocket: WebSocket, principal: Principal):
        self.websocket = websocket
        self.principal = principal
        self.buffer = SpeechBuffer()
        self.pending: deque[Flush] = deque()
        self.unsettled: dict[str, Flush] = {}
        self.work_ready = asyncio.Event()
        self.active: Flush | None = None
        self.active_task: asyncio.Task | None = None
        self.inbound = asyncio.Queue(maxsize=MAX_INBOUND_MESSAGES)
        self.outbound = asyncio.Queue(maxsize=MAX_OUTBOUND_MESSAGES)
        self.last_message = asyncio.get_running_loop().time()
        self.closing = False

    async def emit(self, event: dict) -> None:
        await self.outbound.put(event)

    async def finish(self, flush: Flush, event: dict) -> None:
        """Keep accepted work tracked until its terminal event has entered the send queue."""
        await self.emit(event)
        self.buffer.finish(flush)
        self.unsettled.pop(flush.flush_id, None)

    async def reader(self) -> None:
        try:
            while True:
                message = await self.websocket.receive()
                await self.inbound.put(message)
                if message["type"] == "websocket.disconnect":
                    return
        except Exception:
            await self.inbound.put({"type": "websocket.disconnect"})

    async def sender(self) -> None:
        while True:
            event = await self.outbound.get()
            try:
                async with asyncio.timeout(SEND_TIMEOUT_S):
                    await self.websocket.send_json(event)
            finally:
                self.outbound.task_done()

    async def timer(self) -> _End:
        started = asyncio.get_running_loop().time()
        while True:
            now = asyncio.get_running_loop().time()
            if lifecycle.is_stopping():
                return _End("server_shutting_down", 1012)
            if now - started >= max_session_s():
                return _End("session_too_long")
            if not self.buffer.pending_count and now - self.last_message >= idle_timeout_s():
                return _End("idle_timeout")
            await asyncio.sleep(STOP_POLL_S)

    async def receiver(self) -> _End | None:
        while True:
            message = await self.inbound.get()
            if message["type"] == "websocket.disconnect":
                return None
            self.last_message = asyncio.get_running_loop().time()
            text = message.get("text")
            binary = message.get("bytes")
            if binary is not None and len(binary) > MAX_MESSAGE_BYTES:
                await self.emit(error_event("message_too_large", f"messages are capped at {MAX_MESSAGE_BYTES} bytes"))
                return _End("message_too_large", 1008)
            if text is None:
                await self.emit(error_event("invalid_message", "send JSON text messages, not binary frames"))
                continue
            if len(text.encode("utf-8")) > MAX_MESSAGE_BYTES:
                await self.emit(error_event("message_too_large", f"messages are capped at {MAX_MESSAGE_BYTES} bytes"))
                return _End("message_too_large", 1008)
            flush_id = None
            try:
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise ProtocolError("invalid_message", "a command must be a JSON object")
                value = payload.get("flush_id")
                flush_id = value if isinstance(value, str) and len(value) <= 80 else None
                kind = payload.get("type")
                if kind == "session.close":
                    return _End("client_closed")
                await self.command(kind, payload)
            except json.JSONDecodeError:
                await self.emit(error_event("invalid_json", "message is not valid JSON"))
            except ProtocolError as exc:
                await self.emit(error_event(exc.code, str(exc), flush_id=flush_id))
            except RateLimited as exc:
                await self.emit(
                    error_event(
                        "rate_limit_exceeded",
                        str(exc.detail),
                        flush_id=flush_id,
                        retry_after_s=exc.decision.retry_after_s,
                    )
                )

    async def command(self, kind: str, payload: dict) -> None:
        if kind == "session.update":
            self.buffer.update(payload.get("session"))
            await self.emit({"type": "session.updated", "session": self.buffer.settings})
        elif kind == "input_text.append":
            self.buffer.append(payload.get("text"))
            await self.emit({"type": "input_text.accepted", "buffered_chars": len(self.buffer.text)})
        elif kind == "input_text.flush":
            flush = self.buffer.prepare(payload.get("flush_id"))
            charge("inference")
            self.buffer.commit(flush)
            self.unsettled[flush.flush_id] = flush
            self.pending.append(flush)
            await self.emit({"type": "input_text.flushed", "flush_id": flush.flush_id})
            self.work_ready.set()
        elif kind == "input_text.cancel":
            await self.cancel(payload.get("flush_id"))
        else:
            raise ProtocolError("unknown_event", "unknown command type")

    async def cancel(self, flush_id: str | None) -> None:
        if flush_id is not None and (not isinstance(flush_id, str) or not flush_id):
            raise ProtocolError("invalid_flush_id", "flush_id must identify a pending flush")
        targets = [flush for flush in self.pending if flush_id is None or flush.flush_id == flush_id]
        active = (
            self.active
            if self.active and not self.active.terminal and (flush_id is None or self.active.flush_id == flush_id)
            else None
        )
        if flush_id is not None and not targets and active is None:
            raise ProtocolError("flush_not_pending", "flush_id is not pending")
        for flush in targets:
            self.pending.remove(flush)
            flush.cancelled = True
        # Remove every queued target and stop active synthesis before output backpressure can yield
        # to the worker. Otherwise it could start another target while cancellation is reporting one.
        active_task = self.active_task if active is not None else None
        if active is not None:
            active.cancelled = True
            if active_task is not None:
                active_task.cancel()
        for flush in targets:
            await self.finish(flush, {"type": "audio.cancelled", "flush_id": flush.flush_id})
        if active_task is not None:
            await asyncio.gather(active_task, return_exceptions=True)
        if flush_id is None:
            self.buffer.text = ""
            await self.emit({"type": "input_text.cleared"})

    async def worker(self) -> None:
        while True:
            await self.work_ready.wait()
            if not self.pending:
                self.work_ready.clear()
                continue
            flush = self.pending.popleft()
            self.active = flush
            self.active_task = asyncio.create_task(self.synthesize(flush))
            try:
                result = await self.active_task
            except asyncio.CancelledError:
                if self.closing:
                    raise
                await self.finish(flush, {"type": "audio.cancelled", "flush_id": flush.flush_id})
            except (generation_service.GenerationRefused, HTTPException) as exc:
                detail = exc.detail
                error = detail.get("error", {}) if isinstance(detail, dict) else {}
                code = (
                    "rate_limit_exceeded"
                    if isinstance(exc, RateLimited)
                    else error.get("code") or getattr(exc, "code", None) or f"http_{exc.status_code}"
                )
                retry = int((exc.headers or {}).get("Retry-After", "0")) or None
                await self.finish(
                    flush,
                    error_event(
                        code, error.get("message") or str(detail), flush_id=flush.flush_id, retry_after_s=retry
                    ),
                )
            except Exception:
                logger.exception("Speech flush %s failed", flush.flush_id)
                await self.finish(
                    flush,
                    error_event(
                        "synthesis_failed", "speech synthesis failed; see the server log", flush_id=flush.flush_id
                    ),
                )
            else:
                event = (
                    {"type": "audio.cancelled", "flush_id": flush.flush_id}
                    if flush.cancelled
                    else {"type": "audio.done", "flush_id": flush.flush_id, **result}
                )
                await self.finish(flush, event)
            finally:
                self.active = None
                self.active_task = None

    async def synthesize(self, flush: Flush) -> dict:
        """Use exactly the HTTP speech resolver, queue, encoder, and cancellation cleanup."""
        from .openai_compat import _chunk_overrides, _resolve_model, _resolve_voice

        opened = None
        frames = None
        encoded = None
        finished = False
        try:
            with contextlib.closing(get_db()) as database:
                db = next(database)
                data = flush.request
                engine, model_size = _resolve_model(data.model)
                voice = _resolve_voice(
                    data.voice, self.websocket.headers.get("X-Voicebox-Client-Id"), db, self.principal, engine
                )
                stream_data = models.StreamGenerationRequest(
                    profile_id=voice.id,
                    text=data.input,
                    language=data.language or getattr(voice, "language", None) or "en",
                    engine=engine,
                    model_size=model_size,
                    instruct=data.instructions,
                    format="pcm",
                    speed=data.speed,
                    pronunciation_replacements=data.pronunciation_replacements,
                    **_chunk_overrides(data),
                )
                opened = await generation_service.open_stream(
                    stream_data, db, self.principal, voice=voice if isinstance(voice, profiles.PresetVoice) else None
                )
            await self.emit(
                {
                    "type": "audio.started",
                    "flush_id": flush.flush_id,
                    "job_id": opened.session.job_id,
                    "sample_rate": 24000,
                    "channels": 1,
                    "format": "pcm_s16le",
                    "engine": opened.engine,
                    "voice": voice.id,
                }
            )
            frames = generation_service.stream_frames(opened)
            encoded = encode.encode_stream(frames, opened.sample_rate, "pcm")
            sequence = 0
            samples = 0
            async for chunk in encoded:
                for offset in range(0, len(chunk), PCM_CHUNK_BYTES):
                    pcm = chunk[offset : offset + PCM_CHUNK_BYTES]
                    await self.emit(
                        {
                            "type": "audio.delta",
                            "flush_id": flush.flush_id,
                            "sequence": sequence,
                            "audio": base64.b64encode(pcm).decode("ascii"),
                        }
                    )
                    sequence += 1
                    samples += len(pcm) // 2
            finished = True
            return {"chunks": sequence, "samples": samples}
        finally:
            if encoded is not None:
                await encoded.aclose()
            if frames is not None:
                await frames.aclose()
            if opened is not None and not finished:
                generation_service.abandon_stream(opened.session)


@router.websocket("/v1/audio/speech/stream")
async def speech_stream(websocket: WebSocket) -> None:
    """Accept incremental text and explicit flush/cancel commands; emit ordered PCM events."""
    principal = get_principal()
    await websocket.accept()
    if lifecycle.is_stopping():
        await websocket.send_json(error_event("server_shutting_down", "the server is restarting; reconnect"))
        await websocket.close(code=1012, reason="server_shutting_down")
        return
    if not live_sessions.reserve(
        KIND, principal.key_id, per_key=principal.limits.max_realtime_sessions, max_global=max_sessions()
    ):
        await websocket.send_json(
            error_event("session_limit_reached", "too many live speech/transcription sessions; retry later")
        )
        await websocket.close(code=1013, reason="session_limit_reached")
        return
    connection = _Connection(websocket, principal)
    tasks = []
    end = None
    sender = None
    try:
        sender = asyncio.create_task(connection.sender())
        await connection.emit(
            {
                "type": "session.created",
                "session_id": str(uuid.uuid4()),
                "session": connection.buffer.settings,
                "audio_format": {"format": "pcm_s16le", "sample_rate": 24000, "channels": 1},
                "limits": {
                    "max_text_chars": MAX_TEXT_CHARS,
                    "max_pending_flushes": MAX_PENDING_FLUSHES,
                    "max_session_flushes": MAX_SESSION_FLUSHES,
                    "max_message_bytes": MAX_MESSAGE_BYTES,
                },
            }
        )
        reader = asyncio.create_task(connection.reader())
        receiver = asyncio.create_task(connection.receiver())
        worker = asyncio.create_task(connection.worker())
        timer = asyncio.create_task(connection.timer())
        tasks = [reader, receiver, worker, timer]
        completed, _ = await asyncio.wait([receiver, worker, timer, sender], return_when=asyncio.FIRST_COMPLETED)
        if sender in completed:
            try:
                sender.result()
            except TimeoutError:
                end = _End("slow_consumer", 1013)
            except Exception:
                pass  # A failed write usually means the peer already disconnected.
        elif receiver in completed:
            end = receiver.result()
        elif timer in completed:
            end = timer.result()
        else:
            worker.result()
    except Exception:
        logger.exception("Speech WebSocket failed")
        end = _End("server_error", 1011)
    finally:
        connection.closing = True
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if end is not None and sender is not None and not sender.done():
            with contextlib.suppress(Exception):
                async with asyncio.timeout(SEND_TIMEOUT_S):
                    for flush in list(connection.unsettled.values()):
                        await connection.finish(flush, {"type": "audio.cancelled", "flush_id": flush.flush_id})
                    await connection.emit({"type": "session.closed", "reason": end.reason})
                    await connection.outbound.join()
        if sender is not None:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        if end is not None:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(SEND_TIMEOUT_S):
                    await websocket.close(code=end.close_code, reason=end.reason)
        live_sessions.release(KIND, principal.key_id)
