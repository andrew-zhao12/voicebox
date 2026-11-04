"""Incremental Voicebox speech for the pinned Pipecat 1.15 audio-context API."""

import asyncio
import base64
import copy
import json
import re
import uuid
from collections import deque
from dataclasses import dataclass, field

import aiohttp
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.tts_service import TextAggregationMode, TTSService

from services import SAMPLE_RATE, api_url, require_key
from tts_settings import VoiceboxTTSSettings, complete_settings, speech_options, validate_settings

PCM_FORMAT = {"format": "pcm_s16le", "sample_rate": SAMPLE_RATE, "channels": 1}
MAX_PENDING = 4
MAX_PHRASE_CHARS = 1000
MAX_BUFFERED_FRAMES = 64


@dataclass
class _Turn:
    settings: VoiceboxTTSSettings
    text: str = ""
    pending: set[str] = field(default_factory=set)
    ended: bool = False
    cancelled: bool = False


@dataclass
class _Flush:
    context_id: str
    sequence: int = 0
    started: bool = False
    received_audio: bool = False
    samples: int = 0


class VoiceboxStreamingTTSService(TTSService):
    """Stream bounded phrases from LLM tokens over one authenticated speech socket.

    Text stays local until a phrase boundary, then append+flush are sent atomically.
    This permits targeted cancellation without clearing another context's text.
    Voicebox does not synthesize appended text before a flush. No text is replayed
    after an error, even when some of its audio has already been delivered.
    """

    Settings = VoiceboxTTSSettings

    def __init__(
        self, *, api_key, base_url, voice="af_heart", model="tts-1", settings=None, timeout=120
    ):
        if not 1 <= timeout <= 1800:
            raise ValueError("VOICEBOX_TTS_TIMEOUT must be between 1 and 1800 seconds")
        super().__init__(
            sample_rate=SAMPLE_RATE,
            text_aggregation_mode=TextAggregationMode.TOKEN,
            push_start_frame=True,
            push_stop_frames=False,
            stop_frame_timeout_s=timeout + 10,
            settings=complete_settings(model, voice, settings),
        )
        self._key = require_key(api_key)
        self._timeout = timeout
        self._url = api_url(base_url) + "/audio/speech/stream"
        self._session = None
        self._ws = None
        self._reader = None
        self._connect_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(MAX_PENDING)
        self._turns: dict[str, _Turn] = {}
        self._pending: dict[str, _Flush] = {}
        self._retired = deque(maxlen=64)
        self._failed = False
        self._closing = False
        self._flush_count = 0

    def can_generate_metrics(self):
        return True

    @property
    def supports_processing_metrics(self):
        # run_tts only uploads text; audio arrives asynchronously.
        return False

    async def _update_settings(self, delta):
        candidate = copy.deepcopy(self._settings)
        candidate.apply_update(delta)
        validate_settings(candidate)
        return await super()._update_settings(delta)

    async def on_turn_context_created(self, context_id):
        # Snapshot settings so a mid-turn update cannot change half an utterance.
        for previous, turn in list(self._turns.items()):
            if not turn.pending and not turn.text and not self.audio_context_available(previous):
                self._turns.pop(previous, None)
        self._turns[context_id] = _Turn(copy.deepcopy(self._settings))

    async def on_turn_context_completed(self):
        context_id = self._turn_context_id
        await super().on_turn_context_completed()
        # Empty LLM turns never open an audio context, so the base class does not
        # call flush_audio for them.
        turn = self._turns.get(context_id)
        if turn and not turn.pending and not turn.text:
            self._turns.pop(context_id, None)

    async def run_tts(self, text, context_id):
        """Consume tokens directly, without the SDK's default sentence aggregation."""
        turn = self._turns.setdefault(context_id, _Turn(copy.deepcopy(self._settings)))
        try:
            if not turn.cancelled and not self._failed:
                await self.start_tts_usage_metrics(text)
                # A caller may provide one very large frame. Consume it in bounded slices.
                for offset in range(0, len(text), MAX_PHRASE_CHARS):
                    turn.text += text[offset : offset + MAX_PHRASE_CHARS]
                    while len(turn.text) >= MAX_PHRASE_CHARS:
                        boundary = turn.text.rfind(" ", 0, MAX_PHRASE_CHARS)
                        await self._flush(
                            context_id, boundary + 1 if boundary > 0 else MAX_PHRASE_CHARS
                        )
                    if turn.text and re.search(r'[.!?]["\')\]]?\s*$', turn.text):
                        await self._flush(context_id, len(turn.text))
                    elif len(turn.text) >= turn.settings.text_flush_chars:
                        boundary = turn.text.rfind(" ")
                        if boundary >= 20:
                            await self._flush(context_id, boundary + 1)
        except Exception as exc:
            await self._fail(exc)
        # The SDK must leave completion to the background audio receiver.
        yield None

    async def flush_audio(self, context_id=None):
        if not context_id or context_id not in self._turns:
            return
        turn = self._turns[context_id]
        try:
            if turn.text and not turn.cancelled and not self._failed:
                await self._flush(context_id, len(turn.text))
            turn.ended = True
            await self._finish(context_id)
        except Exception as exc:
            await self._fail(exc)

    async def _flush(self, context_id, count):
        turn = self._turns[context_id]
        text, turn.text = turn.text[:count], turn.text[count:]
        if not text.strip() or turn.cancelled or self._failed:
            return
        async with asyncio.timeout(self._timeout):
            await self._slots.acquire()
        registered = False
        try:
            await self._connect()
            async with self._write_lock:
                if turn.cancelled or self._failed:
                    return
                flush_id = uuid.uuid4().hex
                self._pending[flush_id] = _Flush(context_id)
                turn.pending.add(flush_id)
                registered = True
                # At most four <=1000-character phrases are pending, below the
                # server's shared 10000-character append+pending limit.
                # Interruption must not leave appended text without its flush.
                # Finish this bounded transaction before the SDK invokes cancel.
                transaction = asyncio.create_task(self._submit(turn, text, flush_id))
                try:
                    await asyncio.shield(transaction)
                except asyncio.CancelledError:
                    try:
                        await transaction
                    except Exception as exc:
                        await self._fail(exc)
                    raise
                self._flush_count += 1
        finally:
            if not registered:
                self._slots.release()

    async def _submit(self, turn, text, flush_id):
        async with asyncio.timeout(2):
            await self._send(
                {
                    "type": "session.update",
                    "session": {
                        "model": turn.settings.model,
                        "voice": turn.settings.voice,
                        "language": None,
                        "speed": 1.0,
                        "instructions": None,
                        "pronunciation_replacements": None,
                        "first_chunk_chars": None,
                        "max_chunk_chars": None,
                        **speech_options(turn.settings),
                    },
                }
            )
            await self._send({"type": "input_text.append", "text": text})
            await self._send({"type": "input_text.flush", "flush_id": flush_id})

    async def _connect(self):
        async with self._connect_lock:
            if self._failed or self._closing:
                raise RuntimeError("Voicebox speech service is closed")
            # Recycle only while no work is outstanding, before the session's
            # 1000-flush lifetime limit. Never replay a submitted phrase.
            if self._flush_count >= 950:
                async with asyncio.timeout(self._timeout):
                    while self._pending:
                        await asyncio.sleep(0.01)
                await self._disconnect()
            if self._ws and not self._ws.closed:
                return
            if self._pending:
                raise RuntimeError("Voicebox speech connection lost with pending audio")
            if self._ws:
                # A closed transport's old receiver must finish before a new
                # connection can own the shared pending registry.
                await self._disconnect()
            if self._session is None:
                self._session = aiohttp.ClientSession()
            ws = None
            try:
                async with asyncio.timeout(10):
                    ws = await self._session.ws_connect(
                        self._url,
                        headers={"Authorization": f"Bearer {self._key}"},
                        max_msg_size=65536,
                    )
                    event = await ws.receive_json()
                    if (
                        event.get("type") != "session.created"
                        or event.get("audio_format") != PCM_FORMAT
                    ):
                        raise ValueError("Voicebox speech requires PCM16 mono at 24000 Hz")
                    limits = event.get("limits", {})
                    if (
                        limits.get("max_pending_flushes", 0) < MAX_PENDING
                        or limits.get("max_text_chars", 0) < 4000
                    ):
                        raise ValueError(
                            "Voicebox speech session limits are below adapter requirements"
                        )
            except BaseException:
                if ws:
                    await ws.close()
                raise
            self._ws = ws
            self._flush_count = 0
            self._reader = asyncio.create_task(self._receive(), name="voicebox-pipecat-speech")

    async def _send(self, payload):
        async with asyncio.timeout(10):
            await self._ws.send_json(payload)

    async def _receive(self):
        try:
            while not self._closing:
                try:
                    async with asyncio.timeout(self._timeout):
                        message = await self._ws.receive()
                except TimeoutError:
                    if self._pending:
                        raise TimeoutError("Voicebox speech progress timed out") from None
                    # A quiet conversation is not a provider failure. Reconnect
                    # lazily when the next phrase is ready, without replay.
                    await self._close_idle()
                    return
                if message.type != aiohttp.WSMsgType.TEXT:
                    if self._pending:
                        raise RuntimeError("Voicebox speech closed before audio.done")
                    return
                event = json.loads(message.data)
                kind = event.get("type")
                if kind == "error":
                    error = event.get("error", {})
                    raise RuntimeError(
                        f"Voicebox {error.get('code', 'error')}: {error.get('message', '')}"
                    )
                if kind == "session.closed":
                    if self._pending:
                        raise RuntimeError("Voicebox speech session ended with pending audio")
                    await self._close_idle()
                    return
                if kind in {
                    "session.updated",
                    "input_text.accepted",
                    "input_text.flushed",
                    "input_text.cleared",
                }:
                    continue
                await self._audio_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(exc)

    async def _close_idle(self):
        # Serialize idle retirement with connection reuse. A phrase submitted
        # since the idle decision must fail explicitly, never disappear.
        async with self._connect_lock:
            if self._pending:
                raise RuntimeError("Voicebox speech closed while a phrase was being submitted")
            async with asyncio.timeout(2):
                await self._ws.close()

    async def _audio_event(self, event):
        kind, flush_id = event.get("type"), event.get("flush_id")
        if flush_id in self._retired:
            return
        flush = self._pending.get(flush_id)
        if flush is None:
            raise ValueError("Voicebox sent an unknown speech flush ID")
        turn = self._turns[flush.context_id]
        if kind == "audio.cancelled":
            if not turn.cancelled:
                raise RuntimeError("Voicebox unexpectedly cancelled speech")
            await self._terminal(flush_id)
            return
        if flush_id != next(iter(self._pending)):
            raise ValueError("Voicebox returned speech flushes out of order")
        if kind == "audio.started":
            if flush.started or any(event.get(key) != value for key, value in PCM_FORMAT.items()):
                raise ValueError("Voicebox speech has invalid PCM metadata or duplicate start")
            flush.started = True
        elif kind == "audio.delta":
            if (
                not flush.started
                or type(event.get("sequence")) is not int
                or event["sequence"] != flush.sequence
            ):
                raise ValueError("Voicebox speech has an invalid audio sequence")
            pcm = base64.b64decode(event.get("audio", ""), validate=True)
            if not pcm or len(pcm) % 2 or len(pcm) > 4800:
                raise ValueError("Voicebox speech has invalid PCM16 audio")
            flush.sequence += 1
            flush.received_audio = True
            flush.samples += len(pcm) // 2
            if not turn.cancelled:
                await self.stop_ttfb_metrics()
                await self._append_audio(flush.context_id, turn, pcm)
        elif kind == "audio.done":
            if not flush.started or not flush.received_audio:
                raise ValueError("Voicebox returned an empty speech response")
            if (
                type(event.get("chunks")) is not int
                or event["chunks"] != flush.sequence
                or type(event.get("samples")) is not int
                or event["samples"] != flush.samples
            ):
                raise ValueError("Voicebox speech completion counters do not match received audio")
            await self._terminal(flush_id)
        else:
            raise ValueError(f"Unexpected Voicebox speech event: {kind}")

    async def _append_audio(self, context_id, turn, pcm):
        # Pipecat 1.15 context queues are unbounded. Bound audio ahead of playback
        # to 64 x 100ms frames, without replacing SDK queues/keepalive behavior.
        async with asyncio.timeout(5):
            while not turn.cancelled and self.audio_context_available(context_id):
                buffered = sum(
                    queue.qsize() for queue in self._audio_context_manager._contexts.values()
                )
                if buffered < MAX_BUFFERED_FRAMES:
                    await self.append_to_audio_context(
                        context_id, TTSAudioRawFrame(pcm, SAMPLE_RATE, 1, context_id=context_id)
                    )
                    return
                await asyncio.sleep(0.01)

    async def _terminal(self, flush_id):
        flush = self._pending.pop(flush_id)
        self._retired.append(flush_id)
        self._slots.release()
        self._turns[flush.context_id].pending.discard(flush_id)
        await self._finish(flush.context_id)

    async def _finish(self, context_id):
        turn = self._turns.get(context_id)
        if turn and not turn.pending and (turn.ended or turn.cancelled):
            if not turn.cancelled and self.audio_context_available(context_id):
                await self.append_to_audio_context(
                    context_id, TTSStoppedFrame(context_id=context_id)
                )
                await self.remove_audio_context(context_id)
            self._turns.pop(context_id, None)

    async def on_audio_context_interrupted(self, context_id):
        turn = self._turns.get(context_id)
        if not turn:
            return
        turn.cancelled = True
        turn.text = ""
        try:
            async with self._write_lock:
                async with asyncio.timeout(2):
                    for flush_id in list(turn.pending):
                        if flush_id in self._pending:
                            await self._send({"type": "input_text.cancel", "flush_id": flush_id})
            await self._finish(context_id)
        except Exception as exc:
            await self._fail(exc)

    async def _fail(self, exc):
        if self._failed or self._closing:
            return
        self._failed = True
        # Publish the cause before ending contexts: Pipecat may synthesize a stop
        # or zero-audio error when it reads their terminal sentinel.
        await self.push_error_frame(
            ErrorFrame(error=f"Voicebox incremental speech failed: {exc}", exception=exc),
            force_treat_as_permanent=True,
        )
        for context_id, turn in list(self._turns.items()):
            turn.cancelled = True
            if self.audio_context_available(context_id):
                await self.remove_audio_context(context_id)
        await self._disconnect()

    async def _disconnect(self):
        reader, self._reader = self._reader, None
        if reader and reader is not asyncio.current_task():
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        if self._ws:
            ws, self._ws = self._ws, None
            try:
                async with asyncio.timeout(2):
                    await ws.close()
            except TimeoutError:
                pass
        for _ in self._pending:
            self._slots.release()
        self._pending.clear()

    async def stop(self, frame):
        try:
            await super().stop(frame)
        finally:
            await self._close()

    async def cancel(self, frame):
        await self._close()
        await super().cancel(frame)

    async def _close(self):
        self._closing = True
        for turn in self._turns.values():
            turn.cancelled = True
        await self._disconnect()
        self._turns.clear()
        if self._session:
            await self._session.close()

    async def cleanup(self):
        try:
            await self._close()
        finally:
            await super().cleanup()
