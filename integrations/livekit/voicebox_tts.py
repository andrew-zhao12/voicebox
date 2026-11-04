"""Native LiveKit TTS over Voicebox's incremental-text speech WebSocket.

One SDK stream owns one socket. Text is flushed in bounded phrases while the LLM
is still producing it; all phrases belong to the SDK's original transcript segment.
Requests are never replayed automatically after a transport or inference failure.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import re
import uuid
from dataclasses import replace
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, APIError, tts

SAMPLE_RATE = 24000
MAX_INPUT_CHARS = 10_000
MAX_AUDIO_BYTES = SAMPLE_RATE * 2 * 600
MAX_EVENT_BYTES = 128 * 1024
APPEND_CHARS = 256
END_OF_PHRASE = re.compile(r"[.!?\u3002\uff01\uff1f](?:[\"'\u201d\u2019)]*)(?=\s|$)")


def websocket_url(base_url: str) -> str:
    """Accept a server root or /v1 base URL, without URL credentials or query secrets."""
    url = urlsplit(base_url)
    if (
        url.scheme not in {"http", "https", "ws", "wss"}
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Voicebox URL must be HTTP(S) or WS(S), without credentials/query/fragment"
        )
    path = url.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit(
        (
            "wss" if url.scheme in {"https", "wss"} else "ws",
            url.netloc,
            path + "/v1/audio/speech/stream",
            "",
            "",
        )
    )


def pronunciation_dictionary(value: object) -> dict[str, str]:
    """Validate the same bounded literal dictionary accepted by Voicebox."""
    if not isinstance(value, dict) or len(value) > 100:
        raise ValueError("Pronunciations must be an object with at most 100 entries")
    for source, spoken in value.items():
        if (
            not isinstance(source, str)
            or not source.strip()
            or len(source) > 128
            or not isinstance(spoken, str)
            or not spoken.strip()
            or len(spoken) > 256
        ):
            raise ValueError("Pronunciation keys need 1-128 characters and values need 1-256")
    if sum(len(key) + len(item) for key, item in value.items()) > 8192:
        raise ValueError("Pronunciation dictionary exceeds 8192 characters")
    return dict(value)


def _cut(text: str, limit: int, *, final: bool = False) -> tuple[str, str]:
    """Take a sentence or bounded phrase, retaining the exact text across boundaries."""
    for boundary in END_OF_PHRASE.finditer(text, 20):
        if boundary.end() <= limit:
            return text[: boundary.end()], text[boundary.end() :]
        break
    if len(text) >= limit:
        boundary = text.rfind(" ", 0, limit + 1)
        boundary = boundary + 1 if boundary > 0 else limit
        return text[:boundary], text[boundary:]
    return (text, "") if final else ("", text)


class VoiceboxTTS(tts.TTS):
    """Voicebox's append/flush/cancel protocol exposed as a native LiveKit TTS."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "tts-1",
        voice: str = "af_heart",
        language: str | None = None,
        speed: float = 1.0,
        instructions: str | None = None,
        first_chunk_chars: int | None = None,
        max_chunk_chars: int | None = None,
        pronunciation_replacements: dict[str, str] | None = None,
        text_flush_chars: int = 160,
        speech_timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("Voicebox client key is required")
        if not math.isfinite(speed) or not 0.25 <= speed <= 4:
            raise ValueError("Speech speed must be between 0.25 and 4")
        for name, value, low, high in (
            ("first_chunk_chars", first_chunk_chars, 20, 1000),
            ("max_chunk_chars", max_chunk_chars, 100, 5000),
            ("text_flush_chars", text_flush_chars, 20, 2000),
        ):
            if value is not None and (type(value) is not int or not low <= value <= high):
                raise ValueError(f"{name} must be an integer between {low} and {high}")
        if instructions is not None and len(instructions) > 500:
            raise ValueError("Speech instructions are limited to 500 characters")
        if not math.isfinite(speech_timeout) or speech_timeout <= 0:
            raise ValueError("Speech progress timeout must be a positive finite number")
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self.url = websocket_url(base_url)
        self.api_key = api_key
        self.text_flush_chars = text_flush_chars
        self.speech_timeout = speech_timeout
        self.settings = {"model": model, "voice": voice, "speed": speed}
        for name, value in (
            ("language", language),
            ("instructions", instructions),
            ("first_chunk_chars", first_chunk_chars),
            ("max_chunk_chars", max_chunk_chars),
        ):
            if value is not None:
                self.settings[name] = value
        if pronunciation_replacements is not None:
            self.settings["pronunciation_replacements"] = pronunciation_dictionary(
                pronunciation_replacements
            )
        self._streams: set[_SynthesizeStream] = set()
        self._closed = False

    @property
    def model(self) -> str:
        return self.settings["model"]

    @property
    def provider(self) -> str:
        return "voicebox"

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        if self._closed:
            raise RuntimeError("Voicebox TTS is closed")
        stream = _SynthesizeStream(tts=self, conn_options=replace(conn_options, max_retry=0))
        self._streams.add(stream)
        stream._task.add_done_callback(lambda _task: self._streams.discard(stream))
        return stream

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=replace(conn_options, max_retry=0))

    async def aclose(self) -> None:
        self._closed = True
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)))


class _SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: VoiceboxTTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.owner = tts
        self._characters = 0
        self._audio_bytes = 0

    def push_text(self, token: str) -> None:
        if len(token) + self._characters > MAX_INPUT_CHARS:
            raise APIError("Voicebox stream input exceeds 10000 characters", retryable=False)
        self._characters += len(token)
        super().push_text(token)

    async def _receive(self, socket, *, timeout: float | None = None) -> dict:
        message = await asyncio.wait_for(
            socket.receive(), self._conn_options.timeout if timeout is None else timeout
        )
        if message.type != aiohttp.WSMsgType.TEXT:
            raise APIError("Voicebox speech socket closed before completion", retryable=False)
        try:
            event = json.loads(message.data)
        except (ValueError, TypeError) as exc:
            raise APIError("Invalid Voicebox speech event", retryable=False) from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise APIError("Invalid Voicebox speech event", retryable=False)
        if event["type"] == "error":
            error = event.get("error")
            error = error if isinstance(error, dict) else {}
            detail = error.get("message", "Unknown error")
            raise APIError(
                f"Voicebox speech failed ({error.get('code', 'unknown')}): {detail}",
                retryable=False,
            )
        return event

    async def _expect(self, socket, expected: str) -> dict:
        event = await self._receive(socket)
        if event["type"] != expected:
            raise APIError(
                f"Expected {expected} from Voicebox, got {event['type']}", retryable=False
            )
        return event

    async def _speak(self, socket, phrase: str, output: tts.AudioEmitter) -> None:
        flush_id = uuid.uuid4().hex
        self._mark_started()
        for offset in range(0, len(phrase), APPEND_CHARS):
            await asyncio.wait_for(
                socket.send_json(
                    {"type": "input_text.append", "text": phrase[offset : offset + APPEND_CHARS]}
                ),
                self._conn_options.timeout,
            )
            await self._expect(socket, "input_text.accepted")
        await asyncio.wait_for(
            socket.send_json({"type": "input_text.flush", "flush_id": flush_id}),
            self._conn_options.timeout,
        )
        event = await self._expect(socket, "input_text.flushed")
        if event.get("flush_id") != flush_id:
            raise APIError("Voicebox acknowledged another speech flush", retryable=False)
        started = False
        chunks = samples = 0
        while True:
            event = await self._receive(socket, timeout=self.owner.speech_timeout)
            if event.get("flush_id") != flush_id:
                raise APIError("Voicebox speech flush ID mismatch", retryable=False)
            kind = event["type"]
            if kind == "audio.started" and not started:
                if (event.get("format"), event.get("sample_rate"), event.get("channels")) != (
                    "pcm_s16le",
                    SAMPLE_RATE,
                    1,
                ):
                    raise APIError("Unsupported Voicebox speech audio format", retryable=False)
                started = True
            elif kind == "audio.delta" and started:
                if type(event.get("sequence")) is not int or event["sequence"] != chunks:
                    raise APIError("Voicebox speech sequence mismatch", retryable=False)
                try:
                    audio = base64.b64decode(event["audio"], validate=True)
                except (KeyError, ValueError, TypeError) as exc:
                    raise APIError("Invalid Voicebox PCM payload", retryable=False) from exc
                if not audio or len(audio) % 2:
                    raise APIError("Voicebox PCM payload has incomplete samples", retryable=False)
                self._audio_bytes += len(audio)
                if self._audio_bytes > MAX_AUDIO_BYTES:
                    raise APIError("Voicebox stream audio exceeds ten minutes", retryable=False)
                chunks += 1
                samples += len(audio) // 2
                output.push(audio)
            elif kind == "audio.done" and started:
                if (
                    chunks == 0
                    or type(event.get("chunks")) is not int
                    or type(event.get("samples")) is not int
                    or event["chunks"] != chunks
                    or event["samples"] != samples
                ):
                    raise APIError(
                        "Voicebox speech completion does not match received audio", retryable=False
                    )
                return
            else:
                raise APIError(f"Unexpected Voicebox speech event: {kind}", retryable=False)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        socket = None
        completed = False
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self._conn_options.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as client:
            try:
                socket = await asyncio.wait_for(
                    client.ws_connect(
                        self.owner.url,
                        headers={"Authorization": f"Bearer {self.owner.api_key}"},
                        max_msg_size=MAX_EVENT_BYTES,
                        heartbeat=20,
                    ),
                    self._conn_options.timeout,
                )
                created = await self._expect(socket, "session.created")
                if created.get("audio_format") != {
                    "format": "pcm_s16le",
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                }:
                    raise APIError("Unsupported Voicebox speech session format", retryable=False)
                await asyncio.wait_for(
                    socket.send_json({"type": "session.update", "session": self.owner.settings}),
                    self._conn_options.timeout,
                )
                await self._expect(socket, "session.updated")
                output_emitter.initialize(
                    request_id=created.get("session_id") or uuid.uuid4().hex,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    mime_type="audio/pcm",
                    stream=True,
                )
                buffer = ""
                segment_started = False
                async for token in self._input_ch:
                    final = isinstance(token, self._FlushSentinel)
                    if not final:
                        buffer += token
                    while True:
                        phrase, rest = _cut(buffer, self.owner.text_flush_chars, final=final)
                        if not phrase:
                            break
                        buffer = rest
                        if phrase.strip():
                            if not segment_started:
                                output_emitter.start_segment(segment_id=uuid.uuid4().hex)
                                segment_started = True
                            await self._speak(socket, phrase, output_emitter)
                    if final and segment_started:
                        output_emitter.end_segment()
                        segment_started = False
                completed = True
            except APIError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                raise APIError(
                    "Voicebox speech transport failed or timed out; request was not replayed",
                    retryable=False,
                ) from exc
            finally:
                if socket is not None and not socket.closed:
                    with contextlib.suppress(Exception):
                        if not completed:
                            await asyncio.wait_for(
                                socket.send_json({"type": "input_text.cancel"}), 1
                            )
                        await asyncio.wait_for(socket.send_json({"type": "session.close"}), 1)
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(socket.close(), 2)
