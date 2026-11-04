"""Exercise the real LiveKit SDK lifecycle against a local Voicebox protocol peer."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json

import pytest
import pytest_asyncio
from aiohttp import WSMsgType, web
from livekit.agents import APIConnectOptions, APIError

from voicebox_tts import MAX_INPUT_CHARS, VoiceboxTTS, pronunciation_dictionary, websocket_url

PCM = b"\x01\x00" * 4800
FORMAT = {"format": "pcm_s16le", "sample_rate": 24000, "channels": 1}


class Peer:
    def __init__(self):
        self.mode = "normal"
        self.connections = 0
        self.settings = []
        self.requests = []
        self.phrases = []
        self.headers = []
        self.cancelled = asyncio.Event()
        self.closed = asyncio.Event()
        self.url = ""

    async def handle(self, request):
        self.connections += 1
        self.headers.append(request.headers.get("Authorization"))
        if self.mode == "unauthorized":
            return web.Response(status=401)
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        audio_format = {**FORMAT}
        if self.mode == "session_format":
            audio_format["channels"] = 2
        await socket.send_json(
            {"type": "session.created", "session_id": "test-session", "audio_format": audio_format}
        )
        buffer = ""
        try:
            async for message in socket:
                if message.type != WSMsgType.TEXT:
                    break
                event = json.loads(message.data)
                self.requests.append(event)
                kind = event["type"]
                if kind == "session.update":
                    self.settings.append(event["session"])
                    await socket.send_json({"type": "session.updated"})
                elif kind == "input_text.append":
                    buffer += event["text"]
                    await socket.send_json(
                        {"type": "input_text.accepted", "buffered_chars": len(buffer)}
                    )
                elif kind == "input_text.flush":
                    self.phrases.append(buffer)
                    buffer = ""
                    flush_id = event["flush_id"]
                    await socket.send_json({"type": "input_text.flushed", "flush_id": flush_id})
                    if self.mode == "overload":
                        await socket.send_json(
                            {
                                "type": "error",
                                "flush_id": flush_id,
                                "error": {
                                    "code": "queue_full",
                                    "message": "Queue full",
                                    "retry_after_s": 1,
                                },
                            }
                        )
                        continue
                    if self.mode == "timeout":
                        continue
                    if self.mode == "early_close":
                        await socket.close()
                        break
                    started = {"type": "audio.started", "flush_id": flush_id, **FORMAT}
                    if self.mode == "format":
                        started["sample_rate"] = 44100
                    await socket.send_json(started)
                    delta = {
                        "type": "audio.delta",
                        "flush_id": flush_id,
                        "sequence": 0,
                        "audio": base64.b64encode(PCM).decode(),
                    }
                    if self.mode == "sequence":
                        delta["sequence"] = 1
                    elif self.mode == "base64":
                        delta["audio"] = "not base64!"
                    elif self.mode == "odd_pcm":
                        delta["audio"] = "AA=="
                    elif self.mode == "flush_id":
                        delta["flush_id"] = "another-request"
                    await socket.send_json(delta)
                    if self.mode == "hang_after_audio":
                        continue
                    if self.mode == "partial_close":
                        await asyncio.sleep(0.02)
                        await socket.close()
                        break
                    if self.mode == "partial_error":
                        await socket.send_json(
                            {
                                "type": "error",
                                "flush_id": flush_id,
                                "error": {"code": "inference_failed", "message": "test error"},
                            }
                        )
                        continue
                    await socket.send_json(
                        {
                            "type": "audio.done",
                            "flush_id": flush_id,
                            "chunks": 1,
                            "samples": 1 if self.mode == "counts" else len(PCM) // 2,
                        }
                    )
                elif kind == "input_text.cancel":
                    self.cancelled.set()
                elif kind == "session.close":
                    break
        finally:
            self.closed.set()
            with contextlib.suppress(ConnectionError):
                await socket.close()
        return socket


@pytest_asyncio.fixture
async def peer():
    peer = Peer()
    app = web.Application()
    app.router.add_get("/v1/audio/speech/stream", peer.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    peer.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        yield peer
    finally:
        await runner.cleanup()


def service(peer, **kwargs):
    return VoiceboxTTS(base_url=peer.url, api_key="vbx_test", **kwargs)


@pytest.mark.asyncio
async def test_incremental_audio_precedes_end_input_and_preserves_one_sdk_segment(peer):
    tts = service(
        peer,
        speed=1.2,
        first_chunk_chars=40,
        max_chunk_chars=300,
        pronunciation_replacements={"SDK": "ess dee kay"},
    )
    stream = tts.stream()
    text = "This first sentence is ready now. This second sentence comes later."
    try:
        stream.push_text("This first sentence is ready now.")
        first = await asyncio.wait_for(anext(stream), 2)
        assert first.frame.sample_rate == 24000
        assert first.frame.num_channels == 1
        assert not first.is_final
        stream.push_text(" This second sentence comes later.")
        stream.end_input()
        frames = [first, *[frame async for frame in stream]]
        assert sum(frame.is_final for frame in frames) == 1
        assert len({frame.segment_id for frame in frames}) == 1
        assert sum(frame.frame.samples_per_channel for frame in frames) == 9600
        assert "".join(peer.phrases) == text
        assert peer.headers == ["Bearer vbx_test"]
        assert peer.settings[0]["pronunciation_replacements"] == {"SDK": "ess dee kay"}
        assert peer.settings[0]["speed"] == 1.2
        assert not peer.cancelled.is_set()
    finally:
        await stream.aclose()
        await tts.aclose()
    await asyncio.wait_for(peer.closed.wait(), 1)


@pytest.mark.asyncio
async def test_synthesize_and_unpunctuated_text_are_bounded(peer):
    tts = service(peer, text_flush_chars=400)
    text = "a" * 1100
    stream = tts.synthesize(text)
    try:
        frames = [frame async for frame in stream]
        # SDK 1.8.10 may append a 10 ms silent final marker when its flush has
        # already emitted every sample; the source PCM itself must stay exact.
        rendered = b"".join(frame.frame.data.tobytes() for frame in frames)
        assert rendered.rstrip(b"\x00") == (PCM * 3).rstrip(b"\x00")
        assert "".join(peer.phrases) == text
        assert max(map(len, peer.phrases)) <= 400
        appends = [item["text"] for item in peer.requests if item["type"] == "input_text.append"]
        assert max(map(len, appends)) <= 256
        assert peer.connections == 1
    finally:
        await stream.aclose()
        await tts.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "format",
        "sequence",
        "base64",
        "odd_pcm",
        "flush_id",
        "counts",
        "partial_error",
        "partial_close",
        "early_close",
        "timeout",
        "session_format",
        "unauthorized",
        "overload",
    ],
)
async def test_failures_never_retry_or_replay_even_after_audio(peer, mode):
    peer.mode = mode
    tts = service(peer, speech_timeout=0.1)
    stream = tts.synthesize(
        "A sentence for the failure test.", conn_options=APIConnectOptions(max_retry=3)
    )
    try:
        with pytest.raises(APIError) as failure:
            async for _ in stream:
                pass
        assert not failure.value.retryable
        assert peer.connections == 1
        assert len(peer.phrases) == (0 if mode in {"session_format", "unauthorized"} else 1)
    finally:
        await stream.aclose()
        await tts.aclose()


@pytest.mark.asyncio
async def test_interruption_cancels_active_audio_and_closes_socket(peer):
    peer.mode = "hang_after_audio"
    tts = service(peer)
    stream = tts.stream()
    stream.push_text("This sentence should be interrupted.")
    await asyncio.wait_for(anext(stream), 2)
    await asyncio.wait_for(stream.aclose(), 2)
    await asyncio.wait_for(peer.cancelled.wait(), 1)
    await asyncio.wait_for(peer.closed.wait(), 1)
    assert peer.connections == 1
    await tts.aclose()


@pytest.mark.asyncio
async def test_explicit_sdk_flush_sends_a_short_unpunctuated_tail(peer):
    tts = service(peer)
    stream = tts.stream()
    try:
        stream.push_text("A short tail")
        stream.flush()
        frames = []
        while not frames or not frames[-1].is_final:
            frames.append(await asyncio.wait_for(anext(stream), 2))
        assert peer.phrases == ["A short tail"]
        stream.end_input()
        assert [frame async for frame in stream] == []
    finally:
        await stream.aclose()
        await tts.aclose()


@pytest.mark.asyncio
async def test_whitespace_only_input_does_not_create_a_job(peer):
    tts = service(peer)
    stream = tts.stream()
    try:
        stream.push_text("   ")
        stream.end_input()
        assert [frame async for frame in stream] == []
        assert peer.phrases == []
    finally:
        await stream.aclose()
        await tts.aclose()


@pytest.mark.asyncio
async def test_provider_close_cancels_concurrent_streams(peer):
    peer.mode = "hang_after_audio"
    tts = service(peer)
    streams = [tts.stream(), tts.stream()]
    for stream in streams:
        stream.push_text("This sentence will be cancelled.")
    await asyncio.wait_for(asyncio.gather(*(anext(stream) for stream in streams)), 2)
    await asyncio.wait_for(tts.aclose(), 2)
    assert peer.connections == 2
    assert sum(item["type"] == "input_text.cancel" for item in peer.requests) == 2
    with pytest.raises(RuntimeError, match="closed"):
        tts.stream()


@pytest.mark.asyncio
async def test_incremental_input_has_a_total_bound(peer):
    tts = service(peer)
    stream = tts.stream()
    try:
        stream.push_text("a" * MAX_INPUT_CHARS)
        with pytest.raises(APIError, match="10000"):
            stream.push_text("b")
    finally:
        await stream.aclose()
        await tts.aclose()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.test/v1/", "wss://example.test/v1/audio/speech/stream"),
        ("http://localhost:17493", "ws://localhost:17493/v1/audio/speech/stream"),
    ],
)
def test_url_conversion(url, expected):
    assert websocket_url(url) == expected


@pytest.mark.parametrize("url", ["ftp://localhost", "http://key@localhost", "http://x/?key=y"])
def test_url_rejects_credentials_and_non_web_transports(url):
    with pytest.raises(ValueError, match="Voicebox URL"):
        websocket_url(url)


@pytest.mark.parametrize("value", [[], {"": "word"}, {"word": ""}, {"x": "a" * 257}])
def test_invalid_pronunciation_dictionary(value):
    with pytest.raises(ValueError, match="Pronunciation"):
        pronunciation_dictionary(value)
