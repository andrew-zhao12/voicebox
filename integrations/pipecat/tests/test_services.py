"""Exercise real SDK HTTP/WebSocket contracts and pipeline cleanup without model servers."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from openai import AuthenticationError
from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.errors import ErrorCategory
from pipecat.workers.runner import WorkerRunner

from bot import ChatService, Settings
from services import SAMPLE_RATE, VoiceboxTTSService, api_url, voicebox_stt
from smoke import transcribe_realtime, wav_bytes, word_error_rate


@asynccontextmanager
async def server(path, handler, *, websocket=False):
    app = web.Application()
    app.router.add_route("GET" if websocket else "POST", path, handler)
    async with TestServer(app) as running:
        yield str(running.make_url("/"))


def test_settings_require_dedicated_voicebox_key(monkeypatch):
    monkeypatch.delenv("VOICEBOX_CLIENT_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    with pytest.raises(ValueError, match="VOICEBOX_CLIENT_KEY"):
        Settings.from_env()


def test_custom_llm_does_not_receive_openai_key(monkeypatch):
    monkeypatch.setenv("VOICEBOX_CLIENT_KEY", "voicebox-only")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    settings = Settings.from_env()
    assert settings.llm_key == "local"
    assert "voicebox-only" not in repr(settings)
    assert "openai-only" not in repr(settings)


@pytest.mark.asyncio
async def test_chat_cleanup_closes_sdk_client():
    chat = ChatService(api_key="local", base_url="http://127.0.0.1:11434/v1")
    await chat.cleanup()
    assert chat._client.is_closed()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:17493", "http://localhost:17493/v1"),
        ("https://example.test/speech/v1/", "https://example.test/speech/v1"),
    ],
)
def test_api_url(url, expected):
    assert api_url(url) == expected


@pytest.mark.parametrize(
    "url", ["ftp://example.test", "https://key@example.test", "https://example.test?q=key"]
)
def test_bad_url(url):
    with pytest.raises(ValueError):
        api_url(url)


@pytest.mark.asyncio
async def test_tts_preserves_custom_voice_pcm_and_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-voicebox")
    pcm = b"\x01\x02" * 5001

    async def speech(request):
        assert request.headers["Authorization"] == "Bearer voicebox"
        assert await request.json() == {
            "model": "kokoro",
            "voice": "custom-profile-id",
            "input": "Hello.",
            "response_format": "pcm",
        }
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        for begin, end in [(0, 7), (7, 101), (101, len(pcm))]:
            await response.write(pcm[begin:end])
        await response.write_eof()
        return response

    async with server("/v1/audio/speech", speech) as url:
        tts = VoiceboxTTSService(
            api_key="voicebox", base_url=url, voice="custom-profile-id", model="kokoro"
        )
        try:
            frames = [frame async for frame in tts.run_tts("Hello.", "context")]
            assert b"".join(f.audio for f in frames) == pcm
            assert all(f.sample_rate == SAMPLE_RATE and f.num_channels == 1 for f in frames)
            assert all(f.context_id == "context" for f in frames)
        finally:
            await tts.cleanup()
        assert tts._client.is_closed()


@pytest.mark.asyncio
async def test_tts_auth_failure_is_not_silent_success():
    async def speech(request):
        return web.json_response(
            {"error": {"message": "Unauthorized", "type": "authentication_error"}}, status=401
        )

    async with server("/v1/audio/speech", speech) as url:
        tts = VoiceboxTTSService(api_key="invalid", base_url=url)
        try:
            frames = [frame async for frame in tts.run_tts("Hello", "failure")]
            assert len(frames) == 1
            assert isinstance(frames[0], ErrorFrame)
            assert isinstance(frames[0].exception, AuthenticationError)
        finally:
            await tts.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"", b"\0"])
async def test_invalid_pcm_response_reports_failure(body):
    async def speech(request):
        return web.Response(body=body, content_type="audio/pcm")

    async with server("/v1/audio/speech", speech) as url:
        tts = VoiceboxTTSService(api_key="voicebox", base_url=url)
        try:
            frames = [frame async for frame in tts.run_tts("Hello", "bad-pcm")]
            assert len(frames) == 1 and isinstance(frames[0], ErrorFrame)
        finally:
            await tts.cleanup()


@pytest.mark.asyncio
async def test_failed_tts_closes_audio_context_promptly():
    stopped = asyncio.Event()
    errors = []

    class Collector(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TTSStoppedFrame):
                stopped.set()
            await self.push_frame(frame, direction)

    async def speech(request):
        return web.json_response({"error": {"message": "Unauthorized"}}, status=401)

    async with server("/v1/audio/speech", speech) as url:
        tts = VoiceboxTTSService(api_key="invalid", base_url=url)

        @tts.event_handler("on_error")
        async def on_error(service, error):
            errors.append(error)

        worker = PipelineWorker(Pipeline([tts, Collector()]), enable_rtvi=False)
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        running = asyncio.create_task(runner.run())
        try:
            await worker.queue_frames([TTSSpeakFrame("Authentication will fail.")])
            await asyncio.wait_for(stopped.wait(), 2)
            assert errors[0].category == ErrorCategory.AUTHENTICATION
            assert not tts.is_usable
            assert not tts.get_audio_contexts()
        finally:
            await runner.cancel()
            await running


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"", b"\0\0" * 2400])
async def test_truncated_http_transfer_reports_failure_without_retry(body):
    requests = []

    async def speech(request):
        requests.append(request)
        response = web.StreamResponse(
            headers={"Content-Type": "audio/pcm", "Content-Length": str(len(body) + 100)}
        )
        await response.prepare(request)
        if body:
            await response.write(body)
        request.transport.close()
        return response

    async with server("/v1/audio/speech", speech) as url:
        tts = VoiceboxTTSService(api_key="voicebox", base_url=url)
        try:
            frames = [frame async for frame in tts.run_tts("Hello", "truncated")]
            assert isinstance(frames[-1], ErrorFrame)
            assert len(requests) == 1
        finally:
            await tts.cleanup()


@pytest.mark.asyncio
async def test_tts_cancellation_closes_http_stream():
    request_seen = asyncio.get_running_loop().create_future()
    release = asyncio.Event()

    async def speech(request):
        request_seen.set_result(request.transport)
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(b"\0\0" * 2400)
        await release.wait()
        return response

    async with server("/v1/audio/speech", speech) as url:
        tts = VoiceboxTTSService(api_key="voicebox", base_url=url)
        generator = tts.run_tts("A long sentence.", "cancel")
        try:
            first = await anext(generator)
            assert len(first.audio) == 4800
            pending = asyncio.create_task(anext(generator))
            await asyncio.sleep(0.05)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            transport = await request_seen
            async with asyncio.timeout(2):
                while not transport.is_closing():
                    await asyncio.sleep(0.01)
        finally:
            release.set()
            await generator.aclose()
            await tts.cleanup()


@pytest.mark.asyncio
async def test_batch_stt_upload_and_finality():
    audio = wav_bytes(b"\0\0" * 2400)

    async def transcribe(request):
        assert request.headers["Authorization"] == "Bearer voicebox"
        form = await request.post()
        assert form["model"] == "whisper-1"
        assert form["language"] == "en"
        assert form["file"].file.read() == audio
        return web.json_response({"text": "Hello there."})

    async with server("/v1/audio/transcriptions", transcribe) as url:
        stt = voicebox_stt(api_key="voicebox", base_url=url, realtime=False)
        try:
            frames = [frame async for frame in stt.run_stt(audio)]
            assert len(frames) == 1
            assert frames[0].text == "Hello there."
            assert frames[0].finalized
        finally:
            await stt.cleanup()
        assert stt._client.is_closed()


@pytest.mark.asyncio
async def test_realtime_ga_handshake_commit_and_transcript(monkeypatch):
    audio = b"\1\0" * 4800
    received = bytearray()
    finalized = []
    closed = asyncio.Event()

    async def realtime(request):
        assert request.headers["Authorization"] == "Bearer voicebox"
        assert request.query["intent"] == "transcription"
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "session": {"type": "transcription"}})
        async for message in ws:
            data = json.loads(message.data)
            if data["type"] == "session.update":
                audio_settings = data["session"]["audio"]["input"]
                assert audio_settings["format"] == {"type": "audio/pcm", "rate": 24000}
                assert audio_settings["turn_detection"] is None
                assert audio_settings["transcription"]["model"] == "whisper-1"
                await ws.send_json({"type": "session.updated"})
            elif data["type"] == "input_audio_buffer.append":
                received.extend(base64.b64decode(data["audio"]))
            elif data["type"] == "input_audio_buffer.commit":
                assert bytes(received).startswith(audio)
                await ws.send_json(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": "Hello there.",
                        "item_id": "one",
                    }
                )
        closed.set()
        return ws

    async with server("/v1/realtime", realtime, websocket=True) as url:
        stt = voicebox_stt(api_key="voicebox", base_url=url)
        original = FrameProcessor.push_frame

        async def observe(self, frame, direction=FrameDirection.DOWNSTREAM):
            if self is stt and hasattr(frame, "finalized"):
                finalized.append(frame.finalized)
            await original(self, frame, direction)

        monkeypatch.setattr(FrameProcessor, "push_frame", observe)
        assert await transcribe_realtime(stt, audio, 10) == "Hello there."
        assert finalized == [True]
        await asyncio.wait_for(closed.wait(), 2)


@pytest.mark.asyncio
async def test_delayed_tts_does_not_end_audio_context_before_audio():
    frames = []
    done = asyncio.Event()

    class Collector(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame)):
                frames.append(frame)
                if isinstance(frame, TTSStoppedFrame):
                    done.set()
            await self.push_frame(frame, direction)

    async def speech(request):
        await asyncio.sleep(3.2)
        return web.Response(body=b"\0\0" * 2400, content_type="audio/pcm")

    async with server("/v1/audio/speech", speech) as url:
        tts = VoiceboxTTSService(api_key="voicebox", base_url=url)
        worker = PipelineWorker(
            Pipeline([tts, Collector()]), params=PipelineParams(), enable_rtvi=False
        )
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        running = asyncio.create_task(runner.run())
        try:
            await worker.queue_frames([TTSSpeakFrame("A cold model takes time.")])
            await asyncio.wait_for(done.wait(), 10)
            assert [type(f) for f in frames] == [TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame]
        finally:
            await runner.cancel()
            await running
        assert tts._client.is_closed()


def test_word_error_rate():
    assert word_error_rate("Hello, there!", "hello there") == 0
    assert word_error_rate("one two three four", "one two four") == 0.25
