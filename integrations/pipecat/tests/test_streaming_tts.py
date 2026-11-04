"""Native incremental speech contract using real Pipecat lifecycle frames."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pipecat.frames.frames import (
    EndFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from bot import Settings
from smoke import synthesize
from streaming_tts import PCM_FORMAT, VoiceboxStreamingTTSService
from tts_settings import VoiceboxTTSSettings, complete_settings, pronunciation_file

PCM = b"\1\0" * 2400


async def audio(ws, flush_id, *, pcm=PCM, sequence=0, metadata=None, done=True):
    await ws.send_json({"type": "audio.started", "flush_id": flush_id, **(metadata or PCM_FORMAT)})
    await ws.send_json(
        {
            "type": "audio.delta",
            "flush_id": flush_id,
            "sequence": sequence,
            "audio": base64.b64encode(pcm).decode(),
        }
    )
    if done:
        await ws.send_json(
            {"type": "audio.done", "flush_id": flush_id, "chunks": 1, "samples": len(pcm) // 2}
        )


@asynccontextmanager
async def speech_server(on_flush=None, *, audio_format=None):
    commands, flushed, settings = [], [], []
    closed = asyncio.Event()

    async def handler(request):
        assert request.headers["Authorization"] == "Bearer voicebox"
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json(
            {
                "type": "session.created",
                "audio_format": audio_format or PCM_FORMAT,
                "limits": {"max_pending_flushes": 4, "max_text_chars": 10000},
            }
        )
        text = ""
        async for message in ws:
            event = json.loads(message.data)
            commands.append(event)
            if event["type"] == "session.update":
                settings.append(event["session"])
                await ws.send_json({"type": "session.updated"})
            elif event["type"] == "input_text.append":
                text += event["text"]
                await ws.send_json({"type": "input_text.accepted"})
            elif event["type"] == "input_text.flush":
                flushed.append(text)
                text = ""
                if on_flush:
                    await on_flush(ws, event["flush_id"])
                else:
                    await audio(ws, event["flush_id"])
            elif event["type"] == "input_text.cancel":
                await ws.send_json({"type": "audio.cancelled", "flush_id": event["flush_id"]})
        closed.set()
        return ws

    app = web.Application()
    app.router.add_get("/prefix/v1/audio/speech/stream", handler)
    async with TestServer(app) as server:
        yield str(server.make_url("/prefix")), commands, flushed, settings, closed


@asynccontextmanager
async def pipeline(tts):
    frames, errors = [], []

    class Collector(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            frames.append(frame)
            await self.push_frame(frame, direction)

    @tts.event_handler("on_error")
    async def on_error(service, error):
        errors.append(error)

    worker = PipelineWorker(Pipeline([tts, Collector()]), enable_rtvi=False)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        yield worker, frames, errors
    finally:
        await runner.cancel()
        await running


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_tokens_flush_before_llm_end_and_complete_once():
    async with speech_server() as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(
            api_key="voicebox",
            base_url=url,
            voice="cloned-profile",
            settings=VoiceboxTTSSettings(
                text_flush_chars=20,
                speed=1.2,
                pronunciation_replacements={"API": "A P I"},
                first_chunk_chars=40,
                max_chunk_chars=300,
            ),
        )
        async with pipeline(tts) as (worker, frames, errors):
            await worker.queue_frames(
                [
                    LLMFullResponseStartFrame(),
                    LLMTextFrame("This API uses "),
                    LLMTextFrame("incremental text "),
                ]
            )
            await until(lambda: any(isinstance(f, TTSAudioRawFrame) for f in frames))
            assert phrases == ["This API uses incremental text "]
            assert not any(isinstance(f, TTSStoppedFrame) for f in frames)
            await worker.queue_frames([LLMTextFrame("and finishes."), LLMFullResponseEndFrame()])
            await until(lambda: any(isinstance(f, TTSStoppedFrame) for f in frames))
            assert "".join(phrases) == "This API uses incremental text and finishes."
            audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
            assert len(audio_frames) == 2
            assert {f.sample_rate for f in audio_frames} == {24000}
            assert {f.num_channels for f in audio_frames} == {1}
            assert len({f.context_id for f in audio_frames}) == 1
            assert sum(isinstance(f, TTSStartedFrame) for f in frames) == 1
            assert sum(isinstance(f, TTSStoppedFrame) for f in frames) == 1
            assert settings[0]["voice"] == "cloned-profile"
            assert settings[0]["pronunciation_replacements"] == {"API": "A P I"}
            assert settings[0]["first_chunk_chars"] == 40
            assert settings[0]["speed"] == 1.2
            assert not errors
        await asyncio.wait_for(closed.wait(), 2)
        assert tts._session.closed
        assert tts._reader is None


@pytest.mark.asyncio
async def test_long_single_frame_respects_pending_text_limit():
    high_water = 0
    async with speech_server() as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url)
        original_send = tts._send

        async def observe(event):
            nonlocal high_water
            high_water = max(high_water, len(tts._pending))
            await original_send(event)

        tts._send = observe
        text = "A long streamed response with many words " * 600
        async with pipeline(tts) as (worker, frames, errors):
            await worker.queue_frames(
                [LLMFullResponseStartFrame(), LLMTextFrame(text), LLMFullResponseEndFrame()]
            )
            await until(lambda: any(isinstance(f, TTSStoppedFrame) for f in frames))
            assert "".join(phrases) == text
            assert max(map(len, phrases)) <= 1000
            assert high_water <= 4
            assert not errors


@pytest.mark.asyncio
async def test_interruption_cancels_target_and_next_turn_uses_fresh_context():
    first_id = None

    async def respond(ws, flush_id):
        nonlocal first_id
        if first_id is None:
            first_id = flush_id
            await audio(ws, flush_id, done=False)
        else:
            await audio(ws, flush_id, pcm=b"\2\0" * 2400)

    async with speech_server(respond) as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url)
        async with pipeline(tts) as (worker, frames, errors):
            await worker.queue_frames(
                [LLMFullResponseStartFrame(), LLMTextFrame("First sentence.")]
            )
            await until(lambda: any(isinstance(f, TTSAudioRawFrame) for f in frames))
            first_context = next(f.context_id for f in frames if isinstance(f, TTSAudioRawFrame))
            await worker.queue_frames([InterruptionFrame()])
            await until(lambda: any(e["type"] == "input_text.cancel" for e in commands))
            await worker.queue_frames(
                [
                    LLMFullResponseStartFrame(),
                    LLMTextFrame("New sentence."),
                    LLMFullResponseEndFrame(),
                ]
            )
            await until(lambda: any(isinstance(f, TTSStoppedFrame) for f in frames))
            cancel = next(e for e in commands if e["type"] == "input_text.cancel")
            assert cancel["flush_id"] == first_id
            last = [f for f in frames if isinstance(f, TTSAudioRawFrame)][-1]
            assert last.context_id != first_context
            assert last.audio == b"\2\0" * 2400
            assert not errors


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["sequence", "odd", "metadata", "empty", "partial", "close", "counters"]
)
async def test_bad_or_partial_audio_fails_without_replay(failure):
    async def respond(ws, flush_id):
        if failure == "empty":
            await ws.send_json({"type": "audio.done", "flush_id": flush_id})
        elif failure == "close":
            await ws.close()
        else:
            await audio(
                ws,
                flush_id,
                sequence=1 if failure == "sequence" else 0,
                pcm=b"x" if failure == "odd" else PCM,
                metadata={**PCM_FORMAT, "sample_rate": 16000} if failure == "metadata" else None,
                done=failure not in {"partial", "counters"},
            )
            if failure == "partial":
                await ws.send_json(
                    {
                        "type": "error",
                        "flush_id": flush_id,
                        "error": {"code": "synthesis_failed", "message": "partial failure"},
                    }
                )
            elif failure == "counters":
                await ws.send_json(
                    {"type": "audio.done", "flush_id": flush_id, "chunks": 2, "samples": 9999}
                )

    async with speech_server(respond) as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url)
        with pytest.raises(RuntimeError, match="Voicebox"):
            await synthesize(tts, "An utterance.", 3)
        assert phrases == ["An utterance."]
        assert tts._session.closed
        assert not tts.is_usable


@pytest.mark.asyncio
async def test_wrong_session_format_sends_no_text():
    async with speech_server(audio_format={**PCM_FORMAT, "channels": 2}) as (
        url,
        commands,
        phrases,
        settings,
        closed,
    ):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url)
        with pytest.raises(RuntimeError, match="24000"):
            await synthesize(tts, "Hello.", 3)
        assert not commands


@pytest.mark.asyncio
async def test_idle_timeout_reconnects_without_replaying_speech():
    async with speech_server() as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url, timeout=1)
        async with pipeline(tts) as (worker, frames, errors):
            await worker.queue_frames([TTSSpeakFrame("First turn.")])
            await until(lambda: sum(isinstance(f, TTSStoppedFrame) for f in frames) == 1)
            await asyncio.wait_for(closed.wait(), 2)
            assert tts.is_usable
            await worker.queue_frames([TTSSpeakFrame("Second turn.")])
            await until(lambda: sum(isinstance(f, TTSStoppedFrame) for f in frames) == 2)
            assert phrases == ["First turn.", "Second turn."]
            assert not errors


@pytest.mark.asyncio
async def test_graceful_end_waits_for_audio_and_closes_resources():
    async with speech_server() as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url)
        async with pipeline(tts) as (worker, frames, errors):
            await worker.queue_frames([TTSSpeakFrame("Finish before closing."), EndFrame()])
            await until(lambda: any(isinstance(f, EndFrame) for f in frames))
            kinds = [type(f) for f in frames]
            assert (
                kinds.index(TTSAudioRawFrame) < kinds.index(TTSStoppedFrame) < kinds.index(EndFrame)
            )
            assert not errors
        assert tts._session.closed


@pytest.mark.asyncio
async def test_out_of_order_flush_audio_fails_without_replay():
    count = 0

    async def respond(ws, flush_id):
        nonlocal count
        count += 1
        if count == 2:
            await audio(ws, flush_id)

    async with speech_server(respond) as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url)
        with pytest.raises(RuntimeError, match="out of order"):
            await synthesize(tts, "First. Second.", 3)
        assert phrases == ["First. ", "Second."]


@pytest.mark.asyncio
async def test_pending_speech_timeout_fails_and_closes_socket():
    async def respond(ws, flush_id):
        pass

    async with speech_server(respond) as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url, timeout=1)
        with pytest.raises(RuntimeError, match="progress timed out"):
            await synthesize(tts, "This speech stalls.", 3)
        assert phrases == ["This speech stalls."]
        assert tts._session.closed


@pytest.mark.asyncio
async def test_settings_are_snapshotted_per_turn_and_can_reset():
    async with speech_server() as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(
            api_key="voicebox",
            base_url=url,
            settings=VoiceboxTTSSettings(speed=1.5, pronunciation_replacements={"SDK": "S D K"}),
        )
        async with pipeline(tts) as (worker, frames, errors):
            await worker.queue_frames(
                [
                    LLMFullResponseStartFrame(),
                    LLMTextFrame("First sentence."),
                    TTSUpdateSettingsFrame(
                        delta=VoiceboxTTSSettings(speed=None, pronunciation_replacements=None)
                    ),
                    LLMTextFrame("Same turn."),
                    LLMFullResponseEndFrame(),
                    TTSSpeakFrame("New turn."),
                ]
            )
            await until(lambda: sum(isinstance(f, TTSStoppedFrame) for f in frames) == 2)
            assert [s["speed"] for s in settings] == [1.5, 1.5, 1.0]
            assert [s["pronunciation_replacements"] for s in settings] == [
                {"SDK": "S D K"},
                {"SDK": "S D K"},
                None,
            ]
            assert not errors


@pytest.mark.asyncio
async def test_interruption_preserves_uninterruptible_context():
    pending = []

    async def respond(ws, flush_id):
        pending.append((ws, flush_id))

    async with speech_server(respond) as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url)
        protected = TTSSpeakFrame("Protected speech.")
        protected.interruptible = False
        async with pipeline(tts) as (worker, frames, errors):
            await worker.queue_frames(
                [
                    protected,
                    LLMFullResponseStartFrame(),
                    LLMTextFrame("Interruptible speech."),
                    LLMTextFrame("Unflushed text"),
                ]
            )
            await until(lambda: len(pending) == 2)
            await worker.queue_frames([InterruptionFrame()])
            await until(lambda: any(e["type"] == "input_text.cancel" for e in commands))
            cancellations = [e["flush_id"] for e in commands if e["type"] == "input_text.cancel"]
            assert cancellations == [pending[1][1]]
            await audio(*pending[0])
            await until(lambda: any(isinstance(f, TTSStoppedFrame) for f in frames))
            assert phrases == ["Protected speech.", "Interruptible speech."]
            assert len([f for f in frames if isinstance(f, TTSAudioRawFrame)]) == 1
            assert not errors


@pytest.mark.asyncio
async def test_interruption_during_append_finishes_transaction_before_cancel():
    appended, release = asyncio.Event(), asyncio.Event()

    async def respond(ws, flush_id):
        pass

    async with speech_server(respond) as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url)
        original_send = tts._send

        async def pause_after_append(event):
            await original_send(event)
            if event["type"] == "input_text.append":
                appended.set()
                await release.wait()

        tts._send = pause_after_append
        async with pipeline(tts) as (worker, frames, errors):
            await worker.queue_frames([TTSSpeakFrame("A bounded transaction.")])
            await asyncio.wait_for(appended.wait(), 2)
            await worker.queue_frames([InterruptionFrame()])
            await asyncio.sleep(0.02)
            release.set()
            await until(lambda: any(e["type"] == "input_text.cancel" for e in commands))
            kinds = [e["type"] for e in commands]
            assert kinds.index("input_text.flush") < kinds.index("input_text.cancel")
            assert phrases == ["A bounded transaction."]
            assert not errors


def test_settings_validate_pronunciation_and_transport(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICEBOX_CLIENT_KEY", "voicebox")
    monkeypatch.setenv("VOICEBOX_LLM", "echo")
    monkeypatch.setenv("VOICEBOX_TTS_TRANSPORT", "websocket")
    monkeypatch.setenv("VOICEBOX_FIRST_CHUNK_CHARS", "40")
    monkeypatch.setenv("VOICEBOX_TEXT_FLUSH_CHARS", "60")
    filename = tmp_path / "pronunciations.json"
    filename.write_text(json.dumps({"API": "A P I"}))
    monkeypatch.setenv("VOICEBOX_PRONUNCIATIONS_FILE", str(filename))
    settings = Settings.from_env()
    assert settings.tts_settings.pronunciation_replacements == {"API": "A P I"}
    assert settings.tts_settings.text_flush_chars == 60
    monkeypatch.setenv("VOICEBOX_TTS_TRANSPORT", "http")
    with pytest.raises(ValueError, match="require websocket"):
        Settings.from_env()
    filename.write_text('["invalid"]')
    with pytest.raises(ValueError, match="JSON object"):
        pronunciation_file(str(filename))
    with pytest.raises(ValueError, match="8192"):
        complete_settings(
            "tts-1",
            "af_heart",
            VoiceboxTTSSettings(pronunciation_replacements={str(i): "a" * 256 for i in range(40)}),
        )
