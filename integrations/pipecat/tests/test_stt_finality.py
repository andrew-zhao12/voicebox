"""Delayed local VAD fragments must finish one turn, including beyond SDK fallbacks."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMContextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from services import VoiceboxRealtimeSTTService, VoiceboxSpeechTimeoutUserTurnStopStrategy


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


@asynccontextmanager
async def conversation(*, commit_timeout_s=2):
    commits, transcripts, contexts, errors = [], [], [], []
    socket = None

    async def provider(request):
        nonlocal socket
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_json({"type": "session.created"})
        async for message in socket:
            event = json.loads(message.data)
            if event["type"] == "session.update":
                await socket.send_json({"type": "session.updated"})
            elif event["type"] == "input_audio_buffer.commit":
                commits.append(event)
                await socket.send_json(
                    {"type": "input_audio_buffer.committed", "item_id": str(len(commits))}
                )
        return socket

    app = web.Application()
    app.router.add_get("/v1/realtime", provider)
    async with TestServer(app) as server:
        stt = VoiceboxRealtimeSTTService(
            api_key="test",
            base_url=str(server.make_url("/v1/realtime")).replace("http://", "ws://"),
            ttfs_p99_latency=0.03,
            commit_timeout_s=commit_timeout_s,
        )
        user, _ = LLMContextAggregatorPair(
            LLMContext(),
            user_params=LLMUserAggregatorParams(
                user_turn_stop_timeout=0.15,
                user_turn_strategies=UserTurnStrategies(
                    stop=[VoiceboxSpeechTimeoutUserTurnStopStrategy(stt, user_speech_timeout=0.2)]
                ),
            ),
        )

        class Observe(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, TranscriptionFrame):
                    transcripts.append((frame.text, frame.finalized))
                elif isinstance(frame, LLMContextFrame):
                    contexts.append(json.loads(json.dumps(frame.context.messages)))
                await self.push_frame(frame, direction)

        @stt.event_handler("on_error")
        async def on_error(_service, error):
            errors.append(error)

        worker = PipelineWorker(
            Pipeline([stt, Observe(), user, Observe()]),
            params=PipelineParams(audio_in_sample_rate=24000),
            enable_rtvi=False,
        )
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        running = asyncio.create_task(runner.run())
        try:
            await until(lambda: stt._session_ready)

            async def segment():
                await worker.queue_frames(
                    [
                        VADUserStartedSpeakingFrame(start_secs=0.01),
                        InputAudioRawFrame(b"\1\0" * 2400, 24000, 1),
                        VADUserStoppedSpeakingFrame(stop_secs=0.01),
                    ]
                )

            async def final(item_id, text):
                await socket.send_json(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": item_id,
                        "transcript": text,
                    }
                )

            yield stt, segment, final, commits, transcripts, contexts, errors, socket
        finally:
            await runner.cancel()
            await running
        assert not stt.pending_transcriptions
        assert stt._commit_watchdog is None


@pytest.mark.asyncio
@pytest.mark.parametrize("already_forwarded", [False, True])
async def test_delayed_fragment_survives_provider_and_controller_timeouts(already_forwarded):
    async with conversation() as (stt, segment, final, commits, transcripts, contexts, _, socket):
        await segment()
        await until(lambda: len(commits) == 1)
        if already_forwarded:
            await final("1", "Sorry to interrupt.")
            await until(lambda: len(transcripts) == 1)
            assert transcripts[0][1]
        await segment()
        await until(lambda: len(commits) == 2)
        # An automatic silent cap is unrelated to either explicitly committed fragment.
        await socket.send_json({"type": "input_audio_buffer.speech_stopped", "item_id": "auto"})
        await final("auto", "")
        if not already_forwarded:
            await final("1", "Sorry to interrupt.")
            await until(lambda: len(transcripts) == 1)
            assert not transcripts[0][1]
        assert stt.pending_transcriptions
        await asyncio.sleep(0.55)
        assert not contexts
        await final("2", "What time is it?")
        await until(lambda: len(contexts) == 1)
        assert transcripts[-1] == ("What time is it?", True)
        assert contexts[0] == [{"role": "user", "content": "Sorry to interrupt. What time is it?"}]
        await asyncio.sleep(0.25)
        assert len(contexts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_empty_or_failed_last_fragment_releases_earlier_words(failed):
    async with conversation() as (stt, segment, final, commits, _, contexts, errors, socket):
        await segment()
        await segment()
        await until(lambda: len(commits) == 2)
        await final("1", "Keep these words.")
        await asyncio.sleep(0.3)
        assert not contexts
        if failed:
            await socket.send_json(
                {
                    "type": "conversation.item.input_audio_transcription.failed",
                    "item_id": "2",
                    "error": {"message": "Recognition failed"},
                }
            )
        else:
            await final("2", "")
        await until(lambda: len(contexts) == 1)
        assert contexts[0] == [{"role": "user", "content": "Keep these words."}]
        assert bool(errors) is failed
        assert not stt.pending_transcriptions


@pytest.mark.asyncio
async def test_missing_final_has_bounded_explicit_failure():
    async with conversation(commit_timeout_s=0.08) as (
        stt,
        segment,
        _,
        commits,
        _,
        contexts,
        errors,
        _,
    ):
        await segment()
        await until(lambda: len(commits) == 1)
        await until(lambda: bool(errors))
        assert "commit timed out" in errors[0].error
        assert not contexts
        assert not stt.pending_transcriptions
