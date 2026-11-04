"""Regression checks for repeated real-audio conversation orchestration."""

import asyncio
from types import SimpleNamespace

import pytest

from e2e import Microphone, Recording, parse_args


@pytest.mark.asyncio
async def test_microphone_can_play_multiple_turns_without_replacing_track():
    mic = Microphone()
    try:
        mic.play(b"\1\0" * 480)
        with pytest.raises(RuntimeError, match="already speaking"):
            mic.play(b"\2\0" * 480)
        first = await mic.recv()
        assert mic.done.is_set()
        first_end = mic.speech_end
        mic.play(b"\2\0" * 480)
        assert not mic.done.is_set()
        second = await mic.recv()
        assert second.pts > first.pts
        assert mic.speech_end > first_end
        assert second.to_ndarray().ravel()[0] == 2
    finally:
        mic.stop()


@pytest.mark.asyncio
async def test_recording_does_not_mistake_previous_turn_for_new_reply():
    recording = Recording()
    recording.loud = [(0, 0.0)]
    with pytest.raises(TimeoutError):
        await recording.wait_audible(1, 0.01)
    assert await recording.wait_audible(0, 0.01) == 0.0


@pytest.mark.parametrize(
    "argv", [["--turns", "0"], ["--duration", "-1"], ["--idle-between", "nan"], ["--timeout", "0"]]
)
def test_bad_soak_limits_fail_before_launch(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_real_llm_mode_is_explicit_and_echo_is_default():
    assert parse_args([]).llm == "echo"
    args = parse_args(["--llm", "configured", "--turns", "3", "--barge-in", "--duration", "300"])
    assert args.llm == "configured" and args.turns == 3 and args.barge_in


@pytest.mark.asyncio
async def test_configured_llm_requires_credentials_even_if_parent_selected_echo(monkeypatch):
    from e2e import check

    for name in ("LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VOICEBOX_CLIENT_KEY", "test-client")
    monkeypatch.setenv("VOICEBOX_LLM", "echo")
    with pytest.raises(ValueError, match="Set OPENAI_API_KEY"):
        await check(parse_args(["--llm", "configured"]))


@pytest.mark.asyncio
async def test_pending_phrases_apply_backpressure_without_discarding_text():
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        TTSStoppedFrame,
    )
    from test_streaming_tts import audio, pipeline, speech_server, until

    from streaming_tts import VoiceboxStreamingTTSService

    pending = []

    async def respond(ws, flush_id):
        pending.append((ws, flush_id))

    async with speech_server(respond) as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url)
        sentences = [f"This is sentence number {n}." for n in range(8)]
        async with pipeline(tts) as (worker, frames, errors):
            await worker.queue_frames(
                [
                    LLMFullResponseStartFrame(),
                    *map(LLMTextFrame, sentences),
                    LLMFullResponseEndFrame(),
                ]
            )
            await until(lambda: len(pending) == 4)
            await asyncio.sleep(0.05)
            assert len(pending) == len(tts._pending) == 4
            for index in range(8):
                await until(lambda index=index: len(pending) > index)
                await audio(*pending[index])
            await until(lambda: any(isinstance(frame, TTSStoppedFrame) for frame in frames))
            assert phrases == sentences
            assert not tts._pending and not errors


@pytest.mark.asyncio
async def test_stalled_sdk_audio_consumer_fails_with_bounded_buffer(monkeypatch):
    from streaming_tts import MAX_BUFFERED_FRAMES, VoiceboxStreamingTTSService

    tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url="http://localhost:1")
    queue = asyncio.Queue()
    for _ in range(MAX_BUFFERED_FRAMES):
        queue.put_nowait(object())
    monkeypatch.setattr(tts, "audio_context_available", lambda _: True)
    monkeypatch.setattr(
        tts, "_audio_context_manager", SimpleNamespace(_contexts={"blocked": queue})
    )
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                tts._append_audio("blocked", SimpleNamespace(cancelled=False), b"\0\0"), 6
            )
        assert queue.qsize() == MAX_BUFFERED_FRAMES
    finally:
        await tts._close()


@pytest.mark.asyncio
async def test_overload_fails_without_retrying_the_rejected_phrase():
    from test_streaming_tts import speech_server

    from smoke import synthesize
    from streaming_tts import VoiceboxStreamingTTSService

    async def overloaded(ws, flush_id):
        await ws.send_json(
            {
                "type": "error",
                "flush_id": flush_id,
                "error": {"code": "queue_full", "message": "Queue full", "retry_after_s": 1},
            }
        )

    async with speech_server(overloaded) as (url, commands, phrases, settings, closed):
        tts = VoiceboxStreamingTTSService(api_key="voicebox", base_url=url)
        with pytest.raises(RuntimeError, match="queue_full"):
            await synthesize(tts, "This phrase is rejected.", 3)
        assert phrases == ["This phrase is rejected."]
        assert not tts.is_usable and tts._session.closed
