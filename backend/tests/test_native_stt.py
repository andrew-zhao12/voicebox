"""Native decoder state consumes new audio once and survives caller cancellation safely."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from backend.services import native_stt
from backend.services.inference_slots import InferenceSlot
from backend.services.realtime_stt import Work
from backend.utils.subtitles import Transcript


class Stream:
    def __init__(self):
        self.chunks = []
        self.closed = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def accept_audio(self, audio):
        self.started.set()
        await self.release.wait()
        self.chunks.append(audio.copy())
        return "hello"

    async def finish(self):
        return Transcript("hello world")

    async def close(self):
        self.closed = True


@pytest.fixture
def setup(monkeypatch):
    streams = []

    async def open_stream(*args, **kwargs):
        stream = Stream()
        streams.append(stream)
        return stream

    monkeypatch.setattr(native_stt, "whisper_slot", InferenceSlot("test"))
    monkeypatch.setattr(
        native_stt.transcribe, "resolve_stt", lambda model: (SimpleNamespace(open_stream=open_stream), "test", None)
    )
    monkeypatch.setattr(native_stt.transcribe, "validate_options", lambda *a, **kw: None)
    return native_stt.NativeTranscriber(), streams


def work(kind, samples, item="one"):
    return Work(kind, item, np.arange(samples, dtype=np.float32), "en", "native")


async def test_chunks_are_not_redecoded_and_utterances_have_distinct_states(setup):
    native, streams = setup
    assert await native.transcribe(work("partial", 10)) == "hello"
    await native.transcribe(work("partial", 30))
    await native.transcribe(work("partial", 5, "two"))
    final = await native.transcribe(work("final", 40))
    assert final.text == "hello world"
    assert [list(c) for c in streams[0].chunks] == [list(range(10)), list(range(10, 30)), list(range(30, 40))]
    assert streams[0].closed
    assert not streams[1].closed
    await native.prune(set())
    assert streams[1].closed


async def test_disconnect_defers_close_until_shielded_inference_finishes(setup):
    native, streams = setup
    await native.transcribe(work("partial", 10))
    stream = streams[0]
    stream.release.clear()
    stream.started.clear()
    task = asyncio.create_task(native.transcribe(work("partial", 20)))
    await stream.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await native.prune(set())
    assert not stream.closed
    assert native_stt.whisper_slot.busy
    stream.release.set()
    for _ in range(100):
        if stream.closed and not native_stt.whisper_slot.busy:
            break
        await asyncio.sleep(0.001)
    assert stream.closed
    assert not native_stt.whisper_slot.busy


async def test_failed_decoder_cannot_report_an_empty_success(setup, monkeypatch):
    native, streams = setup
    await native.transcribe(work("partial", 10))

    async def fail(audio):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(streams[0], "accept_audio", fail)
    with pytest.raises(RuntimeError, match="decoder failed"):
        await native.transcribe(work("partial", 20))
    assert streams[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        await native.transcribe(work("final", 30))


async def test_decoder_cleanup_failure_does_not_strand_other_sessions(setup, monkeypatch):
    native, streams = setup
    await native.transcribe(work("partial", 10, "one"))
    await native.transcribe(work("partial", 10, "two"))

    async def fail_close():
        raise RuntimeError("isolated worker exited during close")

    monkeypatch.setattr(streams[0], "close", fail_close)
    await native.prune(set())
    assert not native.states
    assert streams[1].closed


async def test_decoder_cleanup_failure_preserves_original_inference_error(setup, monkeypatch):
    native, streams = setup
    await native.transcribe(work("partial", 10))

    async def fail_audio(_audio):
        raise ValueError("original inference error")

    async def fail_close():
        raise RuntimeError("cleanup error")

    monkeypatch.setattr(streams[0], "accept_audio", fail_audio)
    monkeypatch.setattr(streams[0], "close", fail_close)
    with pytest.raises(ValueError, match="original inference error"):
        await native.transcribe(work("partial", 20))
    assert not native_stt.whisper_slot.busy
