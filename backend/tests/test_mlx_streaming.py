"""Tests for the thread-to-asyncio bridge and the MLX backend's generator, without MLX models."""

import threading
import time

import numpy as np
import pytest

from backend.backends.base import iterate_in_thread
from backend.backends.mlx_backend import MLXTTSBackend


async def test_bridge_forwards_items_in_order():
    def make_iter():
        yield from range(5)

    assert [item async for item in iterate_in_thread(make_iter)] == [0, 1, 2, 3, 4]


async def test_bridge_reraises_iterator_exceptions():
    def make_iter():
        yield 1
        raise RuntimeError("boom")

    stream = iterate_in_thread(make_iter)
    assert await stream.__anext__() == 1
    with pytest.raises(RuntimeError, match="boom"):
        await stream.__anext__()


async def test_bridge_stops_and_closes_the_iterator_when_the_consumer_leaves():
    closed = threading.Event()
    produced: list[int] = []

    def make_iter():
        try:
            for i in range(200):
                produced.append(i)
                time.sleep(0.005)
                yield i
        finally:
            closed.set()

    stop = threading.Event()
    stream = iterate_in_thread(make_iter, stop=stop)
    assert await stream.__anext__() == 0
    await stream.aclose()

    assert stop.is_set()
    assert closed.is_set()  # the worker was awaited, so the generator is closed
    assert len(produced) < 200  # and it did not run to completion


class FakeResult:
    def __init__(self, audio, sample_rate=24000):
        self.audio = audio
        self.sample_rate = sample_rate


class FakeModel:
    """Mimics mlx-audio's ``generate``: a generator with optional ``stream`` support."""

    def __init__(self, fail_after=None):
        self.calls: list[dict] = []
        self.fail_after = fail_after

    def generate(self, text, lang_code="auto", ref_audio=None, ref_text="", stream=False, streaming_interval=2.0):
        self.calls.append(
            {
                "text": text,
                "lang_code": lang_code,
                "ref_audio": ref_audio,
                "stream": stream,
                "interval": streaming_interval,
            }
        )
        pieces = 3 if stream else 1
        for i in range(pieces):
            if self.fail_after is not None and i == self.fail_after and ref_audio is not None:
                raise RuntimeError("clone failed")
            yield FakeResult(np.full(4, 0.1 * (i + 1), dtype=np.float64))


def make_backend(model):
    backend = MLXTTSBackend.__new__(MLXTTSBackend)
    backend.model = model
    backend.model_size = "0.6B"
    backend._current_model_size = "0.6B"
    return backend


@pytest.fixture
def ref_audio(tmp_path):
    path = tmp_path / "ref.wav"
    path.write_bytes(b"RIFF")
    return str(path)


def test_non_streaming_drains_all_results(ref_audio):
    model = FakeModel()
    backend = make_backend(model)

    pieces = list(
        backend._iter_generate_sync("hello", {"ref_audio": ref_audio, "ref_text": "hi"}, "en", None, stream=False)
    )

    assert len(pieces) == 1
    assert pieces[0][0].dtype == np.float32
    assert model.calls[0]["stream"] is False
    assert model.calls[0]["ref_audio"] == ref_audio
    assert model.calls[0]["lang_code"] == "english"


def test_streaming_passes_the_stream_kwargs():
    model = FakeModel()
    backend = make_backend(model)

    pieces = list(backend._iter_generate_sync("hello", {}, "en", None, stream=True))

    assert len(pieces) == 3
    assert model.calls[0]["stream"] is True
    assert model.calls[0]["interval"] == MLXTTSBackend.STREAMING_INTERVAL_S
    assert all(sr == 24000 and audio.dtype == np.float32 for audio, sr in pieces)


def test_clone_failure_before_any_audio_falls_back_to_the_plain_voice(ref_audio):
    model = FakeModel(fail_after=0)
    backend = make_backend(model)

    pieces = list(
        backend._iter_generate_sync("hello", {"ref_audio": ref_audio, "ref_text": "hi"}, "en", None, stream=True)
    )

    assert len(pieces) == 3
    assert [call["ref_audio"] for call in model.calls] == [ref_audio, None]


def test_clone_failure_after_audio_was_emitted_is_raised(ref_audio):
    model = FakeModel(fail_after=1)
    backend = make_backend(model)

    with pytest.raises(RuntimeError, match="clone failed"):
        list(backend._iter_generate_sync("hello", {"ref_audio": ref_audio, "ref_text": "hi"}, "en", None, stream=True))

    assert len(model.calls) == 1


def test_missing_reference_file_generates_without_cloning(tmp_path):
    model = FakeModel()
    backend = make_backend(model)

    pieces = list(
        backend._iter_generate_sync("hello", {"ref_audio": str(tmp_path / "gone.wav")}, "en", None, stream=False)
    )

    assert len(pieces) == 1
    assert model.calls[0]["ref_audio"] is None


async def test_generate_stream_yields_pieces_through_the_bridge(monkeypatch):
    model = FakeModel()
    backend = make_backend(model)

    async def no_load(model_size=None):
        return None

    monkeypatch.setattr(backend, "load_model_async", no_load)

    pieces = [piece async for piece in backend.generate_stream("hello", {}, "en")]

    assert len(pieces) == 3
    assert model.calls[0]["stream"] is True


async def test_generate_still_returns_one_array(monkeypatch):
    model = FakeModel()
    backend = make_backend(model)

    async def no_load(model_size=None):
        return None

    monkeypatch.setattr(backend, "load_model_async", no_load)

    audio, sample_rate = await backend.generate("hello", {}, "en")

    assert sample_rate == 24000
    assert audio.shape == (4,)
    assert audio.dtype == np.float32
    assert model.calls[0]["stream"] is False
