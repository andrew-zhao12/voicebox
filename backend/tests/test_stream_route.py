"""Tests for ``POST /generate/stream`` and its queued streaming job.

The route is called directly with fakes for the profile lookup, the engine
preparation and the backend, so no model, database or HTTP server is needed.
``backend.routes.generations`` still imports torch transitively, so these run
under the project venv (``just test``).
"""

import asyncio
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from backend import models
from backend.routes import generations
from backend.services import generation as generation_service, task_queue
from backend.utils.wav_stream import WAV_HEADER_BYTES, float_to_pcm16_bytes, streaming_wav_header

SAMPLE_RATE = 24000
# Three sentences over 50 characters each: with the minimum max_chunk_chars of
# 100, only one fits per window, so the text always splits into exactly three chunks.
SENTENCES = [
    "The first sentence of this test text is fairly long.",
    "The second sentence of this test text is also long.",
    "The third sentence of this test text ends the run.",
]
TEXT = " ".join(SENTENCES)


class FakeBackend:
    def __init__(self):
        self.calls: list[str] = []
        self.gate: asyncio.Event | None = None
        self.gate_on_call = 0
        self.fail_with: Exception | None = None

    async def generate(self, text, voice_prompt, language="en", seed=None, instruct=None):
        self.calls.append(text)
        if self.gate is not None and len(self.calls) == self.gate_on_call:
            await self.gate.wait()
        if self.fail_with is not None:
            raise self.fail_with
        return np.full(len(text), 0.25, dtype=np.float32), SAMPLE_RATE


class FakeDb:
    """Just enough of a SQLAlchemy session for the effects-chain lookup."""

    def query(self, *_args):
        return self

    def filter_by(self, **_kwargs):
        return self

    def first(self):
        return SimpleNamespace(effects_chain=None)


@pytest.fixture
async def fake_backend(monkeypatch, tmp_path):
    # The streaming job opens a real session around prepare_engine, so point
    # the database at a throwaway directory for the duration of the test.
    from backend import config, database

    previous_data_dir = config.get_data_dir()
    config.set_data_dir(tmp_path)
    database.init_db()

    task_queue.init_queue(force=True)
    backend = FakeBackend()
    profile = SimpleNamespace(id="p1", personality=None, effects_chain=None)

    async def get_profile(profile_id, db):
        return profile if profile_id == "p1" else None

    async def prepare_engine(engine, model_size, profile_id, db, *, on_loading=None):
        return generation_service.EnginePrep(
            backend=backend,
            voice_prompt={},
            trim_fn=None,
            runaway_detector=None,
            runaway_cut_fn=None,
        )

    async def ensure_cached(engine, model_size="default"):
        return None

    monkeypatch.setattr(generations.profiles, "get_profile", get_profile)
    monkeypatch.setattr(generations.profiles, "validate_profile_engine", lambda profile, engine: None)
    monkeypatch.setattr(generation_service, "prepare_engine", prepare_engine)
    monkeypatch.setattr("backend.backends.ensure_model_cached_or_raise", ensure_cached)
    monkeypatch.setattr("backend.backends.engine_has_model_sizes", lambda engine: False)
    yield backend
    config.set_data_dir(previous_data_dir)


def request(**overrides):
    fields = {
        "profile_id": "p1",
        "text": TEXT,
        "normalize": False,
        "crossfade_ms": 0,
        "max_chunk_chars": 100,
        "first_chunk_chars": None,
    }
    fields.update(overrides)
    return models.StreamGenerationRequest(**fields)


async def drain(response) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return b"".join(chunks)


async def test_wav_stream_has_header_metadata_and_full_audio(fake_backend):
    response = await generations.stream_speech(request(), db=FakeDb())
    body = await drain(response)

    assert response.media_type == "audio/wav"
    assert response.headers["x-voicebox-sample-rate"] == str(SAMPLE_RATE)
    assert response.headers["x-voicebox-channels"] == "1"
    assert response.headers["x-voicebox-sample-format"] == "s16le"
    assert response.headers["x-voicebox-stream-mode"] == "chunked"
    assert response.headers["x-voicebox-job-id"].startswith(task_queue.STREAM_JOB_PREFIX)
    assert response.headers["content-disposition"].startswith("inline")

    assert fake_backend.calls == SENTENCES
    expected_audio = np.full(sum(len(sentence) for sentence in SENTENCES), 0.25, dtype=np.float32)
    assert body[:WAV_HEADER_BYTES] == streaming_wav_header(SAMPLE_RATE)
    assert body[WAV_HEADER_BYTES:] == float_to_pcm16_bytes(expected_audio)


async def test_pcm_format_sends_raw_samples_only(fake_backend):
    response = await generations.stream_speech(request(format="pcm", text="Hello world."), db=FakeDb())
    body = await drain(response)

    assert response.media_type == "audio/pcm"
    assert "content-disposition" not in response.headers
    assert len(body) == 2 * len("Hello world.")
    assert struct.unpack("<h", body[:2])[0] == int(np.floor(0.25 * 32768))


async def test_default_first_chunk_cap_makes_the_first_engine_call_short(fake_backend):
    text = "Sure, I can help with that. " + TEXT  # 184 chars, one chunk without the cap

    response = await generations.stream_speech(
        request(text=text, max_chunk_chars=800, first_chunk_chars=models.DEFAULT_FIRST_CHUNK_CHARS),
        db=FakeDb(),
    )
    await drain(response)

    assert fake_backend.calls == ["Sure, I can help with that.", TEXT]


async def test_null_first_chunk_cap_keeps_the_normal_chunking(fake_backend):
    text = "Sure, I can help with that. " + TEXT

    response = await generations.stream_speech(request(text=text, max_chunk_chars=800), db=FakeDb())
    await drain(response)

    assert fake_backend.calls == [text]


async def test_engine_failure_before_first_chunk_is_a_real_http_error(fake_backend):
    fake_backend.fail_with = RuntimeError("model exploded")

    with pytest.raises(HTTPException) as excinfo:
        await generations.stream_speech(request(), db=FakeDb())

    assert excinfo.value.status_code == 500
    assert "model exploded" in excinfo.value.detail


async def test_value_error_before_first_chunk_is_a_400(fake_backend):
    fake_backend.fail_with = ValueError("bad reference audio")

    with pytest.raises(HTTPException) as excinfo:
        await generations.stream_speech(request(), db=FakeDb())

    assert excinfo.value.status_code == 400


async def test_unknown_profile_is_a_404(fake_backend):
    with pytest.raises(HTTPException) as excinfo:
        await generations.stream_speech(request(profile_id="missing"), db=FakeDb())

    assert excinfo.value.status_code == 404


async def test_stream_waits_behind_queued_generation_jobs(fake_backend):
    release = asyncio.Event()
    started = asyncio.Event()

    async def blocking_job():
        started.set()
        await release.wait()

    task_queue.enqueue_generation("gen-blocking", blocking_job())
    await asyncio.wait_for(started.wait(), timeout=1)

    stream_task = asyncio.create_task(generations.stream_speech(request(), db=FakeDb()))
    await asyncio.sleep(0.05)
    assert not stream_task.done()
    assert fake_backend.calls == []

    release.set()
    response = await asyncio.wait_for(stream_task, timeout=2)
    await drain(response)
    assert len(fake_backend.calls) == 3


async def test_closing_the_response_cancels_the_rest_of_the_job(fake_backend):
    fake_backend.gate = asyncio.Event()
    fake_backend.gate_on_call = 2

    response = await generations.stream_speech(request(), db=FakeDb())
    first = await response.body_iterator.__anext__()
    assert len(first) > WAV_HEADER_BYTES

    await response.body_iterator.aclose()
    fake_backend.gate.set()
    await asyncio.sleep(0.1)

    assert fake_backend.calls == SENTENCES[:2]
    assert task_queue.cancel_generation(response.headers["x-voicebox-job-id"]) is None


async def test_streaming_job_forwards_failures_after_the_first_chunk(fake_backend):
    session = generation_service.new_stream_session()
    fake_backend.gate = asyncio.Event()
    fake_backend.gate_on_call = 2

    task_queue.enqueue_generation(
        session.job_id,
        generation_service.run_generation_stream(
            session=session,
            profile_id="p1",
            text=TEXT,
            language="en",
            engine="qwen",
            model_size=None,
            seed=None,
            instruct=None,
            normalize=True,
            effects_chain=None,
            max_chunk_chars=100,
            crossfade_ms=0,
            first_chunk_chars=None,
        ),
    )

    first = await asyncio.wait_for(session.frames.get(), timeout=1)
    assert isinstance(first, tuple)
    assert np.all(np.abs(first[0]) <= 0.85)  # normalized and clipped

    fake_backend.fail_with = RuntimeError("late failure")
    fake_backend.gate.set()
    second = await asyncio.wait_for(session.frames.get(), timeout=1)
    assert isinstance(second, RuntimeError)
