"""Tests for ``POST /generate/stream`` and its queued streaming job.

The route is called directly with fakes for the profile lookup, the engine
preparation, the backend and the request's receive channel (so a test can
hang up), so no model, database or HTTP server is needed.
``backend.routes.generations`` still imports torch transitively, so these run
under the project venv (``just test``).
"""

import asyncio
import logging
import struct
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException, Request
from pydantic import ValidationError

from backend import models
from backend.auth.principal import KeyLimits, Principal
from backend.routes import generations
from backend.services import generation as generation_service, task_queue
from backend.services.inference_slots import InferenceSlot
from backend.utils.wav_stream import WAV_HEADER_BYTES, float_to_pcm16_bytes, streaming_wav_header

SAMPLE_RATE = 24000
ADMIN = Principal(key_id="test", role="admin", via="header", limits=KeyLimits.defaults_for("admin"))
CLIENT = Principal(key_id="app", role="client", via="header", limits=KeyLimits.defaults_for("client"))
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
        self.rate = SAMPLE_RATE

    async def generate(self, text, voice_prompt, language="en", seed=None, instruct=None):
        self.calls.append(text)
        if self.gate is not None and len(self.calls) == self.gate_on_call:
            await self.gate.wait()
        if self.fail_with is not None:
            raise self.fail_with
        return np.full(len(text), 0.25, dtype=np.float32), self.rate


class FakeConnection:
    """The receive side of a request whose body has been read: nothing arrives until ``hang_up()``."""

    def __init__(self) -> None:
        self.gone = asyncio.Event()
        self.readers = 0  # receive() calls waiting right now

    async def receive(self) -> dict:
        self.readers += 1
        try:
            await self.gone.wait()
        finally:
            self.readers -= 1
        return {"type": "http.disconnect"}

    def hang_up(self) -> None:
        self.gone.set()

    def request(self) -> Request:
        return Request({"type": "http", "method": "POST", "path": "/generate/stream", "headers": []}, self.receive)


def connected() -> Request:
    return FakeConnection().request()


async def wait_until(condition, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        assert loop.time() < deadline, "condition not reached in time"
        await asyncio.sleep(0.005)


def holds_a_slot(owner: str = CLIENT.key_id) -> bool:
    """Whether *owner* still holds a pending-job slot (checked against a cap of one)."""
    try:
        task_queue.ensure_capacity(owner, 1)
    except task_queue.QueueFullError:
        return True
    return False


class FakeDb:
    """Just enough of a SQLAlchemy session for the effects-chain lookup."""

    def query(self, *_args):
        return self

    def get(self, *_args):
        return None

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

    async def prepare_engine(engine, model_size, profile_id, db, *, on_loading=None, voice_prompt=None):
        return generation_service.EnginePrep(
            backend=backend,
            voice_prompt={},
            trim_fn=None,
            runaway_detector=None,
            runaway_cut_fn=None,
        )

    async def ensure_cached(engine, model_size="default"):
        return None

    # The route reads the caller from the auth middleware's ContextVar and
    # charges its rate-limit buckets; neither exists in this direct-call test.
    monkeypatch.setattr(generations, "get_principal", lambda: ADMIN)
    monkeypatch.setattr(generation_service, "charge", lambda *args, **kwargs: None)
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
    response = await generations.stream_speech(request(), connected(), db=FakeDb())
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
    response = await generations.stream_speech(request(format="pcm", text="Hello world."), connected(), db=FakeDb())
    body = await drain(response)

    assert response.media_type == "audio/pcm"
    assert "content-disposition" not in response.headers
    assert len(body) == 2 * len("Hello world.")
    assert struct.unpack("<h", body[:2])[0] == int(np.floor(0.25 * 32768))


async def test_pcm_keeps_the_engine_rate(fake_backend):
    """Unlike ``/v1`` pcm, the native stream reports its rate in a header and never resamples."""
    fake_backend.rate = 48000

    response = await generations.stream_speech(request(format="pcm", text="Hello world."), connected(), db=FakeDb())
    body = await drain(response)

    assert response.headers["x-voicebox-sample-rate"] == "48000"
    assert len(body) == 2 * len("Hello world.")


async def test_default_first_chunk_cap_makes_the_first_engine_call_short(fake_backend):
    text = "Sure, I can help with that. " + TEXT  # 184 chars, one chunk without the cap

    response = await generations.stream_speech(
        request(text=text, max_chunk_chars=800, first_chunk_chars=models.DEFAULT_FIRST_CHUNK_CHARS),
        connected(),
        db=FakeDb(),
    )
    await drain(response)

    assert fake_backend.calls == ["Sure, I can help with that.", TEXT]


async def test_null_first_chunk_cap_keeps_the_normal_chunking(fake_backend):
    text = "Sure, I can help with that. " + TEXT

    response = await generations.stream_speech(request(text=text, max_chunk_chars=800), connected(), db=FakeDb())
    await drain(response)

    assert fake_backend.calls == [text]


async def test_engine_failure_before_first_chunk_is_a_real_http_error(fake_backend, caplog):
    fake_backend.fail_with = RuntimeError("model exploded")

    with pytest.raises(HTTPException) as excinfo:
        await generations.stream_speech(request(), connected(), db=FakeDb())

    # Engine errors can carry paths, so the client gets a generic message
    # and the real one goes to the log.
    assert excinfo.value.status_code == 500
    assert "model exploded" not in excinfo.value.detail
    assert "model exploded" in caplog.text


async def test_value_error_before_first_chunk_is_a_400(fake_backend):
    fake_backend.fail_with = ValueError("bad reference audio")

    with pytest.raises(HTTPException) as excinfo:
        await generations.stream_speech(request(), connected(), db=FakeDb())

    assert excinfo.value.status_code == 400


async def test_unknown_profile_is_a_404(fake_backend):
    with pytest.raises(HTTPException) as excinfo:
        await generations.stream_speech(request(profile_id="missing"), connected(), db=FakeDb())

    assert excinfo.value.status_code == 404


async def test_stream_waits_behind_queued_generation_jobs(fake_backend):
    release = asyncio.Event()
    started = asyncio.Event()

    async def blocking_job():
        started.set()
        await release.wait()

    task_queue.enqueue_generation("gen-blocking", blocking_job())
    await asyncio.wait_for(started.wait(), timeout=1)

    stream_task = asyncio.create_task(generations.stream_speech(request(), connected(), db=FakeDb()))
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

    response = await generations.stream_speech(request(), connected(), db=FakeDb())
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


async def test_generate_stream_accepts_speed(fake_backend, monkeypatch):
    calls: list[float] = []

    def fake_stretch(audio, sample_rate, speed):
        calls.append(speed)
        return audio[::2]

    monkeypatch.setattr(generation_service, "_time_stretch", fake_stretch)
    response = await generations.stream_speech(request(format="pcm", speed=2.0), connected(), db=FakeDb())
    body = await drain(response)
    assert calls
    assert set(calls) == {2.0}
    assert len(body) < len(TEXT) * 2

    with pytest.raises(ValidationError):
        request(speed=0.1)


async def test_the_disconnect_watcher_stops_before_the_response(fake_backend):
    connection = FakeConnection()

    response = await generations.stream_speech(request(), connection.request(), db=FakeDb())

    assert connection.readers == 0  # the StreamingResponse will be the only reader
    await drain(response)


async def test_hanging_up_while_queued_skips_the_job_and_frees_the_slot(fake_backend, monkeypatch, caplog):
    monkeypatch.setattr(generations, "get_principal", lambda: CLIENT)
    release = asyncio.Event()

    async def blocking_job():
        await release.wait()

    task_queue.enqueue_generation("gen-blocking", blocking_job())
    connection = FakeConnection()
    handler = asyncio.create_task(generations.stream_speech(request(), connection.request(), db=FakeDb()))
    await wait_until(lambda: task_queue.pending_count() == 2)
    assert holds_a_slot()

    with caplog.at_level(logging.INFO, logger=generation_service.__name__):
        connection.hang_up()
        response = await asyncio.wait_for(handler, timeout=1)

    assert response.status_code == 499  # nginx's "client closed request"; nothing reaches the client
    assert task_queue.pending_count() == 1  # just the blocking job
    assert not holds_a_slot()
    assert "its job was queued" in caplog.text
    assert connection.readers == 0

    release.set()
    await wait_until(lambda: task_queue.pending_count() == 0)
    await asyncio.sleep(0.05)
    assert fake_backend.calls == []  # the worker skipped it


async def test_hanging_up_during_the_first_synthesis_cancels_the_job(fake_backend, monkeypatch):
    monkeypatch.setattr(generations, "get_principal", lambda: CLIENT)
    fake_backend.gate = asyncio.Event()
    fake_backend.gate_on_call = 1
    connection = FakeConnection()
    handler = asyncio.create_task(generations.stream_speech(request(), connection.request(), db=FakeDb()))
    await wait_until(lambda: fake_backend.calls == SENTENCES[:1])

    connection.hang_up()
    response = await asyncio.wait_for(handler, timeout=1)
    assert response.status_code == 499

    # The engine call in flight always completes; nothing after it starts.
    fake_backend.gate.set()
    await wait_until(lambda: task_queue.pending_count() == 0)
    await asyncio.sleep(0.05)
    assert fake_backend.calls == SENTENCES[:1]
    assert not holds_a_slot()


async def test_hanging_up_before_the_job_is_queued_leaves_nothing_behind(fake_backend, monkeypatch):
    monkeypatch.setattr(generations, "get_principal", lambda: CLIENT)
    checking = asyncio.Event()
    outcome: list[str] = []

    async def slow_cache_check(engine, model_size="default"):
        checking.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise

    monkeypatch.setattr("backend.backends.ensure_model_cached_or_raise", slow_cache_check)
    connection = FakeConnection()
    handler = asyncio.create_task(generations.stream_speech(request(), connection.request(), db=FakeDb()))
    await asyncio.wait_for(checking.wait(), timeout=1)

    connection.hang_up()
    response = await asyncio.wait_for(handler, timeout=1)

    assert response.status_code == 499
    assert outcome == ["cancelled"]
    assert task_queue.pending_count() == 0
    assert not holds_a_slot()
    await asyncio.sleep(0.05)
    assert fake_backend.calls == []


async def test_hanging_up_during_the_personality_rewrite_keeps_the_llm_slot_until_it_ends(fake_backend, monkeypatch):
    """The LLM runs in a worker thread that the hang-up cannot stop, so the next LLM call must wait for it."""
    started, release = threading.Event(), threading.Event()

    def rewrite_in_thread(text: str) -> SimpleNamespace:
        started.set()
        release.wait(5)
        return SimpleNamespace(text=text)

    async def rewrite_as_profile(character, text):
        return await asyncio.to_thread(rewrite_in_thread, text)

    slot = InferenceSlot("llm", wait_timeout_s=5)
    profile = SimpleNamespace(id="p1", personality="A cheerful pirate.", effects_chain=None)

    async def get_profile(profile_id, db):
        return profile

    monkeypatch.setattr(generation_service, "llm_slot", slot)
    monkeypatch.setattr(generation_service.personality, "rewrite_as_profile", rewrite_as_profile)
    monkeypatch.setattr(generations.profiles, "get_profile", get_profile)
    connection = FakeConnection()
    handler = asyncio.create_task(
        generations.stream_speech(request(personality=True), connection.request(), db=FakeDb())
    )
    await wait_until(started.is_set)

    connection.hang_up()
    response = await asyncio.wait_for(handler, timeout=1)

    assert response.status_code == 499
    assert slot.busy
    release.set()
    await wait_until(lambda: not slot.busy)
    assert task_queue.pending_count() == 0
    assert fake_backend.calls == []


async def test_a_stream_that_opens_as_the_client_hangs_up_is_abandoned(monkeypatch):
    session = generation_service.new_stream_session()
    opened = generation_service.OpenedStream(
        session=session, first_audio=np.zeros(4, dtype=np.float32), sample_rate=SAMPLE_RATE, engine="qwen", text="x"
    )

    async def open_stream(*_args, **_kwargs):
        return opened

    async def receive():
        return {"type": "http.disconnect"}

    # Both finish in the same loop step, so the first chunk and the disconnect arrive together.
    monkeypatch.setattr(generation_service, "open_stream", open_stream)
    with pytest.raises(generation_service.ClientDisconnectedError):
        await generation_service.open_stream_while_connected(receive, request(), FakeDb(), ADMIN)

    assert session.consumer_gone.is_set()


async def test_cancelling_the_handler_abandons_the_job(fake_backend):
    release = asyncio.Event()

    async def blocking_job():
        await release.wait()

    task_queue.enqueue_generation("gen-blocking", blocking_job())
    handler = asyncio.create_task(generations.stream_speech(request(), connected(), db=FakeDb()))
    await wait_until(lambda: task_queue.pending_count() == 2)

    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    await wait_until(lambda: task_queue.pending_count() == 1)

    release.set()
    await wait_until(lambda: task_queue.pending_count() == 0)
    await asyncio.sleep(0.05)
    assert fake_backend.calls == []


async def test_a_failure_after_the_first_chunk_aborts_the_body(fake_backend):
    """Ending the body cleanly would turn a failed synthesis into a short 200."""
    fake_backend.gate = asyncio.Event()
    fake_backend.gate_on_call = 2
    response = await generations.stream_speech(request(), connected(), db=FakeDb())
    first = await response.body_iterator.__anext__()
    assert len(first) > WAV_HEADER_BYTES

    fake_backend.fail_with = RuntimeError("late failure")
    fake_backend.gate.set()
    with pytest.raises(generation_service.StreamAbortedError, match="late failure"):
        await response.body_iterator.__anext__()
