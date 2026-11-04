"""Captures hold the Whisper and LLM slots around the model calls only (torch-free, routes aside)."""

from __future__ import annotations

import asyncio
import io
import sys
import threading
import types

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.auth.principal import KeyLimits, Principal, principal_scope
from backend.database import Base, Capture as DBCapture
from backend.services import captures
from backend.services.inference_slots import InferenceBusyError, InferenceSlot
from backend.services.refinement import RefinementFlags
from backend.services.speech_settings import update_settings
from backend.speech_settings_models import SpeechSettingsUpdate


class ThreadCall:
    """A model call in a worker thread, which cancelling its awaiter cannot stop."""

    def __init__(self, result) -> None:
        self.result = result
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def _work(self):
        self.started.set()
        self.release.wait(5)
        self.finished.set()
        return self.result

    async def __call__(self):
        return await asyncio.to_thread(self._work)


class FakeModel:
    """Whisper's ``transcribe`` and ``refine_transcript``, noting whether the slot was held when called."""

    model_size = "base"

    def __init__(self, call: ThreadCall, slot: InferenceSlot) -> None:
        self.call = call
        self.slot = slot
        self.busy_when_called: list[bool] = []

    async def transcribe(self, path: str, language: str | None, model_size: str) -> str:
        self.busy_when_called.append(self.slot.busy)
        return await self.call()

    async def refine_transcript(self, transcript: str, flags: RefinementFlags, model_size: str | None = None):
        self.busy_when_called.append(self.slot.busy)
        return await self.call()


async def _until(predicate, timeout_s: float = 5.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.fixture
def db(tmp_path):
    previous = config.get_data_dir()
    config.set_data_dir(tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
        config.set_data_dir(previous)


@pytest.fixture
def slots(monkeypatch) -> types.SimpleNamespace:
    whisper, llm = InferenceSlot("whisper", wait_timeout_s=5), InferenceSlot("llm", wait_timeout_s=5)
    monkeypatch.setattr(captures, "whisper_slot", whisper)
    monkeypatch.setattr(captures, "llm_slot", llm)
    return types.SimpleNamespace(whisper=whisper, llm=llm)


@pytest.fixture
def whisper(monkeypatch, slots):
    fake = FakeModel(ThreadCall("new transcript"), slots.whisper)
    monkeypatch.setattr(captures, "get_whisper_model", lambda: fake)
    yield fake
    fake.call.release.set()


@pytest.fixture
def llm(monkeypatch, slots):
    fake = FakeModel(ThreadCall(("refined", "0.6B")), slots.llm)
    monkeypatch.setattr(captures, "refine_transcript", fake.refine_transcript)
    yield fake
    fake.call.release.set()


@pytest.fixture
def decoder(monkeypatch):
    audio = types.ModuleType("backend.utils.audio")  # the real module needs librosa
    audio.load_audio = lambda path, **_kwargs: (np.zeros(16000, dtype=np.float32), 16000)
    monkeypatch.setitem(sys.modules, "backend.utils.audio", audio)


def stored_capture(db, transcript: str = "old transcript") -> DBCapture:
    path = config.get_captures_dir() / "stored.wav"
    sf.write(str(path), np.zeros(1600, dtype=np.float32), 16000)
    row = DBCapture(id="cap-1", audio_path=config.to_storage_path(path), source="file", transcript_raw=transcript)
    db.add(row)
    db.commit()
    return row


async def test_capture_and_retranscribe_apply_only_the_callers_saved_vocabulary(db, slots, decoder, monkeypatch):
    calls = []

    async def recognize(path, language, model_size, **options):
        assert slots.whisper.busy
        calls.append(options)
        return "recognized"

    monkeypatch.setattr(
        captures, "get_whisper_model", lambda: types.SimpleNamespace(model_size="base", transcribe=recognize)
    )
    update_settings(db, "first", SpeechSettingsUpdate(stt_prompt="Names", stt_hotwords=["Acme"]))
    first = Principal("first", "admin", "header", KeyLimits.defaults_for("admin"))
    other = Principal("other", "admin", "header", KeyLimits.defaults_for("admin"))
    with principal_scope(first):
        created = await captures.create_capture(
            audio_bytes=b"RIFF", filename="clip.wav", source="dictation", language=None, stt_model=None, db=db
        )
        await captures.retranscribe_capture(created.id, stt_model=None, language=None, db=db)
    with principal_scope(other):
        await captures.retranscribe_capture(created.id, stt_model=None, language=None, db=db)
    assert calls == [
        {"prompt": "Names", "hotwords": ["Acme"]},
        {"prompt": "Names", "hotwords": ["Acme"]},
        {},
    ]


async def _hold(slot: InferenceSlot) -> tuple[asyncio.Task, asyncio.Event]:
    release = asyncio.Event()
    holder = asyncio.create_task(slot.run(release.wait))
    await asyncio.sleep(0)
    return holder, release


async def _hold_both(slots: types.SimpleNamespace) -> list[tuple[asyncio.Task, asyncio.Event]]:
    return [await _hold(slots.whisper), await _hold(slots.llm)]


async def _release_both(holders: list[tuple[asyncio.Task, asyncio.Event]]) -> None:
    for task, release in holders:
        release.set()
        await task


async def test_a_cancelled_capture_keeps_the_whisper_slot_until_whisper_ends(db, slots, whisper, decoder):
    caller = asyncio.create_task(
        captures.create_capture(
            audio_bytes=b"RIFF", filename="clip.wav", source="dictation", language=None, stt_model=None, db=db
        )
    )
    await _until(whisper.call.started.is_set)
    assert whisper.busy_when_called == [True]
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert slots.whisper.busy

    whisper.call.release.set()
    await _until(lambda: not slots.whisper.busy)
    assert whisper.call.finished.is_set()
    assert db.query(DBCapture).count() == 0


async def test_a_cancelled_retranscribe_keeps_the_whisper_slot_and_writes_nothing(db, slots, whisper):
    stored_capture(db)
    caller = asyncio.create_task(captures.retranscribe_capture("cap-1", stt_model=None, language=None, db=db))
    await _until(whisper.call.started.is_set)
    assert whisper.busy_when_called == [True]
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert slots.whisper.busy

    whisper.call.release.set()
    await _until(lambda: not slots.whisper.busy)
    db.expire_all()
    assert db.get(DBCapture, "cap-1").transcript_raw == "old transcript"


async def test_a_cancelled_refine_keeps_the_llm_slot_until_the_llm_ends(db, slots, llm):
    stored_capture(db)
    caller = asyncio.create_task(captures.refine_capture("cap-1", RefinementFlags(), model_size=None, db=db))
    await _until(llm.call.started.is_set)
    assert llm.busy_when_called == [True]
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert slots.llm.busy

    llm.call.release.set()
    await _until(lambda: not slots.llm.busy)
    db.expire_all()
    assert db.get(DBCapture, "cap-1").transcript_refined is None


async def test_finished_calls_are_stored(db, slots, whisper, llm, decoder):
    whisper.call.release.set()
    llm.call.release.set()

    created = await captures.create_capture(
        audio_bytes=b"RIFF", filename="clip.wav", source="file", language="en", stt_model=None, db=db
    )
    assert (created.transcript_raw, created.stt_model, created.duration_ms) == ("new transcript", "base", 1000)

    stored_capture(db, transcript="before")
    refined = await captures.refine_capture("cap-1", RefinementFlags(), model_size="0.6B", db=db)
    assert (refined.transcript_refined, refined.llm_model) == ("refined", "0.6B")
    again = await captures.retranscribe_capture("cap-1", stt_model="small", language="de", db=db)
    assert (again.transcript_raw, again.stt_model, again.language, again.transcript_refined) == (
        "new transcript",
        "small",
        "de",
        None,
    )
    assert not slots.whisper.busy
    assert not slots.llm.busy


async def test_a_full_slot_turns_the_call_away_and_keeps_nothing(db, slots, whisper, llm, decoder):
    stored_capture(db)
    slots.whisper.max_waiters = slots.llm.max_waiters = 0
    whisper_holder, release_whisper = await _hold(slots.whisper)
    llm_holder, release_llm = await _hold(slots.llm)
    try:
        with pytest.raises(InferenceBusyError):
            await captures.create_capture(
                audio_bytes=b"RIFF", filename="clip.wav", source="file", language=None, stt_model=None, db=db
            )
        assert sorted(path.name for path in config.get_captures_dir().iterdir()) == ["stored.wav"]
        with pytest.raises(InferenceBusyError):
            await captures.retranscribe_capture("cap-1", stt_model=None, language=None, db=db)
        with pytest.raises(InferenceBusyError):
            await captures.refine_capture("cap-1", RefinementFlags(), model_size=None, db=db)

        # Lookups happen before the slot, so these answer without waiting for it.
        assert await captures.retranscribe_capture("missing", stt_model=None, language=None, db=db) is None
        assert await captures.refine_capture("missing", RefinementFlags(), model_size=None, db=db) is None
        (config.get_captures_dir() / "stored.wav").unlink()
        with pytest.raises(FileNotFoundError):
            await captures.retranscribe_capture("cap-1", stt_model=None, language=None, db=db)
    finally:
        release_whisper.set()
        release_llm.set()
        await asyncio.gather(whisper_holder, llm_holder)
    assert whisper.busy_when_called == []  # turned away before Whisper or the LLM was called
    assert llm.busy_when_called == []
    assert db.query(DBCapture).count() == 1
    assert db.get(DBCapture, "cap-1").transcript_raw == "old transcript"


def test_the_routes_answer_a_full_slot_with_429(db, slots, whisper, llm):
    pytest.importorskip("librosa", reason="the captures router imports the audio stack")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.database import get_db
    from backend.routes.captures import router

    stored_capture(db)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: db
    clip = io.BytesIO()
    sf.write(clip, np.zeros(1600, dtype=np.float32), 16000, format="WAV")

    with TestClient(app) as client:
        slots.whisper.max_waiters = slots.llm.max_waiters = 0
        holders = client.portal.call(_hold_both, slots)
        try:
            responses = [
                client.post("/captures", files={"file": ("clip.wav", clip.getvalue(), "audio/wav")}),
                client.post("/captures/cap-1/retranscribe", json={}),
                client.post("/captures/cap-1/refine", json={}),
            ]
        finally:
            client.portal.call(_release_both, holders)

    for response in responses:
        assert response.status_code == 429, response.text
        assert response.headers["retry-after"] == "5"
        assert "busy" in response.json()["detail"]
    assert sorted(path.name for path in config.get_captures_dir().iterdir()) == ["stored.wav"]
