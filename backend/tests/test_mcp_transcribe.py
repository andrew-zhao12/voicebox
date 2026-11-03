"""``voicebox.transcribe`` holds the Whisper slot until Whisper ends, even when the tool call is cancelled."""

from __future__ import annotations

import asyncio
import sys
import threading
import types

import numpy as np
import pytest
from fastmcp.exceptions import ToolError

from backend.mcp_server import tools
from backend.services import transcribe as transcribe_service
from backend.services.inference_slots import InferenceSlot


class FakeWhisper:
    """Transcribes in a worker thread that cancelling the tool call cannot stop."""

    model_size = "base"

    def __init__(self, slot: InferenceSlot) -> None:
        self.slot = slot
        self.busy_when_called: list[bool] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def is_loaded(self) -> bool:
        return True

    def _is_model_cached(self, model_size: str) -> bool:
        return True

    def _work(self) -> str:
        self.started.set()
        self.release.wait(5)
        self.finished.set()
        return "hello there"

    async def transcribe(self, path: str, language: str | None, model_size: str) -> str:
        self.busy_when_called.append(self.slot.busy)
        return await asyncio.to_thread(self._work)


@pytest.fixture
def slot(monkeypatch) -> InferenceSlot:
    fresh = InferenceSlot("whisper", wait_timeout_s=5)
    monkeypatch.setattr(tools, "whisper_slot", fresh)
    return fresh


@pytest.fixture
def whisper(monkeypatch, slot):
    fake = FakeWhisper(slot)
    monkeypatch.setattr(transcribe_service, "get_whisper_model", lambda: fake)
    audio = types.ModuleType("backend.utils.audio")
    audio.load_audio = lambda path, **_kwargs: (np.zeros(32000, dtype=np.float32), 16000)
    monkeypatch.setitem(sys.modules, "backend.utils.audio", audio)
    yield fake
    fake.release.set()


async def _until(predicate, timeout_s: float = 5.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.005)


async def test_a_cancelled_call_keeps_the_whisper_slot_until_whisper_ends(tmp_path, slot, whisper):
    caller = asyncio.create_task(tools._transcribe_file(tmp_path / "clip.wav", None, None))
    await _until(whisper.started.is_set)
    assert whisper.busy_when_called == [True]
    caller.cancel()  # what an MCP ``notifications/cancelled`` does to the tool call
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert slot.busy

    whisper.release.set()
    await _until(lambda: not slot.busy)
    assert whisper.finished.is_set()


async def test_the_transcript_comes_back_and_a_full_slot_is_a_tool_error(tmp_path, slot, whisper):
    whisper.release.set()
    result = await tools._transcribe_file(tmp_path / "clip.wav", "en", None)
    assert result == {"text": "hello there", "duration": 2.0, "language": "en", "model": "base"}
    assert not slot.busy

    slot.max_waiters = 0
    release = asyncio.Event()
    holder = asyncio.create_task(slot.run(release.wait))
    await asyncio.sleep(0)
    try:
        with pytest.raises(ToolError, match="busy"):
            await tools._transcribe_file(tmp_path / "clip.wav", None, None)
    finally:
        release.set()
        await holder
    assert whisper.busy_when_called == [True]
