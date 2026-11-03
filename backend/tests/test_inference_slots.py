"""Tests for the one-at-a-time inference slots."""

import asyncio
import gc
import logging
import sys
import threading
import time
import types

import numpy as np
import pytest

from backend.services.inference_slots import InferenceBusyError, InferenceSlot


class ThreadCall:
    """A stand-in for a Whisper call: a worker thread that cancelling its awaiter cannot stop."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def _work(self) -> str:
        self.started.set()
        self.release.wait(5)
        self.finished.set()
        return "transcript"

    async def __call__(self) -> str:
        return await asyncio.to_thread(self._work)


@pytest.fixture
def thread_call():
    call = ThreadCall()
    yield call
    call.release.set()


async def _until(predicate, timeout_s: float = 5.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.005)


async def test_cancelled_caller_keeps_the_slot_until_its_thread_ends(thread_call):
    slot = InferenceSlot("test", wait_timeout_s=5)
    caller = asyncio.create_task(slot.run(thread_call))
    await _until(thread_call.started.is_set)

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert slot.busy
    assert slot.last_used is None

    async def second() -> bool:
        return thread_call.finished.is_set()

    waiter = asyncio.create_task(slot.run(second))
    await asyncio.sleep(0.05)
    assert not waiter.done()

    thread_call.release.set()
    assert await asyncio.wait_for(waiter, 5) is True  # it started only after the thread had ended
    assert not slot.busy
    assert slot.last_used is not None


async def test_run_serves_waiters_in_order_and_turns_away_the_rest():
    slot = InferenceSlot("test", max_waiters=1, wait_timeout_s=5)
    release = asyncio.Event()
    order: list[str] = []

    async def first() -> str:
        order.append("first")
        await release.wait()
        return "one"

    async def second() -> str:
        order.append("second")
        return "two"

    holder = asyncio.create_task(slot.run(first))
    await asyncio.sleep(0)
    waiter = asyncio.create_task(slot.run(second))
    await asyncio.sleep(0)
    assert order == ["first"]

    calls: list[str] = []

    def third():
        calls.append("third")
        return asyncio.sleep(0)

    with pytest.raises(InferenceBusyError) as excinfo:
        await slot.run(third)
    assert excinfo.value.retry_after_s == 5
    assert calls == []  # never called without the slot

    release.set()
    assert await asyncio.gather(holder, waiter) == ["one", "two"]
    assert order == ["first", "second"]
    assert not slot.busy


async def test_run_waiting_times_out():
    slot = InferenceSlot("test", wait_timeout_s=0.01)
    release = asyncio.Event()
    holder = asyncio.create_task(slot.run(release.wait))
    await asyncio.sleep(0)

    with pytest.raises(InferenceBusyError):
        await slot.run(asyncio.sleep, 0)
    release.set()
    assert await holder is True


async def test_run_propagates_errors_and_releases_the_slot():
    slot = InferenceSlot("test", wait_timeout_s=5)

    async def fails() -> None:
        await asyncio.sleep(0)
        raise ValueError("model failed")

    with pytest.raises(ValueError, match="model failed"):
        await slot.run(fails)
    assert not slot.busy

    def not_awaitable() -> int:
        return 42

    with pytest.raises(TypeError):
        await slot.run(not_awaitable)
    assert not slot.busy
    assert await slot.run(asyncio.sleep, 0, result="ok") == "ok"


async def test_last_used_is_set_when_the_call_ends():
    slot = InferenceSlot("test")
    release = asyncio.Event()
    holder = asyncio.create_task(slot.run(release.wait))
    await asyncio.sleep(0)
    assert slot.busy
    assert slot.last_used is None

    before = time.monotonic()
    release.set()
    await holder
    assert before <= slot.last_used <= time.monotonic()


async def test_cancelling_a_waiting_caller_never_runs_its_call():
    slot = InferenceSlot("test", max_waiters=1, wait_timeout_s=5)
    release = asyncio.Event()
    ran: list[str] = []

    async def record(label: str) -> None:
        ran.append(label)

    holder = asyncio.create_task(slot.run(release.wait))
    await asyncio.sleep(0)
    waiter = asyncio.create_task(slot.run(record, "cancelled"))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    # With max_waiters=1 this would be turned away if the cancelled wait still counted.
    next_waiter = asyncio.create_task(slot.run(record, "next"))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(holder, next_waiter)
    assert ran == ["next"]
    assert not slot.busy


async def test_failure_after_the_caller_left_is_logged_not_reported_unretrieved(caplog):
    caplog.set_level(logging.DEBUG, logger="backend.services.inference_slots")
    loop = asyncio.get_running_loop()
    loop_errors: list[dict] = []
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    slot = InferenceSlot("test")
    release = asyncio.Event()

    async def fails_later() -> None:
        await release.wait()
        raise RuntimeError("decoder failed")

    try:
        caller = asyncio.create_task(slot.run(fails_later))
        await asyncio.sleep(0)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller

        release.set()
        await _until(lambda: not slot.busy)
        del caller
        gc.collect()  # a task whose exception nobody read reports it when collected
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(None)
    assert loop_errors == []
    assert "failed after its caller was cancelled" in caplog.text


async def test_transcription_holds_the_whisper_slot_after_its_caller_is_cancelled(monkeypatch, tmp_path, thread_call):
    from backend.routes import transcription
    from backend.utils.subtitles import Transcript

    slot = InferenceSlot("whisper", wait_timeout_s=5)
    monkeypatch.setattr(transcription, "whisper_slot", slot)
    audio = types.ModuleType("backend.utils.audio")  # the real module needs librosa
    audio.load_audio = lambda path, **_kwargs: (np.zeros(16000, dtype=np.float32), 16000)
    audio.save_audio = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "backend.utils.audio", audio)

    class FakeWhisper:
        model_size = "base"

        def is_loaded(self) -> bool:
            return True

        async def transcribe_detailed(self, path, language, model_size) -> Transcript:
            return Transcript(text=await thread_call(), language=language, duration=1.0)

    monkeypatch.setattr(transcription.transcribe, "get_whisper_model", lambda: FakeWhisper())

    # The realtime route cancels its worker this way whenever a socket closes.
    caller = asyncio.create_task(transcription.transcribe_file_detailed(str(tmp_path / "clip.wav"), None, None))
    await _until(thread_call.started.is_set)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert slot.busy

    thread_call.release.set()
    await _until(lambda: not slot.busy)
    assert thread_call.finished.is_set()
