"""Preload configuration and the retry loop, with the model loader stubbed out (torch-free)."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from backend import lifecycle
from backend.services import preload
from backend.services.inference_slots import InferenceSlot
from backend.services.task_queue import init_queue


@pytest.fixture(autouse=True)
def _clean_lifecycle():
    lifecycle.reset()
    yield
    lifecycle.reset()


def test_parse_model_names_accepts_commas_and_whitespace():
    assert preload.parse_model_names(None) == []
    assert preload.parse_model_names(" ") == []
    assert preload.parse_model_names("kokoro, whisper-turbo kokoro\nqwen-tts-1.7B,") == [
        "kokoro",
        "whisper-turbo",
        "qwen-tts-1.7B",
    ]


def test_configured_models_drops_unknown_names(caplog):
    registry = {"kokoro": object(), "whisper-turbo": object()}
    names = preload.configured_models(
        {"VOICEBOX_PRELOAD_MODELS": "kokoro,nope,whisper-turbo"},
        lookup=registry.get,
    )
    assert names == ["kokoro", "whisper-turbo"]
    assert "nope" in caplog.text
    assert preload.configured_models({}, lookup=registry.get) == []


async def test_run_marks_models_ready_and_retries_failures(monkeypatch):
    init_queue(force=True)
    attempts: dict[str, int] = {}

    async def fake_load(name: str, done: asyncio.Event) -> None:
        attempts[name] = attempts.get(name, 0) + 1
        try:
            if name == "flaky" and attempts[name] == 1:
                lifecycle.preload_failed(name, "hub down")
            else:
                lifecycle.preload_ready(name)
        finally:
            done.set()

    monkeypatch.setattr(preload, "_load_one", fake_load)
    monkeypatch.setattr(preload, "RETRY_S", 0.01)

    await preload.run(["kokoro", "flaky"])

    assert attempts == {"kokoro": 1, "flaky": 2}
    ready, body = lifecycle.readiness(True)
    assert ready
    assert body["models"]["ready"] == ["kokoro", "flaky"]
    init_queue(force=True)
    await asyncio.sleep(0)


async def test_run_stops_retrying_once_draining(monkeypatch):
    init_queue(force=True)
    attempts = 0

    async def always_fails(name: str, done: asyncio.Event) -> None:
        nonlocal attempts
        attempts += 1
        lifecycle.preload_failed(name, "no network")
        lifecycle.begin_drain("test")
        done.set()

    monkeypatch.setattr(preload, "_load_one", always_fails)
    monkeypatch.setattr(preload, "RETRY_S", 0.01)

    await preload.run(["kokoro"])

    assert attempts == 1
    assert lifecycle.readiness(True)[1]["models"]["failed"] == ["kokoro"]
    init_queue(force=True)
    await asyncio.sleep(0)


def test_the_cli_applies_the_models_dir_before_huggingface_hub_is_imported(tmp_path):
    """``python -m backend.preload`` must write into VOICEBOX_MODELS_DIR, not the default cache."""
    import subprocess
    import sys

    code = (
        "import os, sys; import backend.preload; "
        "print(os.environ.get('HF_HUB_CACHE')); print('huggingface_hub' in sys.modules)"
    )
    env = {**__import__("os").environ, "VOICEBOX_MODELS_DIR": str(tmp_path)}
    env.pop("HF_HUB_CACHE", None)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    hub_cache, hub_imported = out.stdout.split()
    assert hub_cache == str(tmp_path)
    assert hub_imported == "False"


def test_lazy_assets_are_prefetched_only_for_engines_that_need_them(monkeypatch):
    import types

    from backend import preload as preload_cli

    calls = []
    fake_hub = types.SimpleNamespace(snapshot_download=lambda **kw: calls.append(kw))
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)
    assert preload_cli.prefetch_lazy_assets("qwen") == 0
    assert calls == []
    assert preload_cli.prefetch_lazy_assets("kokoro") == 1
    assert calls == [{"repo_id": "hexgrad/Kokoro-82M", "allow_patterns": ["voices/*.pt"]}]


class ThreadLoad:
    """A model load in a worker thread, which cancelling its awaiter cannot stop."""

    def __init__(self, slot: InferenceSlot | None = None) -> None:
        self.slot = slot
        self.busy_when_called: list[bool] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def _work(self) -> None:
        self.started.set()
        self.release.wait(5)

    def __call__(self):
        if self.slot is not None:
            self.busy_when_called.append(self.slot.busy)
        return asyncio.to_thread(self._work)


@pytest.fixture
def registry(monkeypatch):
    """Point ``_load_one`` at a fake model of the requested engine; returns a setter."""

    def use(engine: str, load: ThreadLoad) -> None:
        monkeypatch.setattr("backend.backends.get_model_config", lambda name: SimpleNamespace(engine=engine))
        monkeypatch.setattr("backend.backends.check_model_loaded", lambda cfg: False)
        monkeypatch.setattr("backend.backends.get_model_load_func", lambda cfg: load)

    return use


async def _until(predicate, timeout_s: float = 5.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.parametrize(("engine", "slot_name"), [("whisper", "whisper_slot"), ("qwen_llm", "llm_slot")])
async def test_whisper_and_llm_loads_start_only_once_they_hold_their_slot(monkeypatch, registry, engine, slot_name):
    slot = InferenceSlot(engine, wait_timeout_s=5)
    monkeypatch.setattr(preload, slot_name, slot)
    load = ThreadLoad(slot)
    registry(engine, load)
    release_holder = asyncio.Event()
    holder = asyncio.create_task(slot.run(release_holder.wait))
    await asyncio.sleep(0)

    done = asyncio.Event()
    job = asyncio.create_task(preload._load_one("model", done))
    try:
        await asyncio.sleep(0.05)
        assert load.busy_when_called == []  # not even called while another caller holds the slot

        release_holder.set()
        await holder
        await _until(load.started.is_set)
        assert load.busy_when_called == [True]
    finally:
        release_holder.set()
        load.release.set()
    await job
    assert done.is_set()
    assert lifecycle.preload_state()["ready"] == ["model"]
    assert not slot.busy


@pytest.mark.parametrize(("engine", "slot_name"), [("whisper", "whisper_slot"), ("qwen_llm", "llm_slot")])
async def test_a_cancelled_preload_keeps_the_slot_until_its_load_ends(monkeypatch, registry, engine, slot_name):
    slot = InferenceSlot(engine, wait_timeout_s=5)
    monkeypatch.setattr(preload, slot_name, slot)
    load = ThreadLoad(slot)
    registry(engine, load)
    done = asyncio.Event()
    job = asyncio.create_task(preload._load_one("model", done))
    try:
        await _until(load.started.is_set)
        job.cancel()  # shutdown cancelling the queue job
        with pytest.raises(asyncio.CancelledError):
            await job
        assert done.is_set()
        assert lifecycle.preload_state()["failed"] == {"model": "cancelled"}
        assert slot.busy
    finally:
        load.release.set()
    await _until(lambda: not slot.busy)


async def test_tts_loads_take_no_slot(monkeypatch, registry):
    slots = {name: InferenceSlot(name) for name in ("whisper_slot", "llm_slot")}
    for name, slot in slots.items():
        monkeypatch.setattr(preload, name, slot)
    load = ThreadLoad()
    load.release.set()
    registry("kokoro", load)
    done = asyncio.Event()

    await preload._load_one("kokoro", done)

    assert load.started.is_set()
    assert lifecycle.preload_state()["ready"] == ["kokoro"]
    assert not any(slot.busy or slot.last_used for slot in slots.values())
