"""Idle unload sweeps with fake backends (torch-free)."""

from __future__ import annotations

import pytest

from backend.services import idle_unload, task_queue
from backend.services.inference_slots import llm_slot, whisper_slot


class FakeBackend:
    def __init__(self, loaded: bool = True) -> None:
        self.loaded = loaded

    def is_loaded(self) -> bool:
        return self.loaded

    def unload_model(self) -> None:
        self.loaded = False


@pytest.fixture(autouse=True)
def _reset_slots(monkeypatch):
    monkeypatch.setattr(whisper_slot, "last_used", None)
    monkeypatch.setattr(llm_slot, "last_used", None)
    monkeypatch.setattr(task_queue, "_engine_last_used", {})
    monkeypatch.setattr(task_queue, "_job_engine", {})


def test_configured_idle_s(caplog):
    assert idle_unload.configured_idle_s({}) is None
    assert idle_unload.configured_idle_s({"VOICEBOX_MODEL_IDLE_UNLOAD_S": "600"}) == 600.0
    assert idle_unload.configured_idle_s({"VOICEBOX_MODEL_IDLE_UNLOAD_S": "10"}) is None
    assert idle_unload.configured_idle_s({"VOICEBOX_MODEL_IDLE_UNLOAD_S": "soon"}) is None
    assert "idle unload disabled" in caplog.text


def _sweep(protected, now, backends, stt, llm):
    return idle_unload.sweep(
        300.0,
        protected,
        now=now,
        tts_backends=backends.__getitem__,
        tts_engines=set(backends),
        stt_unload=(stt.is_loaded, stt.unload_model),
        llm_unload=(llm.is_loaded, llm.unload_model),
    )


def test_sweep_unloads_idle_unprotected_models_only(monkeypatch):
    backends = {"kokoro": FakeBackend(), "qwen": FakeBackend(), "luxtts": FakeBackend(), "tada": FakeBackend(False)}
    stt, llm = FakeBackend(), FakeBackend()
    monkeypatch.setattr(task_queue, "_engine_last_used", {"kokoro": 100.0, "qwen": 100.0, "luxtts": 995.0, "tada": 1.0})
    monkeypatch.setattr(whisper_slot, "last_used", 100.0)
    monkeypatch.setattr(llm_slot, "last_used", 950.0)

    unloaded = _sweep({"qwen"}, 1000.0, backends, stt, llm)

    assert unloaded == ["kokoro", "whisper"]
    assert not backends["kokoro"].loaded
    assert backends["qwen"].loaded  # protected (preloaded)
    assert backends["luxtts"].loaded  # used recently
    assert not stt.loaded
    assert llm.loaded  # slot used recently


def test_sweep_skips_engines_with_jobs_and_idle_slots_never_used(monkeypatch):
    backends = {"kokoro": FakeBackend()}
    stt, llm = FakeBackend(), FakeBackend()
    monkeypatch.setattr(task_queue, "_engine_last_used", {"kokoro": 100.0})
    monkeypatch.setattr(task_queue, "_job_engine", {"g1": "kokoro"})

    assert _sweep(set(), 1000.0, backends, stt, llm) == []
    assert backends["kokoro"].loaded
    assert stt.loaded  # whisper_slot.last_used is None: never used, never unloaded
