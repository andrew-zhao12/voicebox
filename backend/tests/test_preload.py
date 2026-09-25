"""Preload configuration and the retry loop, with the model loader stubbed out (torch-free)."""

from __future__ import annotations

import asyncio

import pytest

from backend import lifecycle
from backend.services import preload
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
