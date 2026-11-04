"""Model administration serializes STT work and preserves live decoder state without torch."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from backend import backends, models
from backend.routes import models as routes
from backend.services import inference_slots
from backend.services.inference_slots import InferenceSlot


@pytest.fixture
def registry(monkeypatch, tmp_path):
    slot = InferenceSlot("whisper", wait_timeout_s=1)
    monkeypatch.setattr(inference_slots, "whisper_slot", slot)
    config = SimpleNamespace(
        model_name="nemotron-test",
        display_name="Native STT",
        engine="nemotron",
        model_type="stt",
        model_size="test",
        hf_repo_id="test/native",
        supports_streaming=True,
    )
    backend = SimpleNamespace(active_streams=0, runtime_available=lambda: False)
    monkeypatch.setattr(backends, "get_stt_backend_for_engine", lambda engine: backend)
    monkeypatch.setattr(backends, "get_model_config", lambda name: config if name == config.model_name else None)
    monkeypatch.setattr(backends, "get_all_model_configs", lambda: [config])
    monkeypatch.setattr(backends, "check_model_loaded", lambda cfg: False)
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.constants = SimpleNamespace(HF_HUB_CACHE=str(tmp_path))
    fake_hub.scan_cache_dir = lambda: SimpleNamespace(repos=[])
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    manager = SimpleNamespace(get_pending_downloads=list)
    monkeypatch.setattr(routes, "get_task_manager", lambda: manager)
    return config, backend, slot, manager


@pytest.mark.parametrize("operation", ["unload", "delete", "download"])
async def test_admin_mutations_refuse_active_native_streams(registry, operation):
    config, backend, _slot, _manager = registry
    backend.active_streams = 1
    app = FastAPI()
    app.include_router(routes.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        if operation == "delete":
            result = await client.delete(f"/models/{config.model_name}")
        elif operation == "download":
            result = await client.post("/models/download", json={"model_name": config.model_name})
        else:
            result = await client.post(f"/models/{config.model_name}/unload")
    assert result.status_code == 409
    assert result.headers["retry-after"] == "5"


async def test_stream_admitted_while_admin_waits_is_rechecked(registry):
    config, backend, slot, _manager = registry
    release = asyncio.Event()
    holder = asyncio.create_task(slot.run(release.wait))
    while not slot.busy:
        await asyncio.sleep(0)
    called = []
    action = asyncio.create_task(routes._model_action(config, lambda: called.append(True)))
    await asyncio.sleep(0.01)
    backend.active_streams = 1
    release.set()
    await holder
    with pytest.raises(HTTPException) as error:
        await action
    assert error.value.status_code == 409
    assert called == []


async def test_admin_unload_executes_with_the_stt_slot_reserved(registry, monkeypatch):
    config, _backend, slot, _manager = registry
    busy = []

    def unload(cfg):
        busy.append(slot.busy)
        assert cfg is config
        return True

    monkeypatch.setattr(backends, "unload_model_by_config", unload)
    assert await routes.unload_model_by_name(config.model_name) == {
        "message": "Model nemotron-test unloaded successfully"
    }
    assert busy == [True]
    assert not slot.busy


async def test_background_download_holds_the_stt_slot_and_records_completion(registry, monkeypatch):
    config, _backend, slot, manager = registry
    events = []

    async def load():
        events.append(("load", slot.busy, config.model_size))

    manager.start_download = lambda name: events.append(("start", name))
    manager.complete_download = lambda name: events.append(("complete", name))
    manager.error_download = lambda name, error: events.append(("error", name, error))
    monkeypatch.setattr(backends, "get_model_load_func", lambda cfg: load)
    pending = []
    monkeypatch.setattr(routes, "create_background_task", pending.append)
    monkeypatch.setattr(routes, "get_progress_manager", lambda: SimpleNamespace(update_progress=lambda **kw: None))
    await routes.trigger_model_download(models.ModelDownloadRequest(model_name=config.model_name))
    assert len(pending) == 1
    await pending[0]
    assert events == [("start", "nemotron-test"), ("load", True, "test"), ("complete", "nemotron-test")]


async def test_model_status_exposes_stt_metadata_and_recognizes_nemo_weights(registry, tmp_path):
    config, _backend, _slot, _manager = registry
    weights = tmp_path / "models--test--native" / "snapshots" / "revision" / "model.nemo"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"test-weight-placeholder")
    result = await routes.get_model_status()
    [status] = result.models
    assert status.model_name == config.model_name
    assert status.model_type == "stt"
    assert status.engine == "nemotron"
    assert status.supports_streaming
    assert status.runtime_available is False
    assert status.downloaded
