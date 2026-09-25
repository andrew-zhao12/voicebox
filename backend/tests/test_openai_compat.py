"""``/v1`` routes end to end through the security stack, with the engine faked.

``backend.routes`` imports torch transitively, so these run under the
project venv (``just test``), like ``test_stream_route.py``.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import config, database, lifecycle
from backend.auth.install import install_security
from backend.auth.settings import SecuritySettings
from backend.routes import register_routers
from backend.services import generation as generation_service, task_queue
from backend.utils import encode
from backend.utils.wav_stream import streaming_wav_header

SR = 24000
TEXT = "Hello from the OpenAI compatible route."


class FakeBackend:
    async def generate(self, text, voice_prompt, language="en", seed=None, instruct=None):
        return np.full(len(text), 0.25, dtype=np.float32), SR


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def api(tmp_path, monkeypatch):
    previous_data_dir = config.get_data_dir()
    config.set_data_dir(tmp_path)
    database.init_db()

    async def prepare_engine(engine, model_size, profile_id, db, *, on_loading=None):
        return generation_service.EnginePrep(
            backend=FakeBackend(), voice_prompt={}, trim_fn=None, runaway_detector=None, runaway_cut_fn=None
        )

    async def ensure_cached(engine, model_size="default"):
        return None

    monkeypatch.setattr(generation_service, "prepare_engine", prepare_engine)
    monkeypatch.setattr("backend.backends.ensure_model_cached_or_raise", ensure_cached)

    @asynccontextmanager
    async def lifespan(app):
        task_queue.init_queue(force=True)
        yield

    settings = SecuritySettings.from_env(
        frontend_dir=None,
        environ={
            "VOICEBOX_API_KEY_FILE": str(tmp_path / "api_key"),
            "VOICEBOX_API_KEYS_JSON": str(tmp_path / "api_keys.json"),
        },
    )
    app = FastAPI(lifespan=lifespan)
    runtime = install_security(app, settings)
    register_routers(app)
    runtime.startup()
    admin_key = (tmp_path / "api_key").read_text().strip()
    _record, client_key = runtime.keystore.create("app", "client", None)

    with TestClient(app, raise_server_exceptions=False) as client:
        created = client.post(
            "/profiles",
            json={
                "name": "Smoke Voice",
                "language": "en",
                "voice_type": "preset",
                "preset_engine": "kokoro",
                "preset_voice_id": "af_heart",
            },
            headers=bearer(admin_key),
        )
        assert created.status_code == 200, created.text
        yield SimpleNamespace(client=client, admin=admin_key, key=client_key, profile_id=created.json()["id"])
    lifecycle.reset()
    config.set_data_dir(previous_data_dir)


def test_speech_wav_streams_pcm_with_metadata(api):
    response = api.client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": TEXT, "voice": "Smoke Voice", "response_format": "wav"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["x-voicebox-sample-rate"] == str(SR)
    assert response.headers["x-voicebox-engine"] == "kokoro"
    assert response.headers["x-voicebox-job-id"].startswith(task_queue.STREAM_JOB_PREFIX)
    # Volume normalisation is on by default (as for /generate/stream), so compare
    # the shape of the audio rather than the raw amplitude.
    assert response.content[:44] == streaming_wav_header(SR)
    samples = np.frombuffer(response.content[44:], dtype="<i2")
    assert len(samples) == len(TEXT)
    assert samples[0] != 0
    assert np.all(samples == samples[0])


def test_speech_defaults_to_mp3_and_the_profile_engine(api):
    response = api.client.post(
        "/v1/audio/speech", json={"input": TEXT, "voice": "Smoke Voice"}, headers=bearer(api.key)
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.headers["x-voicebox-engine"] == "kokoro"  # tts-1 maps to the preset engine
    assert response.headers["content-disposition"] == 'inline; filename="speech.mp3"'
    decoded, rate = sf.read(io.BytesIO(response.content), dtype="float32")
    assert rate == SR
    assert abs(len(decoded) - len(TEXT)) < SR // 10  # mp3 padding


def test_voice_by_id_works_and_stock_names_need_a_default(api):
    response = api.client.post(
        "/v1/audio/speech",
        json={"input": TEXT, "voice": api.profile_id, "response_format": "pcm"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200
    assert len(response.content) == len(TEXT) * 2

    response = api.client.post("/v1/audio/speech", json={"input": TEXT, "voice": "alloy"}, headers=bearer(api.key))
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "voice_not_found"
    assert error["param"] == "voice"
    assert error["type"] == "invalid_request_error"
    assert "/v1/voices" in error["message"]


def test_model_speed_and_validation_errors_use_the_envelope(api):
    response = api.client.post(
        "/v1/audio/speech", json={"model": "gpt-9-tts", "input": TEXT, "voice": "Smoke Voice"}, headers=bearer(api.key)
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"

    response = api.client.post(
        "/v1/audio/speech", json={"input": TEXT, "voice": "Smoke Voice", "speed": 1.5}, headers=bearer(api.key)
    )
    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "speed other than 1.0 is not supported",
        "type": "invalid_request_error",
        "param": "speed",
        "code": "unsupported_value",
    }

    response = api.client.post("/v1/audio/speech", json={"voice": "Smoke Voice"}, headers=bearer(api.key))
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "input"
    assert error["code"] == "invalid_value"


def test_unsupported_format_without_ffmpeg(api, monkeypatch):
    monkeypatch.setattr(encode, "ffmpeg_path", lambda: None)
    response = api.client.post(
        "/v1/audio/speech",
        json={"input": TEXT, "voice": "Smoke Voice", "response_format": "aac"},
        headers=bearer(api.key),
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsupported_format"
    assert "ffmpeg" in error["message"]


def test_middleware_errors_use_the_envelope_only_under_v1(api):
    response = api.client.post("/v1/audio/speech", json={"input": TEXT, "voice": "Smoke Voice"})
    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "Authentication required",
            "type": "authentication_error",
            "param": None,
            "code": "invalid_api_key",
        }
    }
    assert response.headers["www-authenticate"].startswith("Bearer")

    response = api.client.get("/v1/nope", headers=bearer(api.key))
    assert response.status_code == 403
    assert response.json()["error"]["type"] == "permission_error"

    response = api.client.get("/profiles")
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


def test_models_and_voices_lists(api):
    response = api.client.get("/v1/models", headers=bearer(api.key))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "list"
    by_id = {entry["id"]: entry for entry in body["data"]}
    assert by_id["kokoro"]["kind"] == "tts"
    assert by_id["kokoro"]["engine"] == "kokoro"
    assert by_id["whisper-turbo"]["kind"] == "stt"
    assert {"downloaded", "loaded", "languages", "supports_instruct"} <= set(by_id["kokoro"])
    assert "wav" in body["formats"]

    response = api.client.get("/v1/voices", headers=bearer(api.key))
    assert response.status_code == 200
    voices = response.json()["data"]
    assert [voice["name"] for voice in voices] == ["Smoke Voice"]
    assert voices[0]["id"] == api.profile_id
    assert voices[0]["engine"] == "kokoro"


def test_transcriptions_json_text_and_errors(api, monkeypatch):
    calls = []

    async def fake_transcribe(file, language, model):
        calls.append((file.filename, language, model))
        return "hello there", 1.5

    monkeypatch.setattr("backend.routes.openai_compat.transcribe_upload", fake_transcribe)
    files = {"file": ("clip.wav", b"RIFF....WAVEfmt ", "audio/wav")}

    response = api.client.post(
        "/v1/audio/transcriptions", files=files, data={"model": "whisper-1"}, headers=bearer(api.key)
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"text": "hello there"}
    assert calls[-1] == ("clip.wav", None, None)

    response = api.client.post(
        "/v1/audio/transcriptions",
        files=files,
        data={"model": "whisper-turbo", "language": "en", "response_format": "text"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200
    assert response.text == "hello there"
    assert calls[-1] == ("clip.wav", "en", "turbo")

    response = api.client.post(
        "/v1/audio/transcriptions", files=files, data={"response_format": "srt"}, headers=bearer(api.key)
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_format"

    response = api.client.post(
        "/v1/audio/transcriptions", files=files, data={"model": "whisper-huge"}, headers=bearer(api.key)
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
