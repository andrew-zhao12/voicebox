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


GENERATE_PROMPTS: list[dict] = []


class FakeBackend:
    async def generate(self, text, voice_prompt, language="en", seed=None, instruct=None):
        GENERATE_PROMPTS.append(dict(voice_prompt))
        return np.full(len(text), 0.25, dtype=np.float32), SR


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def api(tmp_path, monkeypatch):
    previous_data_dir = config.get_data_dir()
    config.set_data_dir(tmp_path)
    database.init_db()
    seen_prompts: list[tuple[str, dict | None]] = []

    async def prepare_engine(engine, model_size, profile_id, db, *, on_loading=None, voice_prompt=None):
        seen_prompts.append((profile_id, voice_prompt))
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
        yield SimpleNamespace(
            client=client,
            admin=admin_key,
            key=client_key,
            profile_id=created.json()["id"],
            runtime=runtime,
            prompts=seen_prompts,
        )
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


def test_speech_ulaw_8000_is_telephony_audio(api):
    response = api.client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": TEXT, "voice": "Smoke Voice", "response_format": "ulaw_8000"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/basic")
    assert response.headers["x-voicebox-sample-rate"] == "8000"
    # The fake engine renders one sample per character at 24 kHz.
    assert abs(len(response.content) - len(TEXT) / 3) <= 1


def test_speech_passes_chunk_size_knobs_to_the_stream_request(api, monkeypatch):
    from backend.routes import openai_compat

    seen = {}
    original = openai_compat.generation_service.open_stream

    async def recording_open_stream(stream_request, *args, **kwargs):
        seen["request"] = stream_request
        return await original(stream_request, *args, **kwargs)

    monkeypatch.setattr(openai_compat.generation_service, "open_stream", recording_open_stream)
    response = api.client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": TEXT,
            "voice": "Smoke Voice",
            "response_format": "pcm",
            "max_chunk_chars": 200,
            "first_chunk_chars": 60,
        },
        headers=bearer(api.key),
    )
    assert response.status_code == 200, response.text
    assert seen["request"].max_chunk_chars == 200
    assert seen["request"].first_chunk_chars == 60


def test_speech_rejects_chunk_caps_below_the_minimum(api):
    response = api.client.post(
        "/v1/audio/speech",
        json={"input": TEXT, "voice": "Smoke Voice", "first_chunk_chars": 5},
        headers=bearer(api.key),
    )
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "invalid_value"
    assert error["param"] == "first_chunk_chars"


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


@pytest.mark.parametrize(
    ("model", "voice"),
    [
        ("luxtts", "Smoke Voice"),  # a Kokoro preset profile on a cloning engine
        ("kokoro", "Aiden"),  # a qwen_custom_voice preset on Kokoro
    ],
)
def test_a_voice_made_for_another_engine_is_a_voice_error(api, model, voice):
    """Clients key their voice fallback on ``param``; a bare 400 reads as a server failure."""
    response = api.client.post(
        "/v1/audio/speech",
        json={"model": model, "input": TEXT, "voice": voice},
        headers=bearer(api.key),
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "voice_engine_mismatch"
    assert error["param"] == "voice"
    assert "engine" in error["message"]


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
        "/v1/audio/speech", json={"input": TEXT, "voice": "Smoke Voice", "speed": 5}, headers=bearer(api.key)
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "speed"
    assert error["code"] == "invalid_value"

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
    assert [voice["name"] for voice in voices if voice["kind"] == "profile"] == ["Smoke Voice"]
    assert voices[0]["id"] == api.profile_id
    assert voices[0]["engine"] == "kokoro"
    assert len(voices) > 50  # the built-in preset voices follow the profiles


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
        "/v1/audio/transcriptions", files=files, data={"response_format": "diarized"}, headers=bearer(api.key)
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_format"

    response = api.client.post(
        "/v1/audio/transcriptions", files=files, data={"model": "whisper-huge"}, headers=bearer(api.key)
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def sample_wav(seconds: float = 2.5) -> bytes:
    t = np.arange(int(SR * seconds)) / SR
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_preset_voices_work_without_a_profile(api):
    response = api.client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": TEXT, "voice": "af_heart", "response_format": "pcm"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-voicebox-engine"] == "kokoro"
    assert response.headers["x-voicebox-voice"] == "kokoro:af_heart"
    assert len(response.content) == len(TEXT) * 2
    assert api.prompts[-1] == (
        "kokoro:af_heart",
        {"voice_type": "preset", "preset_engine": "kokoro", "preset_voice_id": "af_heart"},
    )

    # Qualified and case-insensitive ids, with the alias model picking the preset's engine.
    response = api.client.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "input": TEXT, "voice": "Kokoro:AF_HEART", "response_format": "pcm"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-voicebox-engine"] == "kokoro"

    # A preset of one engine requested with another engine is a clear 400, not an unknown voice.
    response = api.client.post(
        "/v1/audio/speech", json={"model": "kokoro", "input": TEXT, "voice": "Ryan"}, headers=bearer(api.key)
    )
    assert response.status_code == 400
    assert "qwen_custom_voice" in response.json()["error"]["message"]

    # Profiles take precedence over presets and stock names; the profile row's prompt path is used.
    response = api.client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": TEXT, "voice": "smoke voice", "response_format": "pcm"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200
    assert api.prompts[-1] == (api.profile_id, None)


def test_voices_list_has_profiles_and_presets(api):
    response = api.client.get("/v1/voices", headers=bearer(api.key))
    assert response.status_code == 200
    data = response.json()["data"]
    kinds = {entry["kind"] for entry in data}
    assert kinds == {"profile", "preset"}
    profile = next(entry for entry in data if entry["kind"] == "profile")
    assert profile["name"] == "Smoke Voice"
    assert profile["shared"] is True
    presets = {entry["id"]: entry for entry in data if entry["kind"] == "preset"}
    assert presets["af_heart"]["engine"] == "kokoro"
    assert presets["af_heart"]["gender"] == "female"
    assert presets["Ryan"]["engine"] == "qwen_custom_voice"


def test_client_owned_voices_are_private_and_deletable_by_their_key(api, monkeypatch):
    async def fake_transcribe(path, language, model):
        return "transcribed automatically", 2.5

    monkeypatch.setattr("backend.routes.openai_compat.transcribe_file", fake_transcribe)
    _other_record, other_key = api.runtime.keystore.create("other", "client", None)

    files = [("file", ("one.wav", sample_wav(), "audio/wav")), ("file", ("two.wav", sample_wav(), "audio/wav"))]
    response = api.client.post(
        "/v1/voices",
        data={"name": "My App Voice", "text": ["First sample."], "engine": "qwen", "language": "en"},
        files=files,
        headers=bearer(api.key),
    )
    assert response.status_code == 201, response.text
    voice = response.json()
    assert voice["owner"] == "app"
    assert voice["shared"] is False
    assert voice["engine"] == "qwen"

    samples = api.client.get(f"/profiles/{voice['id']}/samples", headers=bearer(api.admin)).json()
    assert [s["reference_text"] for s in samples] == ["First sample.", "transcribed automatically"]

    # Visible to its owner and to admins, invisible to another client key.
    assert "My App Voice" in {v["name"] for v in api.client.get("/v1/voices", headers=bearer(api.key)).json()["data"]}
    assert "My App Voice" in {v["name"] for v in api.client.get("/profiles", headers=bearer(api.admin)).json()}
    other_names = {v["name"] for v in api.client.get("/v1/voices", headers=bearer(other_key)).json()["data"]}
    assert "My App Voice" not in other_names
    assert api.client.get(f"/profiles/{voice['id']}", headers=bearer(other_key)).status_code == 404
    hidden = api.client.post(
        "/v1/audio/speech", json={"input": TEXT, "voice": "My App Voice", "model": "qwen"}, headers=bearer(other_key)
    )
    assert hidden.status_code == 404
    assert hidden.json()["error"]["code"] == "voice_not_found"
    assert (
        api.client.post(
            "/generate/stream",
            json={"profile_id": voice["id"], "text": TEXT, "engine": "qwen"},
            headers=bearer(other_key),
        ).status_code
        == 404
    )

    # The owner streams with it (fake backend), a duplicate name is 409, a shared voice cannot be deleted by a client.
    response = api.client.post(
        "/v1/audio/speech",
        json={"input": TEXT, "voice": "My App Voice", "model": "qwen", "response_format": "pcm"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200, response.text
    duplicate = api.client.post(
        "/v1/voices", data={"name": "My App Voice", "text": ["x"]}, files=files[:1], headers=bearer(api.key)
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "voice_exists"
    shared = api.client.delete("/v1/voices/Smoke Voice", headers=bearer(api.key))
    assert shared.status_code == 403
    assert shared.json()["error"]["code"] == "voice_not_owned"
    assert api.client.delete("/v1/voices/My App Voice", headers=bearer(other_key)).status_code == 404
    assert api.client.delete("/v1/voices/af_heart", headers=bearer(api.key)).status_code == 404

    deleted = api.client.delete(f"/v1/voices/{voice['id']}", headers=bearer(api.key))
    assert deleted.status_code == 200
    assert deleted.json() == {"id": voice["id"], "object": "voice", "deleted": True}
    assert api.client.get(f"/profiles/{voice['id']}", headers=bearer(api.admin)).status_code == 404


def test_voice_creation_limits_and_validation(api, monkeypatch):
    _record, capped_key = api.runtime.keystore.create("capped", "client", {"max_voices": 1})
    files = [("file", ("one.wav", sample_wav(), "audio/wav"))]
    first = api.client.post(
        "/v1/voices", data={"name": "Only One", "text": ["Hi."]}, files=files, headers=bearer(capped_key)
    )
    assert first.status_code == 201, first.text
    second = api.client.post(
        "/v1/voices", data={"name": "Two", "text": ["Hi."]}, files=files, headers=bearer(capped_key)
    )
    assert second.status_code == 403
    assert second.json()["error"]["code"] == "voice_limit_reached"

    bad_engine = api.client.post(
        "/v1/voices",
        data={"name": "Preset?", "text": ["Hi."], "engine": "kokoro"},
        files=files,
        headers=bearer(api.key),
    )
    assert bad_engine.status_code == 400
    assert bad_engine.json()["error"]["param"] == "engine"

    too_short = [("file", ("short.wav", sample_wav(0.5), "audio/wav"))]
    short = api.client.post(
        "/v1/voices", data={"name": "Short", "text": ["Hi."]}, files=too_short, headers=bearer(api.key)
    )
    assert short.status_code == 400
    assert "too short" in short.json()["error"]["message"]
    assert "Short" not in {v["name"] for v in api.client.get("/profiles", headers=bearer(api.admin)).json()}

    # Admin keys are unlimited and may delete any voice through /v1 as well.
    admin_made = api.client.post(
        "/v1/voices", data={"name": "Admin Voice", "text": ["Hi."]}, files=files, headers=bearer(api.admin)
    )
    assert admin_made.status_code == 201
    assert api.client.delete("/v1/voices/Only One", headers=bearer(api.admin)).status_code == 200


def test_speed_is_native_on_kokoro_and_a_time_stretch_elsewhere(api, monkeypatch):
    stretched: list[tuple[int, float]] = []

    def fake_stretch(audio, sample_rate, speed):
        stretched.append((len(audio), speed))
        return audio[:: int(speed)] if speed >= 1 else np.repeat(audio, round(1 / speed))

    monkeypatch.setattr(generation_service, "_time_stretch", fake_stretch)

    # Kokoro takes the rate itself: the prompt carries it and nothing is stretched.
    GENERATE_PROMPTS.clear()
    response = api.client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": TEXT, "voice": "Smoke Voice", "speed": 2.0, "response_format": "pcm"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200, response.text
    assert len(response.content) == len(TEXT) * 2
    assert GENERATE_PROMPTS
    assert GENERATE_PROMPTS[-1]["speed"] == 2.0
    assert stretched == []

    # Any other engine is stretched chunk by chunk after synthesis.
    created = api.client.post(
        "/profiles",
        json={"name": "Cloned Voice", "language": "en", "default_engine": "qwen"},
        headers=bearer(api.admin),
    )
    assert created.status_code == 200, created.text
    GENERATE_PROMPTS.clear()
    response = api.client.post(
        "/v1/audio/speech",
        json={"model": "qwen", "input": TEXT, "voice": "Cloned Voice", "speed": 2.0, "response_format": "pcm"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200, response.text
    assert "speed" not in GENERATE_PROMPTS[-1]
    assert stretched
    assert all(speed == 2.0 for _, speed in stretched)
    assert len(response.content) == 2 * sum(-(-n // 2) for n, _ in stretched)

    # speed=1.0 (the default) touches nothing.
    stretched.clear()
    response = api.client.post(
        "/v1/audio/speech",
        json={"model": "qwen", "input": TEXT, "voice": "Cloned Voice", "response_format": "pcm"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200
    assert stretched == []
    assert len(response.content) == len(TEXT) * 2


def test_time_stretch_changes_duration_and_keeps_mono_float32():
    t = np.arange(SR) / SR
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    faster = generation_service._time_stretch(tone, SR, 2.0)
    slower = generation_service._time_stretch(tone, SR, 0.5)
    assert faster.ndim == 1
    assert faster.dtype == np.float32
    assert abs(len(faster) - SR / 2) < SR * 0.05
    assert abs(len(slower) - SR * 2) < SR * 0.1
    assert np.abs(faster).max() < 1.0


def test_timestamped_transcription_formats(api, monkeypatch):
    from backend.utils.subtitles import Transcript, TranscriptSegment

    detailed_calls = []

    async def fake_detailed(file, language, model):
        detailed_calls.append((file.filename, language, model))
        return Transcript(
            text="Hello world. Second part.",
            segments=(
                TranscriptSegment(id=0, start=0.0, end=1.2, text="Hello world."),
                TranscriptSegment(id=1, start=1.2, end=2.8, text="Second part."),
            ),
            language="en",
            duration=2.8,
        )

    monkeypatch.setattr("backend.routes.openai_compat.transcribe_upload_detailed", fake_detailed)
    files = {"file": ("clip.wav", b"RIFF....WAVEfmt ", "audio/wav")}

    response = api.client.post(
        "/v1/audio/transcriptions",
        files=files,
        data={"model": "whisper-turbo", "response_format": "verbose_json", "timestamp_granularities[]": "segment"},
        headers=bearer(api.key),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["duration"] == 2.8
    assert [s["text"] for s in body["segments"]] == ["Hello world.", "Second part."]
    assert body["segments"][1]["start"] == 1.2
    assert detailed_calls[-1] == ("clip.wav", None, "turbo")

    srt = api.client.post(
        "/v1/audio/transcriptions", files=files, data={"response_format": "srt"}, headers=bearer(api.key)
    )
    assert srt.status_code == 200
    assert srt.text.startswith("1\n00:00:00,000 --> 00:00:01,200\nHello world.")
    vtt = api.client.post(
        "/v1/audio/transcriptions", files=files, data={"response_format": "vtt"}, headers=bearer(api.key)
    )
    assert vtt.status_code == 200
    assert vtt.text.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.200")

    words = api.client.post(
        "/v1/audio/transcriptions",
        files=files,
        data={"response_format": "verbose_json", "timestamp_granularities[]": "word"},
        headers=bearer(api.key),
    )
    assert words.status_code == 400
    assert words.json()["error"]["param"] == "timestamp_granularities"
