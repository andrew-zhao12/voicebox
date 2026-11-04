"""Saved speech settings reach real HTTP/WebSocket routes with inference faked."""

import base64
import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from backend.database import VoiceProfile, session as database_session
from backend.routes import realtime
from backend.services import transcribe
from backend.tests.test_openai_compat import api as shared_api, bearer, sample_wav
from backend.tests.test_realtime_route import MANUAL, collect, tone
from backend.utils.subtitles import Transcript

api = shared_api


def save_client(api, **values):
    response = api.client.put("/speech/settings", headers=bearer(api.key), json=values)
    assert response.status_code == 200, response.text


def set_voice(api, value):
    response = api.client.put(
        f"/profiles/{api.profile_id}/pronunciations",
        headers=bearer(api.admin),
        json={"pronunciation_replacements": value},
    )
    assert response.status_code == 200, response.text


def speak(api, **overrides):
    response = api.client.post(
        "/v1/audio/speech",
        headers=bearer(api.key),
        json={"input": "API", "voice": api.profile_id, "model": "kokoro", "response_format": "pcm", **overrides},
    )
    assert response.status_code == 200, response.text
    # The test backend produces exactly one sample per input character.
    return len(response.content) // 2


def test_http_synthesis_uses_request_voice_then_client_dictionary_once(api):
    save_client(api, pronunciation_replacements={"API": "client words"})
    assert speak(api) == len("client words")
    set_voice(api, {"API": "A P I", "A P I": "do not expand twice"})
    assert speak(api) == len("A P I")
    assert speak(api, pronunciation_replacements={"API": "explicit request"}) == len("explicit request")
    assert speak(api, pronunciation_replacements={}) == len("API")
    set_voice(api, {})
    assert speak(api) == len("API")
    set_voice(api, None)
    assert speak(api) == len("client words")


def test_voice_update_preserves_dictionary_when_omitted_and_bundle_roundtrips(api):
    set_voice(api, {"API": "A P I"})
    response = api.client.put(
        f"/profiles/{api.profile_id}", headers=bearer(api.admin), json={"name": "Renamed", "language": "en"}
    )
    assert response.json()["pronunciation_replacements"] == {"API": "A P I"}
    exported = api.client.get(f"/profiles/{api.profile_id}/export", headers=bearer(api.admin))
    assert exported.status_code == 200
    imported = api.client.post(
        "/profiles/import",
        headers=bearer(api.admin),
        files={"file": ("voice.zip", exported.content, "application/zip")},
    )
    assert imported.status_code == 200, imported.text
    assert imported.json()["pronunciation_replacements"] == {"API": "A P I"}


def test_invalid_dictionary_does_not_destroy_existing_voice_on_bundle_replace(api):
    set_voice(api, {"API": "A P I"})
    exported = api.client.get(f"/profiles/{api.profile_id}/export", headers=bearer(api.admin))
    with zipfile.ZipFile(io.BytesIO(exported.content)) as original:
        entries = {name: original.read(name) for name in original.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["profile"]["pronunciation_replacements"] = {"": "invalid"}
    entries["manifest.json"] = json.dumps(manifest).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as broken:
        for name, content in entries.items():
            broken.writestr(name, content)
    response = api.client.post(
        "/profiles/import?on_conflict=replace",
        headers=bearer(api.admin),
        files={"file": ("bad.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 400
    profile = api.client.get(f"/profiles/{api.profile_id}", headers=bearer(api.admin))
    assert profile.json()["pronunciation_replacements"] == {"API": "A P I"}


def test_voice_dictionary_edit_requires_ownership_or_admin(api):
    path = f"/profiles/{api.profile_id}/pronunciations"
    update = {"pronunciation_replacements": {"API": "A P I"}}
    assert api.client.put(path, headers=bearer(api.key), json=update).status_code == 403
    with database_session.SessionLocal() as db:
        profile = db.get(VoiceProfile, api.profile_id)
        profile.owner_key_id = "app"
        db.commit()
    assert api.client.put(path, headers=bearer(api.key), json=update).status_code == 200
    _, other = api.runtime.keystore.create("other", "client")
    assert api.client.put(path, headers=bearer(other), json=update).status_code == 404
    assert api.client.put(path, headers=bearer(api.key), json={}).status_code == 422


@pytest.mark.parametrize("route", ["/transcribe", "/v1/audio/transcriptions"])
def test_http_transcription_defaults_and_explicit_empty_form_overrides(api, monkeypatch, route):
    calls = []

    async def recognize(path, language, model_size, **options):
        calls.append(options)
        return "recognized"

    fake = SimpleNamespace(model_size="base", is_loaded=lambda: True, transcribe=recognize)
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda: fake)
    save_client(api, stt_prompt="Names", stt_hotwords=["Acme"])
    for data, expected in (
        ({}, {"prompt": "Names", "hotwords": ["Acme"]}),
        ({"prompt": "", "hotwords": ""}, {}),
        ({"prompt": "Custom", "hotwords": "Other"}, {"prompt": "Custom", "hotwords": ["Other"]}),
    ):
        response = api.client.post(
            route,
            headers=bearer(api.key),
            files={"file": ("sample.wav", sample_wav(), "audio/wav")},
            data=data,
        )
        assert response.status_code == 200, response.text
        assert calls[-1] == expected


def test_saved_hints_fail_explicitly_for_unsupported_http_model(api):
    save_client(api, stt_prompt="Names")
    response = api.client.post(
        "/v1/audio/transcriptions",
        headers=bearer(api.key),
        files={"file": ("sample.wav", sample_wav(), "audio/wav")},
        data={"model": "nemotron-streaming-en-0.6b"},
    )
    assert response.status_code == 400
    assert "does not support prompt" in response.json()["error"]["message"]


def test_speech_socket_resolves_saved_defaults_and_explicit_clear(api):
    save_client(api, pronunciation_replacements={"API": "client words"})
    set_voice(api, {"API": "A P I", "A P I": "do not expand twice"})
    with api.client.websocket_connect("/v1/audio/speech/stream", headers=bearer(api.key)) as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({"type": "session.update", "session": {"voice": api.profile_id, "model": "kokoro"}})
        collect(ws, "session.updated")
        for index, expected in enumerate((len("A P I"), len("API"))):
            if index:
                ws.send_json({"type": "session.update", "session": {"pronunciation_replacements": {}}})
                collect(ws, "session.updated")
            ws.send_json({"type": "input_text.append", "text": "API"})
            collect(ws, "input_text.accepted")
            ws.send_json({"type": "input_text.flush", "flush_id": str(index)})
            events = collect(ws, "audio.done")
            assert events[-1]["samples"] == expected
        ws.send_json({"type": "session.close"})
        collect(ws, "session.closed")


def test_saved_pronunciation_expansion_is_charged_to_character_budget(api):
    _, limited = api.runtime.keystore.create("limited", "client", {"tts_chars": 4})
    response = api.client.put(
        "/speech/settings", headers=bearer(limited), json={"pronunciation_replacements": {"API": "longer"}}
    )
    assert response.status_code == 200
    response = api.client.post(
        "/v1/audio/speech",
        headers=bearer(limited),
        json={"input": "API", "voice": api.profile_id, "model": "kokoro", "response_format": "pcm"},
    )
    assert response.status_code == 429, response.text


def test_realtime_settings_snapshot_and_clear_are_not_reapplied(api, monkeypatch):
    calls = []

    async def recognize(path, language, model_size, **options):
        calls.append(options)
        return Transcript("recognized")

    fake = SimpleNamespace(model_size="base", is_loaded=lambda: True, transcribe_detailed=recognize)
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda: fake)
    monkeypatch.setattr(realtime, "check_model", lambda model: None)
    save_client(api, stt_prompt="Names", stt_hotwords=["Acme"])
    with api.client.websocket_connect("/v1/realtime?intent=transcription", headers=bearer(api.key)) as ws:
        settings = ws.receive_json()["session"]["audio"]["input"]["transcription"]
        assert settings["prompt"] == "Names"
        assert settings["hotwords"] == ["Acme"]
        save_client(api, stt_prompt="Changed later", stt_hotwords=["Later"])
        ws.send_json(MANUAL)
        collect(ws, "session.updated")
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(tone(0.4)).decode()})
        ws.send_json({"type": "input_audio_buffer.commit"})
        collect(ws, "conversation.item.input_audio_transcription.completed")
        assert calls[-1] == {"prompt": "Names", "hotwords": ("Acme",)}
        ws.send_json(
            {"type": "session.update", "session": {"input_audio_transcription": {"prompt": "", "hotwords": []}}}
        )
        collect(ws, "session.updated")
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(tone(0.4)).decode()})
        ws.send_json({"type": "input_audio_buffer.commit"})
        collect(ws, "conversation.item.input_audio_transcription.completed")
        assert calls[-1] == {}


def test_realtime_unsupported_saved_hints_can_be_cleared_before_audio(api, monkeypatch):
    calls = []

    async def native_inference(self, work):
        calls.append(work)
        return "recognized"

    monkeypatch.setattr(realtime, "check_model", lambda model: None)
    monkeypatch.setattr(realtime.NativeTranscriber, "supports", lambda self, model: True)
    monkeypatch.setattr(realtime.NativeTranscriber, "transcribe", native_inference)
    save_client(api, stt_prompt="Names", stt_hotwords=["Acme"])
    url = "/v1/realtime?intent=transcription&model=nemotron-streaming-en-0.6b"
    with api.client.websocket_connect(url, headers=bearer(api.key)) as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(tone(0.4)).decode()})
        error = collect(ws, "error")[-1]
        assert error["error"]["code"] == "unsupported_value"
        assert not calls
        ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "input_sample_rate": 16000,
                    "turn_detection": None,
                    "input_audio_transcription": {"prompt": "", "hotwords": []},
                },
            }
        )
        collect(ws, "session.updated")
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(tone(0.4)).decode()})
        ws.send_json({"type": "input_audio_buffer.commit"})
        collect(ws, "conversation.item.input_audio_transcription.completed")
        assert calls
        assert all(not work.prompt and work.hotwords == () for work in calls)
    saved = api.client.get("/speech/settings", headers=bearer(api.key)).json()
    assert saved["stt_prompt"] == "Names"
    assert saved["stt_hotwords"] == ["Acme"]


def test_saved_pronunciations_apply_after_personality_to_stored_generation(api, monkeypatch):
    from backend.routes import generations

    async def rewrite(data, profile):
        return "API", "personality_speak"

    monkeypatch.setattr(generations, "_rewrite_for_personality", rewrite)
    set_voice(api, {"API": "A P I", "A P I": "do not expand twice"})
    response = api.client.post(
        "/generate",
        headers=bearer(api.key),
        json={"profile_id": api.profile_id, "text": "original", "engine": "kokoro", "personality": True},
    )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "A P I"
