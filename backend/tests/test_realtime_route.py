"""``/v1/realtime/transcription`` end to end with a fake Whisper (needs the project venv: the route imports torch)."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import lifecycle
from backend.auth.install import install_security
from backend.auth.settings import SecuritySettings
from backend.routes import realtime
from backend.services import realtime_stt as rt

RATE = 16000


def tone(seconds: float, amplitude: float = 0.3) -> bytes:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()


def silence(seconds: float) -> bytes:
    rng = np.random.default_rng(1)
    return (rng.standard_normal(int(seconds * RATE)) * 30).astype("<i2").tobytes()


def append(audio: bytes) -> dict:
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode()}


UPDATE = {
    "type": "transcription_session.update",
    "session": {
        "input_sample_rate": RATE,
        "input_audio_transcription": {"model": "whisper-1", "language": "en"},
        "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
        "partial_interval_ms": 500,
    },
}


@pytest.fixture
def api(tmp_path, monkeypatch):
    calls: list[tuple[str, float]] = []

    async def fake_transcribe(audio, language, model):
        seconds = len(audio) / RATE
        calls.append(("call", round(seconds, 2)))
        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
        return " ".join(words[: max(1, int(seconds * 2))])

    monkeypatch.setattr(realtime, "transcribe_pcm", fake_transcribe)
    monkeypatch.setattr(realtime, "check_model", lambda model: None)
    monkeypatch.setattr(realtime, "IDLE_TIMEOUT_S", 5.0)
    realtime._sessions_by_key.clear()
    lifecycle.reset()

    settings = SecuritySettings.from_env(
        frontend_dir=None,
        environ={
            "VOICEBOX_API_KEY_FILE": str(tmp_path / "api_key"),
            "VOICEBOX_API_KEYS_JSON": str(tmp_path / "api_keys.json"),
        },
    )
    app = FastAPI()
    runtime = install_security(app, settings)
    app.include_router(realtime.router)
    runtime.startup()
    _record, key = runtime.keystore.create("app", "client", {"max_realtime_sessions": 1})
    client = TestClient(app, raise_server_exceptions=False)
    yield SimpleNamespace(client=client, key=key, runtime=runtime, calls=calls)
    lifecycle.reset()
    realtime._sessions_by_key.clear()


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def collect(socket, until: str, limit: int = 40) -> list[dict]:
    events = []
    for _ in range(limit):
        event = socket.receive_json()
        events.append(event)
        if event["type"] == until:
            return events
    raise AssertionError(f"no {until} event; got {[e['type'] for e in events]}")


def test_server_vad_session_streams_deltas_and_a_final(api):
    with api.client.websocket_connect("/v1/realtime/transcription", headers=bearer(api.key)) as socket:
        created = socket.receive_json()
        assert created["type"] == "transcription_session.created"
        assert created["session"]["input_sample_rate"] == 24000
        socket.send_json(UPDATE)
        updated = socket.receive_json()
        assert updated["type"] == "transcription_session.updated"
        assert updated["session"]["input_sample_rate"] == RATE
        assert updated["session"]["turn_detection"]["silence_duration_ms"] == 500

        socket.send_json(append(silence(0.4)))
        for _ in range(4):
            socket.send_json(append(tone(0.5)))
        socket.send_json(append(silence(0.8)))
        events = collect(socket, rt.COMPLETED_EVENT)

    types = [e["type"] for e in events]
    assert types[0] == "input_audio_buffer.speech_started"
    assert "input_audio_buffer.speech_stopped" in types
    assert "input_audio_buffer.committed" in types
    assert types[-1] == rt.COMPLETED_EVENT
    item_id = events[0]["item_id"]
    assert all(e.get("item_id", item_id) == item_id for e in events)
    deltas = [e["delta"] for e in events if e["type"] == rt.DELTA_EVENT]
    completed = events[-1]["transcript"]
    assert completed.startswith("alpha bravo")
    assert "".join(deltas) == completed  # deltas are append-only and add up to the final
    assert any(kind == "call" for kind, _ in api.calls)


def test_manual_commit_binary_frames_and_protocol_errors(api):
    with api.client.websocket_connect("/v1/realtime/transcription", headers=bearer(api.key)) as socket:
        socket.receive_json()
        socket.send_json({"type": "session.update", "session": {"input_sample_rate": RATE, "turn_detection": None}})
        assert socket.receive_json()["session"]["turn_detection"] is None

        socket.send_json({"type": "input_audio_buffer.commit"})
        empty = socket.receive_json()
        assert empty["type"] == "error"
        assert empty["error"]["code"] == "input_audio_buffer_commit_empty"

        socket.send_text("not json")
        assert socket.receive_json()["error"]["code"] == "invalid_json"
        socket.send_json({"type": "response.create"})
        assert socket.receive_json()["error"]["code"] == "unknown_event"
        socket.send_json({"type": "input_audio_buffer.append", "audio": "@@@"})
        assert socket.receive_json()["error"]["code"] == "invalid_audio"

        socket.send_bytes(tone(1.0))  # raw PCM16 frames are audio too
        socket.send_json({"type": "input_audio_buffer.commit"})
        events = collect(socket, rt.COMPLETED_EVENT)
        assert events[0]["type"] == "input_audio_buffer.committed"
        assert events[-1]["transcript"] == "alpha bravo"

        socket.send_json(append(tone(0.5)))
        socket.send_json({"type": "input_audio_buffer.clear"})
        assert socket.receive_json()["type"] == "input_audio_buffer.cleared"


def test_session_caps_and_oversized_messages(api):
    with api.client.websocket_connect("/v1/realtime/transcription", headers=bearer(api.key)) as first:
        first.receive_json()
        with api.client.websocket_connect("/v1/realtime/transcription", headers=bearer(api.key)) as second:
            refused = second.receive_json()
            assert refused["error"]["code"] == "session_limit_reached"
            with pytest.raises(WebSocketDisconnect) as closed:
                second.receive_json()
            assert closed.value.code == realtime.CLOSE_TRY_LATER
        assert realtime.open_sessions() == 1

        first.send_text("x" * (realtime.MAX_MESSAGE_BYTES + 1))
        assert first.receive_json()["error"]["code"] == "message_too_large"
        with pytest.raises(WebSocketDisconnect) as closed:
            first.receive_json()
        assert closed.value.code == realtime.CLOSE_POLICY
    assert realtime.open_sessions() == 0


def test_model_and_drain_refusals(api, monkeypatch):
    def refuse(model):
        raise rt.ProtocolError("model_not_downloaded", "Whisper model turbo is not downloaded")

    monkeypatch.setattr(realtime, "check_model", refuse)
    with api.client.websocket_connect("/v1/realtime/transcription", headers=bearer(api.key)) as socket:
        assert socket.receive_json()["error"]["code"] == "model_not_downloaded"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == realtime.CLOSE_POLICY

    monkeypatch.setattr(realtime, "check_model", lambda model: None)
    lifecycle.begin_drain("test")
    with api.client.websocket_connect("/v1/realtime/transcription", headers=bearer(api.key)) as socket:
        assert socket.receive_json()["error"]["code"] == "server_shutting_down"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == realtime.CLOSE_RESTART


def test_env_caps():
    assert realtime.max_sessions({}) == realtime.DEFAULT_MAX_SESSIONS
    assert realtime.max_sessions({"VOICEBOX_REALTIME_MAX_SESSIONS": "9"}) == 9
    assert realtime.max_sessions({"VOICEBOX_REALTIME_MAX_SESSIONS": "0"}) == 1
    assert realtime.max_sessions({"VOICEBOX_REALTIME_MAX_SESSIONS": "x"}) == realtime.DEFAULT_MAX_SESSIONS
    assert realtime.max_session_s({"VOICEBOX_REALTIME_MAX_SESSION_S": "120"}) == 120.0
    assert realtime.max_session_s({"VOICEBOX_REALTIME_MAX_SESSION_S": "1"}) == 10.0
