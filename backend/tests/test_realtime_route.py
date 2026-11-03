"""``/v1/realtime`` and ``/v1/realtime/transcription`` end to end with a fake Whisper.

Needs the project venv: the route imports torch.
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
from backend.utils.encode import linear_to_ulaw

RATE = 16000
GA_URL = "/v1/realtime?intent=transcription&model=whisper-base"
VOICEBOX_URL = "/v1/realtime/transcription"


def tone(seconds: float, amplitude: float = 0.3, rate: int = RATE) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()


def silence(seconds: float, rate: int = RATE) -> bytes:
    rng = np.random.default_rng(1)
    return (rng.standard_normal(int(seconds * rate)) * 30).astype("<i2").tobytes()


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
MANUAL = {"type": "session.update", "session": {"input_sample_rate": RATE, "turn_detection": None}}
# The first frame livekit-plugins-openai 1.8.8 sends, as captured on the wire.
PLUGIN_UPDATE = {
    "type": "session.update",
    "session": {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"rate": 24000, "type": "audio/pcm"},
                "transcription": {"language": "en", "model": "whisper-base"},
                "turn_detection": {
                    "type": "server_vad",
                    "prefix_padding_ms": 600,
                    "silence_duration_ms": 350,
                    "threshold": 0.5,
                },
            }
        },
    },
}


def ga_update(**audio_input) -> dict:
    return {"type": "session.update", "session": {"type": "transcription", "audio": {"input": audio_input}}}


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
    monkeypatch.setattr(realtime, "idle_timeout_s", lambda: 5.0)
    monkeypatch.setattr(realtime, "STOP_POLL_S", 0.05)
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


def until_closed(socket, limit: int = 40) -> tuple[list[dict], int, str]:
    """Every event up to the server's close, and the close code and reason."""
    events = []
    for _ in range(limit):
        try:
            events.append(socket.receive_json())
        except WebSocketDisconnect as closed:
            return events, closed.code, closed.reason
    raise AssertionError(f"no close; got {[e['type'] for e in events]}")


def without_deltas(events: list[dict]) -> list[str]:
    return [e["type"] for e in events if e["type"] != rt.DELTA_EVENT]


def test_server_vad_session_streams_deltas_and_a_final(api):
    with api.client.websocket_connect(VOICEBOX_URL, headers=bearer(api.key)) as socket:
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
    with api.client.websocket_connect(VOICEBOX_URL, headers=bearer(api.key)) as socket:
        socket.receive_json()
        socket.send_json(MANUAL)
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
    with api.client.websocket_connect(VOICEBOX_URL, headers=bearer(api.key)) as first:
        first.receive_json()
        with api.client.websocket_connect(VOICEBOX_URL, headers=bearer(api.key)) as second:
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
    with api.client.websocket_connect(VOICEBOX_URL, headers=bearer(api.key)) as socket:
        assert socket.receive_json()["error"]["code"] == "model_not_downloaded"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == realtime.CLOSE_POLICY

    monkeypatch.setattr(realtime, "check_model", lambda model: None)
    lifecycle.begin_drain("test")
    with api.client.websocket_connect(VOICEBOX_URL, headers=bearer(api.key)) as socket:
        assert socket.receive_json()["error"]["code"] == "server_shutting_down"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == realtime.CLOSE_RESTART


def test_openai_dialect_speaks_the_ga_protocol(api, monkeypatch):
    checked: list[str | None] = []
    monkeypatch.setattr(realtime, "check_model", checked.append)
    with api.client.websocket_connect(GA_URL, headers=bearer(api.key)) as socket:
        created = socket.receive_json()
        assert created["type"] == "session.created"
        session = created["session"]
        assert session["id"].startswith("sess_")
        assert (session["object"], session["type"]) == ("realtime.transcription_session", "transcription")
        assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert session["audio"]["input"]["transcription"]["model"] == "whisper-base"  # from the URL
        assert checked == ["whisper-base"]

        socket.send_json(PLUGIN_UPDATE)
        updated = socket.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["session"]["id"] == session["id"]
        assert updated["session"]["audio"]["input"]["turn_detection"] == {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 600,
            "silence_duration_ms": 350,
        }
        assert checked == ["whisper-base"]  # the same model is not checked again

        events = [created, updated]
        for _ in range(2):
            # 24 kHz in 50 ms appends, like the plugin.
            audio = silence(0.4, 24000) + tone(1.5, rate=24000) + silence(0.6, 24000)
            for i in range(0, len(audio), 2400):
                socket.send_json(append(audio[i : i + 2400]))
            events += collect(socket, rt.COMPLETED_EVENT)

    assert len({e["event_id"] for e in events}) == len(events)
    assert all(e["event_id"].startswith("event_") for e in events)
    committed = [e for e in events if e["type"] == "input_audio_buffer.committed"]
    assert committed[0]["previous_item_id"] is None
    assert committed[1]["previous_item_id"] == committed[0]["item_id"]
    completed = [e for e in events if e["type"] == rt.COMPLETED_EVENT]
    for final in completed:
        assert final["content_index"] == 0
        assert final["usage"]["type"] == "duration"
        assert 1.5 <= final["usage"]["seconds"] <= 2.5  # the tone plus padding and tail
        deltas = [e for e in events if e["type"] == rt.DELTA_EVENT and e["item_id"] == final["item_id"]]
        assert all(e["content_index"] == 0 for e in deltas)
        assert "".join(e["delta"] for e in deltas) == final["transcript"]


def test_openai_dialect_g711_binary_frames(api):
    with api.client.websocket_connect(GA_URL, headers=bearer(api.key)) as socket:
        socket.receive_json()
        socket.send_json(ga_update(format={"type": "audio/pcmu"}, turn_detection=None))
        updated = socket.receive_json()
        assert updated["session"]["audio"]["input"]["format"] == {"type": "audio/pcmu"}
        samples = np.frombuffer(tone(1.0, rate=8000), dtype="<i2")
        ulaw = linear_to_ulaw(samples)
        for i in range(0, len(ulaw), 160):  # 20 ms telephony frames
            socket.send_bytes(ulaw[i : i + 160])
        socket.send_json({"type": "input_audio_buffer.commit"})
        events = collect(socket, rt.COMPLETED_EVENT)
    assert events[0]["type"] == "input_audio_buffer.committed"
    assert events[-1]["type"] == rt.COMPLETED_EVENT
    # Whisper got the second at 16 kHz, less the resampler's lookahead and a partial frame.
    assert 0.95 <= api.calls[-1][1] <= 1.0


def test_openai_dialect_protocol_errors_keep_the_socket_open(api):
    with api.client.websocket_connect(GA_URL, headers=bearer(api.key)) as socket:
        socket.receive_json()
        socket.send_json({**ga_update(turn_detection={"type": "semantic_vad"}), "event_id": "evt_1"})
        error = socket.receive_json()
        assert error["type"] == "error"
        assert error["event_id"].startswith("event_")
        assert error["error"]["code"] == "unsupported_turn_detection"
        assert error["error"]["event_id"] == "evt_1"  # names the client event it answers
        for message, code in (
            ({"type": "session.update", "session": {"type": "realtime"}}, "unsupported_session_type"),
            (ga_update(format={"type": "audio/opus"}), "unsupported_audio_format"),
            (ga_update(transcription={"prompt": "p" * (rt.MAX_PROMPT_CHARS + 1)}), "invalid_session"),
            ({"type": "input_audio_buffer.commit"}, "input_audio_buffer_commit_empty"),
            ({"type": "response.create"}, "unknown_event"),
        ):
            socket.send_json(message)
            assert socket.receive_json()["error"]["code"] == code
        socket.send_text("{")
        assert socket.receive_json()["error"]["code"] == "invalid_json"
        socket.send_json(PLUGIN_UPDATE)
        assert socket.receive_json()["type"] == "session.updated"


@pytest.mark.parametrize("url", ["/v1/realtime", "/v1/realtime?intent=conversation&model=whisper-1"])
def test_openai_dialect_refuses_other_intents(api, url):
    with api.client.websocket_connect(url, headers=bearer(api.key)) as socket:
        error = socket.receive_json()
        assert (error["type"], error["error"]["code"]) == ("error", "unsupported_intent")
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert (closed.value.code, closed.value.reason) == (realtime.CLOSE_POLICY, "unsupported_intent")
    assert realtime.open_sessions() == 0


def test_openai_dialect_announces_only_closes_a_retry_cannot_fix(api, monkeypatch):
    def refuse(model):
        raise rt.ProtocolError("model_not_found", f"Unknown model '{model}'")

    monkeypatch.setattr(realtime, "check_model", refuse)
    with api.client.websocket_connect(GA_URL, headers=bearer(api.key)) as socket:
        error = socket.receive_json()
        assert (error["error"]["code"], error["event_id"][:6]) == ("model_not_found", "event_")
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert (closed.value.code, closed.value.reason) == (realtime.CLOSE_POLICY, "model_not_found")

    monkeypatch.setattr(realtime, "check_model", lambda model: None)
    with api.client.websocket_connect(GA_URL, headers=bearer(api.key)) as first:
        first.receive_json()
        with api.client.websocket_connect(GA_URL, headers=bearer(api.key)) as second:
            assert until_closed(second) == ([], realtime.CLOSE_TRY_LATER, "session_limit_reached")
        first.send_text("x" * (realtime.MAX_MESSAGE_BYTES + 1))
        events, code, reason = until_closed(first)
        assert [e["error"]["code"] for e in events] == ["message_too_large"]
        assert (code, reason) == (realtime.CLOSE_POLICY, "message_too_large")

    lifecycle.begin_drain("test")
    with api.client.websocket_connect(GA_URL, headers=bearer(api.key)) as socket:
        assert until_closed(socket) == ([], realtime.CLOSE_RESTART, "server_shutting_down")


@pytest.mark.parametrize("url", [GA_URL, VOICEBOX_URL])
def test_finals_are_delivered_before_an_idle_close(api, monkeypatch, url):
    monkeypatch.setattr(realtime, "idle_timeout_s", lambda: 0.5)
    with api.client.websocket_connect(url, headers=bearer(api.key)) as socket:
        socket.receive_json()
        socket.send_json(MANUAL)
        socket.receive_json()
        socket.send_json(append(tone(1.0)))  # never committed by the client
        events, code, reason = until_closed(socket)
    expected = ["input_audio_buffer.committed", rt.COMPLETED_EVENT]
    if url == VOICEBOX_URL:
        expected.append("error")  # this dialect still announces every close
        assert events[-1]["error"]["code"] == "idle_timeout"
    assert without_deltas(events) == expected
    assert [e["transcript"] for e in events if e["type"] == rt.COMPLETED_EVENT] == ["alpha bravo"]
    assert (code, reason) == (realtime.CLOSE_NORMAL, "idle_timeout")
    assert realtime.open_sessions() == 0


def test_finals_are_delivered_before_the_session_cap(api, monkeypatch):
    monkeypatch.setattr(realtime, "max_session_s", lambda: 1.0)
    with api.client.websocket_connect(GA_URL, headers=bearer(api.key)) as socket:
        socket.receive_json()
        socket.send_json(ga_update(format={"type": "audio/pcm", "rate": RATE}))
        socket.receive_json()
        socket.send_json(append(silence(0.4)))
        socket.send_json(append(tone(1.0)))  # still speaking when the cap hits
        events, code, reason = until_closed(socket)
    assert without_deltas(events) == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        rt.COMPLETED_EVENT,
    ]
    assert (code, reason) == (realtime.CLOSE_NORMAL, "session_too_long")


@pytest.mark.parametrize("url", [GA_URL, VOICEBOX_URL])
def test_a_stopping_server_delivers_finals_then_closes(api, url):
    with api.client.websocket_connect(url, headers=bearer(api.key)) as socket:
        socket.receive_json()
        socket.send_json(MANUAL)
        socket.receive_json()
        socket.send_json(append(tone(1.0)))
        socket.send_json(MANUAL)
        socket.receive_json()  # the audio before this update was processed
        lifecycle.begin_stopping("test", 5.0)
        events, code, reason = until_closed(socket)
    expected = ["input_audio_buffer.committed", rt.COMPLETED_EVENT]
    if url == VOICEBOX_URL:
        expected.append("error")
        assert events[-1]["error"]["code"] == "server_shutting_down"
    assert without_deltas(events) == expected
    assert (code, reason) == (realtime.CLOSE_RESTART, "server_shutting_down")


def test_finals_that_outlast_the_flush_are_reported_failed(api, monkeypatch):
    async def slow(audio, language, model):
        await asyncio.sleep(2.0)
        return "too late"

    monkeypatch.setattr(realtime, "transcribe_pcm", slow)
    monkeypatch.setattr(realtime, "FLUSH_TIMEOUT_S", 0.2)
    monkeypatch.setattr(realtime, "idle_timeout_s", lambda: 0.3)
    started = time.monotonic()
    with api.client.websocket_connect(GA_URL, headers=bearer(api.key)) as socket:
        socket.receive_json()
        socket.send_json({"type": "session.update", "session": {**MANUAL["session"], "partial_interval_ms": 10000}})
        socket.receive_json()
        socket.send_json(append(tone(1.0)))
        events, code, reason = until_closed(socket)
    assert [e["type"] for e in events] == ["input_audio_buffer.committed", rt.FAILED_EVENT]
    assert events[1]["error"]["code"] == "session_closed"
    assert (code, reason) == (realtime.CLOSE_NORMAL, "idle_timeout")
    assert time.monotonic() - started < 1.8  # the slow final was abandoned, not awaited


def test_a_client_disconnect_cancels_at_once(api, monkeypatch):
    async def slow(audio, language, model):
        await asyncio.sleep(3.0)
        return "never sent"

    monkeypatch.setattr(realtime, "transcribe_pcm", slow)
    with api.client.websocket_connect(GA_URL, headers=bearer(api.key)) as socket:
        socket.receive_json()
        socket.send_json(MANUAL)
        socket.receive_json()
        socket.send_json(append(tone(1.0)))
        socket.send_json({"type": "input_audio_buffer.commit"})
        assert socket.receive_json()["type"] == "input_audio_buffer.committed"
        started = time.monotonic()
    assert time.monotonic() - started < 1.5
    assert realtime.open_sessions() == 0


def test_utterances_past_the_backlog_cap_fail_at_once(api, monkeypatch):
    """A client that sends audio faster than Whisper transcribes it cannot queue audio without bound."""
    release = threading.Event()

    async def held(audio, language, model):
        await asyncio.to_thread(release.wait, 10)
        return "held"

    monkeypatch.setattr(realtime, "transcribe_pcm", held)
    manual = {"type": "session.update", "session": {**MANUAL["session"], "partial_interval_ms": 10000}}
    try:
        with api.client.websocket_connect(GA_URL, headers=bearer(api.key)) as socket:
            socket.receive_json()
            socket.send_json(manual)
            socket.receive_json()
            for _ in range(rt.MAX_PENDING_FINALS):  # the first one holds Whisper, the rest wait
                socket.send_bytes(tone(0.5))
                socket.send_json({"type": "input_audio_buffer.commit"})
                assert socket.receive_json()["type"] == "input_audio_buffer.committed"
            socket.send_bytes(tone(0.5))
            socket.send_json({"type": "input_audio_buffer.commit"})
            committed, failed = socket.receive_json(), socket.receive_json()
            assert committed["type"] == "input_audio_buffer.committed"
            assert failed["type"] == rt.FAILED_EVENT
            assert (failed["item_id"], failed["error"]["code"]) == (committed["item_id"], "whisper_busy")

            release.set()
            completed = []
            while len(completed) < rt.MAX_PENDING_FINALS:
                event = socket.receive_json()
                if event["type"] != rt.DELTA_EVENT:
                    completed.append(event["type"])
            assert completed == [rt.COMPLETED_EVENT] * rt.MAX_PENDING_FINALS
    finally:
        release.set()


def test_env_caps():
    assert realtime.max_sessions({}) == realtime.DEFAULT_MAX_SESSIONS
    assert realtime.max_sessions({"VOICEBOX_REALTIME_MAX_SESSIONS": "9"}) == 9
    assert realtime.max_sessions({"VOICEBOX_REALTIME_MAX_SESSIONS": "0"}) == 1
    assert realtime.max_sessions({"VOICEBOX_REALTIME_MAX_SESSIONS": "x"}) == realtime.DEFAULT_MAX_SESSIONS
    assert realtime.max_session_s({"VOICEBOX_REALTIME_MAX_SESSION_S": "120"}) == 120.0
    assert realtime.max_session_s({"VOICEBOX_REALTIME_MAX_SESSION_S": "1"}) == 10.0
    assert realtime.idle_timeout_s({}) == 60.0
    assert realtime.idle_timeout_s({"VOICEBOX_REALTIME_IDLE_S": "300"}) == 300.0
    assert realtime.idle_timeout_s({"VOICEBOX_REALTIME_IDLE_S": "10"}) == 15.0  # LiveKit's default retry timeout
    assert realtime.idle_timeout_s({"VOICEBOX_REALTIME_IDLE_S": "soon"}) == 60.0


async def test_reader_backpressure_preserves_audio_and_disconnect_order(monkeypatch):
    monkeypatch.setattr(realtime, "MAX_INBOUND_MESSAGES", 2)
    messages = [
        *({"type": "websocket.receive", "bytes": bytes([i])} for i in range(3)),
        {"type": "websocket.disconnect", "code": 1000},
    ]
    socket = SimpleNamespace(receive=AsyncMock(side_effect=messages))
    connection = realtime._Connection(socket, rt.TranscriptionSession(rt.SessionConfig()))
    reader = asyncio.create_task(connection.reader())
    try:
        await asyncio.sleep(0)
        assert connection.inbound.qsize() == 2
        assert socket.receive.await_count == 3  # one read is held until a queue slot opens
        assert not reader.done()

        delivered = [await asyncio.wait_for(connection.inbound.get(), timeout=1) for _ in messages]
        await asyncio.wait_for(reader, timeout=1)
        assert delivered == messages
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


@pytest.mark.parametrize("failed_receive", [False, True])
async def test_reader_cancellation_does_not_wait_for_full_queue(monkeypatch, failed_receive):
    monkeypatch.setattr(realtime, "MAX_INBOUND_MESSAGES", 2)
    audio = {"type": "websocket.receive", "bytes": b"\x00\x00"}
    last = RuntimeError("connection lost") if failed_receive else audio
    socket = SimpleNamespace(receive=AsyncMock(side_effect=[audio, audio, last]))
    connection = realtime._Connection(socket, rt.TranscriptionSession(rt.SessionConfig()))
    reader = asyncio.create_task(connection.reader())
    try:
        await asyncio.sleep(0)
        assert connection.inbound.full()
        assert socket.receive.await_count == 3
        assert not reader.done()

        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(reader, timeout=1)
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
