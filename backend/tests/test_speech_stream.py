"""Incremental speech protocol, real queue cancellation, and WebSocket lifecycle checks."""

import asyncio
import base64
import json
import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import lifecycle
from backend.auth.install import install_security
from backend.auth.principal import KeyLimits, Principal
from backend.auth.ratelimit import RateLimiter
from backend.auth.settings import SecuritySettings
from backend.observability import metrics
from backend.routes import openai_compat, speech_stream as route
from backend.services import generation, live_sessions, task_queue
from backend.services.speech_stream import MAX_PENDING_FLUSHES, MAX_TEXT_CHARS, ProtocolError, SpeechBuffer

CLIENT = Principal("speech-test", "client", "header", KeyLimits.defaults_for("client"))


class Socket:
    """In-memory ASGI WebSocket peer with controllable send backpressure."""

    def __init__(self):
        self.headers = {}
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.block_audio = False

    async def accept(self):
        pass

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, event):
        if self.block_audio and event["type"] == "audio.delta":
            await asyncio.Event().wait()
        await self.outgoing.put(event)

    async def close(self, code, reason):
        await self.outgoing.put({"type": "close", "code": code, "reason": reason})

    def send(self, kind, **payload):
        self.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps({"type": kind, **payload})})

    def disconnect(self):
        self.incoming.put_nowait({"type": "websocket.disconnect"})

    async def until(self, kind, *, flush_id=None):
        events = []
        async with asyncio.timeout(3):
            while True:
                event = await self.outgoing.get()
                events.append(event)
                if event["type"] == kind and (flush_id is None or event.get("flush_id") == flush_id):
                    return events


class FakeDb:
    def get(self, *_args):
        return None

    def query(self, *_args):
        return self

    def filter_by(self, **_kwargs):
        return self

    def first(self):
        return SimpleNamespace(effects_chain=None)

    def close(self):
        pass


class Backend:
    """A controllable engine behind the real generation queue and streaming orchestration."""

    def __init__(self):
        self.calls = []
        self.gate = asyncio.Event()
        self.gate.set()
        self.fail_after = None
        self.rate = 24000

    async def generate(self, text, voice_prompt, language, seed, instruct):
        self.calls.append(text)
        await self.gate.wait()
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("model failed")
        return np.full(4800, 0.25, dtype=np.float32), self.rate


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.fixture
async def engine(monkeypatch):
    lifecycle.reset()
    live_sessions.counts_for_kind("speech").clear()
    live_sessions.counts_for_kind("transcription").clear()
    task_queue.init_queue(force=True)
    backend = Backend()
    profile = SimpleNamespace(
        id="profile", default_engine="kokoro", effects_chain=None, personality=None, language="en"
    )

    def database():
        yield FakeDb()

    async def get_profile(*_args):
        return profile

    async def cached(*_args):
        pass

    async def prepare(*_args, **_kwargs):
        return generation.EnginePrep(backend, {}, None, None, None)

    monkeypatch.setattr(route, "get_principal", lambda: CLIENT)
    monkeypatch.setattr(route, "charge", lambda *_args: None)
    monkeypatch.setattr(generation, "charge", lambda *_args: None)
    monkeypatch.setattr(route, "get_db", database)
    monkeypatch.setattr(generation, "get_db", database)
    monkeypatch.setattr(generation, "prepare_engine", prepare)
    monkeypatch.setattr(generation, "lane_for_engine", lambda _engine: task_queue.LANE_CPU)
    monkeypatch.setattr(generation.profiles, "get_profile", get_profile)
    monkeypatch.setattr(generation.profiles, "is_visible", lambda *_args: True)
    monkeypatch.setattr(generation.profiles, "validate_profile_engine", lambda *_args: None)
    monkeypatch.setattr("backend.backends.ensure_model_cached_or_raise", cached)
    monkeypatch.setattr(openai_compat, "_resolve_voice", lambda *_args: profile)
    yield backend
    backend.gate.set()
    await task_queue.shutdown(2)
    lifecycle.reset()
    live_sessions.counts_for_kind("speech").clear()
    live_sessions.counts_for_kind("transcription").clear()


async def connect():
    socket = Socket()
    task = asyncio.create_task(route.speech_stream(socket))
    await socket.until("session.created")
    return socket, task


async def close(socket, task):
    socket.disconnect()
    await asyncio.wait_for(task, 2)
    assert live_sessions.open_sessions("speech") == 0


def submit(socket, text, flush_id):
    socket.send("input_text.append", text=text)
    socket.send("input_text.flush", flush_id=flush_id)


def test_protocol_snapshots_limits_and_duplicate_ids():
    buffer = SpeechBuffer()
    buffer.update({"voice": "first", "pronunciation_replacements": {"API": "A P I"}})
    buffer.append("API")
    flush = buffer.prepare("one")
    buffer.commit(flush)
    buffer.update({"voice": "second"})
    assert flush.request.voice == "first"
    assert flush.request.pronunciation_replacements == {"API": "A P I"}
    buffer.append("next")
    with pytest.raises(ProtocolError, match="already used"):
        buffer.prepare("one")
    with pytest.raises(ProtocolError, match="capped"):
        buffer.append("x" * MAX_TEXT_CHARS)
    for index in range(MAX_PENDING_FLUSHES - 1):
        flush = buffer.prepare(str(index))
        buffer.commit(flush)
        buffer.append("more")
    with pytest.raises(ProtocolError, match="pending"):
        buffer.prepare("last")


@pytest.mark.parametrize(
    "settings", [{"input": "override"}, {"response_format": "wav"}, {"speed": 0}, {"voice": ""}, None]
)
def test_invalid_configuration_preserves_settings(settings):
    buffer = SpeechBuffer()
    original = dict(buffer.settings)
    with pytest.raises(ProtocolError):
        buffer.update(settings)
    assert buffer.settings == original


async def test_repeated_flushes_ordered_pcm_and_pronunciations(engine, monkeypatch):
    charges = []
    monkeypatch.setattr(generation, "charge", lambda dimension, cost: charges.append((dimension, cost)))
    engine.rate = 48000
    socket, task = await connect()
    socket.send("session.update", session={"pronunciation_replacements": {"API": "A P I"}})
    submit(socket, "API is ready.", "first")
    submit(socket, "Next turn.", "second")
    events = await socket.until("audio.done", flush_id="second")
    terminals = [event["flush_id"] for event in events if event["type"] == "audio.done"]
    assert terminals == ["first", "second"]
    for flush_id in terminals:
        audio = [event for event in events if event["type"] == "audio.delta" and event["flush_id"] == flush_id]
        assert [event["sequence"] for event in audio] == list(range(len(audio)))
        assert sum(len(base64.b64decode(event["audio"])) for event in audio) == 4800
        started = next(event for event in events if event["type"] == "audio.started" and event["flush_id"] == flush_id)
        assert started["sample_rate"] == 24000
    assert engine.calls == ["A P I is ready.", "Next turn."]
    assert charges == [("tts_chars", len("A P I is ready.")), ("tts_chars", len("Next turn."))]
    await close(socket, task)


async def test_cancel_waiting_for_queue_never_runs_engine(engine):
    release = asyncio.Event()

    async def blocking():
        await release.wait()

    task_queue.enqueue_generation("blocking", blocking())
    socket, task = await connect()
    submit(socket, "Never synthesize this.", "queued")
    await eventually(lambda: task_queue.pending_count() == 2)
    socket.send("input_text.cancel", flush_id="queued")
    events = await socket.until("audio.cancelled")
    assert all(event["type"] != "audio.done" for event in events)
    await eventually(lambda: task_queue.pending_count() == 1)
    release.set()
    await eventually(lambda: task_queue.pending_count() == 0)
    assert engine.calls == []
    submit(socket, "Still usable.", "next")
    await socket.until("audio.done", flush_id="next")
    await close(socket, task)


async def test_cancel_active_keeps_inference_slot_until_engine_finishes(engine):
    engine.gate.clear()
    socket, task = await connect()
    submit(socket, "Already synthesizing.", "active")
    await eventually(lambda: bool(engine.calls))
    submit(socket, "Still only in the socket queue.", "pending")
    socket.send("input_text.cancel")
    events = await socket.until("audio.cancelled", flush_id="active")
    assert any(event["type"] == "audio.cancelled" and event["flush_id"] == "pending" for event in events)
    assert not any(event["type"] == "audio.done" for event in events)
    assert task_queue.pending_count() == 1
    engine.gate.set()
    await eventually(lambda: task_queue.pending_count() == 0)
    assert engine.calls == ["Already synthesizing."]
    await close(socket, task)


async def test_disconnect_abandons_active_and_releases_session(engine):
    engine.gate.clear()
    socket, task = await connect()
    submit(socket, "Stop on disconnect.", "leaving")
    await eventually(lambda: bool(engine.calls))
    await close(socket, task)
    assert task_queue.pending_count() == 1
    engine.gate.set()
    await eventually(lambda: task_queue.pending_count() == 0)


async def test_rate_limited_flush_retains_buffer_for_retry(engine, monkeypatch):
    limiter = RateLimiter(clock=lambda: 0)
    principal = replace(CLIENT, limits=replace(CLIENT.limits, inference=1))
    monkeypatch.setattr(route, "charge", lambda dimension: limiter.charge_or_raise(principal, dimension))
    socket, task = await connect()
    submit(socket, "First.", "one")
    await socket.until("audio.done")
    submit(socket, "Second.", "two")
    events = await socket.until("error")
    assert events[-1]["error"]["code"] == "rate_limit_exceeded"
    assert events[-1]["error"]["retry_after_s"] == 60
    assert engine.calls == ["First."]
    monkeypatch.setattr(route, "charge", lambda *_args: None)
    socket.send("input_text.flush", flush_id="two")
    await socket.until("audio.done", flush_id="two")
    assert engine.calls == ["First.", "Second."]
    await close(socket, task)


async def test_generation_failure_after_audio_never_sends_done(engine):
    engine.fail_after = 1
    socket, task = await connect()
    socket.send("session.update", session={"max_chunk_chars": 100, "first_chunk_chars": 100})
    text = "The first sentence is long enough that the next must go into a separate chunk. " * 3
    submit(socket, text, "broken")
    events = await socket.until("error")
    assert any(event["type"] == "audio.delta" for event in events)
    assert events[-1]["flush_id"] == "broken"
    assert all(event["type"] != "audio.done" for event in events)
    await close(socket, task)


async def test_per_key_cap_is_shared_with_transcription(engine):
    assert live_sessions.reserve("transcription", CLIENT.key_id, per_key=2, max_global=4)
    assert live_sessions.reserve("transcription", CLIENT.key_id, per_key=2, max_global=4)
    socket = Socket()
    await route.speech_stream(socket)
    events = await socket.until("close")
    assert events[0]["error"]["code"] == "session_limit_reached"
    assert events[-1]["code"] == 1013
    assert live_sessions.open_sessions("speech") == 0


async def test_session_close_cancels_pending_and_reports_reason(engine):
    engine.gate.clear()
    socket, task = await connect()
    submit(socket, "Still active.", "one")
    await eventually(lambda: bool(engine.calls))
    socket.send("session.close")
    events = await socket.until("close")
    await asyncio.wait_for(task, 2)
    assert any(event["type"] == "audio.cancelled" for event in events)
    assert events[-2] == {"type": "session.closed", "reason": "client_closed"}
    assert live_sessions.open_sessions("speech") == 0
    engine.gate.set()


async def test_backpressure_timeout_abandons_job_and_releases_session(engine, monkeypatch):
    monkeypatch.setattr(route, "SEND_TIMEOUT_S", 0.05)
    socket, task = await connect()
    socket.block_audio = True
    submit(socket, "A client that stops reading.", "slow")
    await asyncio.wait_for(task, 2)
    assert live_sessions.open_sessions("speech") == 0
    assert (await socket.until("close"))[-1]["reason"] == "slow_consumer"
    await eventually(lambda: task_queue.pending_count() == 0)


async def test_cancel_all_removes_pending_targets_before_waiting_for_output(monkeypatch):
    connection = route._Connection(Socket(), CLIENT)
    connection.outbound = asyncio.Queue(maxsize=1)
    calls = []

    async def synthesize(flush):
        calls.append(flush.flush_id)
        await asyncio.Event().wait()

    monkeypatch.setattr(route, "charge", lambda *_args: None)
    monkeypatch.setattr(connection, "synthesize", synthesize)
    for flush_id in ("one", "two"):
        connection.buffer.append("Queued speech.")
        await connection.command("input_text.flush", {"flush_id": flush_id})
        connection.outbound.get_nowait()
        connection.outbound.task_done()
    connection.outbound.put_nowait({"type": "blocked"})
    cancelling = asyncio.create_task(connection.cancel(None))
    worker = None
    events = []
    try:
        # The first cancellation cannot be reported while the send queue is full.
        await asyncio.sleep(0)
        worker = asyncio.create_task(connection.worker())
        async with asyncio.timeout(2):
            while not cancelling.done():
                await asyncio.sleep(0)
                while not connection.outbound.empty():
                    events.append(connection.outbound.get_nowait())
                    connection.outbound.task_done()
        await cancelling
        assert calls == []
        assert {event["flush_id"] for event in events if event["type"] == "audio.cancelled"} == {"one", "two"}
        assert connection.buffer.pending_count == 0
        assert connection.unsettled == {}
    finally:
        connection.closing = True
        cancelling.cancel()
        if worker is not None:
            worker.cancel()
        await asyncio.gather(cancelling, *([worker] if worker is not None else []), return_exceptions=True)
        connection.abandon_traces()


async def test_close_reports_terminal_when_completed_flush_is_waiting_for_output(engine, monkeypatch):
    connection = None
    original = route._Connection
    allow_audio = asyncio.Event()

    class TrackedConnection(original):
        def __init__(self, *args):
            super().__init__(*args)
            nonlocal connection
            connection = self

        async def synthesize(self, flush):
            # One delta blocks in send_json and one fills the bounded queue, so the next
            # terminal event must wait even though synthesis has completed successfully.
            for sequence in range(2):
                await self.emit(
                    {"type": "audio.delta", "flush_id": flush.flush_id, "sequence": sequence, "audio": "AAA="}
                )
            return {"chunks": 2, "samples": 2}

    class BackpressuredSocket(Socket):
        async def send_json(self, event):
            if event["type"] == "audio.delta":
                await allow_audio.wait()
            await super().send_json(event)

    monkeypatch.setattr(route, "_Connection", TrackedConnection)
    monkeypatch.setattr(route, "MAX_OUTBOUND_MESSAGES", 1)
    socket = BackpressuredSocket()
    task = asyncio.create_task(route.speech_stream(socket))
    try:
        await socket.until("session.created")
        submit(socket, "Finish while the client pauses reads.", "one")
        await eventually(
            lambda: connection.active_task is not None and connection.active_task.done() and connection.outbound.full()
        )
        socket.send("session.close")
        await eventually(lambda: connection.closing)
        allow_audio.set()
        events = await socket.until("close")
        await asyncio.wait_for(task, 2)
        terminal = [event for event in events if event["type"] in {"audio.done", "audio.cancelled", "error"}]
        assert terminal == [{"type": "audio.cancelled", "flush_id": "one"}]
        assert connection.buffer.pending_count == 0
        assert connection.unsettled == {}
    finally:
        allow_audio.set()
        socket.disconnect()
        await asyncio.wait_for(task, 2)


async def test_idle_close_and_invalid_messages(engine, monkeypatch):
    monkeypatch.setattr(route, "idle_timeout_s", lambda: 0.05)
    monkeypatch.setattr(route, "STOP_POLL_S", 0.01)
    socket, task = await connect()
    socket.incoming.put_nowait({"type": "websocket.receive", "text": "not json"})
    assert (await socket.until("error"))[-1]["error"]["code"] == "invalid_json"
    events = await socket.until("close")
    assert events[-1]["reason"] == "idle_timeout"
    await task


async def test_speech_global_cap(engine, monkeypatch):
    monkeypatch.setattr(route, "max_sessions", lambda: 1)
    first, running = await connect()
    second = Socket()
    await route.speech_stream(second)
    assert (await second.until("error"))[-1]["error"]["code"] == "session_limit_reached"
    assert live_sessions.open_sessions("speech") == 1
    await close(first, running)


def test_speech_socket_auth_header_and_browser_token(tmp_path):
    app = FastAPI()
    security = SecuritySettings.from_env(
        frontend_dir=None,
        environ={
            "VOICEBOX_API_KEY_FILE": str(tmp_path / "api_key"),
            "VOICEBOX_API_KEYS_JSON": str(tmp_path / "keys.json"),
        },
    )
    runtime = install_security(app, security)
    runtime.startup()
    _, key = runtime.keystore.create("speech-auth", "client")
    app.include_router(route.router)
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/v1/audio/speech/stream"):
            pass
        token = client.post("/auth/media-token", headers={"Authorization": f"Bearer {key}"}).json()["token"]
        for path, headers in [
            ("/v1/audio/speech/stream", {"Authorization": f"Bearer {key}"}),
            (f"/v1/audio/speech/stream?token={token}", {}),
        ]:
            with client.websocket_connect(path, headers=headers) as socket:
                assert socket.receive_json()["type"] == "session.created"
                socket.send_json({"type": "session.close"})
                assert socket.receive_json() == {"type": "session.closed", "reason": "client_closed"}
    assert live_sessions.open_sessions("speech") == 0


def metric(name, **labels):
    return metrics.REGISTRY.get_sample_value(name, labels) or 0


@pytest.mark.skipif(not metrics.ENABLED, reason="prometheus_client not installed")
async def test_flush_metrics_and_content_free_job_correlation(engine, caplog):
    accepted = metric("voicebox_speech_flushes_total", status="accepted")
    completed = metric("voicebox_speech_flushes_total", status="completed")
    first = metric("voicebox_speech_first_audio_seconds_count")
    pending = metric("voicebox_speech_pending_flushes")
    with caplog.at_level(logging.INFO):
        socket, task = await connect()
        submit(socket, "Private text must never enter lifecycle logs.", "trace-flush")
        await socket.until("audio.done")
        await close(socket, task)
    assert metric("voicebox_speech_flushes_total", status="accepted") == accepted + 1
    assert metric("voicebox_speech_flushes_total", status="completed") == completed + 1
    assert metric("voicebox_speech_first_audio_seconds_count") == first + 1
    assert metric("voicebox_speech_pending_flushes") == pending
    records = [record for record in caplog.records if getattr(record, "event", "") == "speech.flush.started"]
    assert len(records) == 1
    assert records[0].flush_id == "trace-flush"
    assert records[0].job_id.startswith("stream-")
    assert records[0].session_id
    assert "Private text" not in caplog.text


@pytest.mark.skipif(not metrics.ENABLED, reason="prometheus_client not installed")
async def test_slow_reader_records_abandonment_and_balances_pending_gauge(engine, monkeypatch):
    monkeypatch.setattr(route, "SEND_TIMEOUT_S", 0.05)
    monkeypatch.setattr(route, "MAX_OUTBOUND_MESSAGES", 1)
    abandoned = metric("voicebox_speech_flushes_total", status="abandoned")
    completed = metric("voicebox_speech_flushes_total", status="completed")
    pending = metric("voicebox_speech_pending_flushes")
    blocked = metric("voicebox_speech_backpressure_total", direction="outbound")
    timeouts = metric("voicebox_speech_send_timeouts_total")
    socket, task = await connect()
    socket.block_audio = True
    submit(socket, "This audio cannot be handed to the transport.", "slow-trace")
    await asyncio.wait_for(task, 2)
    assert metric("voicebox_speech_flushes_total", status="abandoned") == abandoned + 1
    assert metric("voicebox_speech_flushes_total", status="completed") == completed
    assert metric("voicebox_speech_pending_flushes") == pending
    assert metric("voicebox_speech_backpressure_total", direction="outbound") > blocked
    assert metric("voicebox_speech_send_timeouts_total") == timeouts + 1


@pytest.mark.skipif(not metrics.ENABLED, reason="prometheus_client not installed")
async def test_cancelled_and_failed_flushes_have_distinct_terminal_metrics(engine):
    cancelled = metric("voicebox_speech_flushes_total", status="cancelled")
    failed = metric("voicebox_speech_flushes_total", status="failed")
    pending = metric("voicebox_speech_pending_flushes")
    engine.gate.clear()
    socket, task = await connect()
    submit(socket, "Cancel this audio.", "cancel-trace")
    await eventually(lambda: bool(engine.calls))
    socket.send("input_text.cancel", flush_id="cancel-trace")
    await socket.until("audio.cancelled")
    engine.gate.set()
    await eventually(lambda: task_queue.pending_count() == 0)
    engine.fail_after = 1
    submit(socket, "This generation fails.", "failed-trace")
    await socket.until("error")
    await close(socket, task)
    assert metric("voicebox_speech_flushes_total", status="cancelled") == cancelled + 1
    assert metric("voicebox_speech_flushes_total", status="failed") == failed + 1
    assert metric("voicebox_speech_pending_flushes") == pending


@pytest.mark.skipif(not metrics.ENABLED, reason="prometheus_client not installed")
async def test_inbound_queue_backpressure_and_unknown_errors_have_fixed_labels():
    connection = route._Connection(Socket(), CLIENT)
    queue = asyncio.Queue(maxsize=1)
    queue.put_nowait({"occupied": True})
    blocked = metric("voicebox_speech_backpressure_total", direction="inbound")
    putting = asyncio.create_task(connection._put(queue, {}, "inbound"))
    await asyncio.sleep(0)
    assert not putting.done()
    queue.get_nowait()
    await putting
    assert metric("voicebox_speech_backpressure_total", direction="inbound") == blocked + 1
    other = metric("voicebox_speech_errors_total", code="other")
    await connection.emit(route.error_event("untrusted-code", "private error detail"))
    assert metric("voicebox_speech_errors_total", code="other") == other + 1
    assert metric("voicebox_speech_errors_total", code="untrusted-code") == 0


@pytest.mark.skipif(not metrics.ENABLED, reason="prometheus_client not installed")
async def test_recoverable_command_error_does_not_finish_active_flush_metrics(engine):
    engine.gate.clear()
    failed = metric("voicebox_speech_flushes_total", status="failed")
    completed = metric("voicebox_speech_flushes_total", status="completed")
    pending = metric("voicebox_speech_pending_flushes")
    socket, task = await connect()
    submit(socket, "Keep this active.", "same-id")
    await eventually(lambda: bool(engine.calls))
    submit(socket, "Duplicate command.", "same-id")
    assert (await socket.until("error"))[-1]["error"]["code"] == "duplicate_flush_id"
    assert metric("voicebox_speech_pending_flushes") == pending + 1
    assert metric("voicebox_speech_flushes_total", status="failed") == failed
    engine.gate.set()
    await socket.until("audio.done")
    await close(socket, task)
    assert metric("voicebox_speech_flushes_total", status="completed") == completed + 1
    assert metric("voicebox_speech_pending_flushes") == pending


@pytest.mark.parametrize(
    ("voice_dictionary", "request_dictionary", "expected"),
    [
        (None, None, "Client pronunciation is ready."),
        ({"API": "Voice pronunciation"}, None, "Voice pronunciation is ready."),
        ({"API": "Voice pronunciation"}, {}, "API is ready."),
        ({"API": "Voice pronunciation"}, {"API": "SDK", "SDK": "S D K"}, "SDK is ready."),
    ],
)
async def test_socket_saved_pronunciations_resolve_once_with_explicit_override(
    engine, monkeypatch, voice_dictionary, request_dictionary, expected
):
    profile = await generation.profiles.get_profile(None)
    profile.pronunciation_replacements = voice_dictionary
    monkeypatch.setattr(
        FakeDb,
        "get",
        lambda *_args: SimpleNamespace(
            pronunciation_replacements={"API": "Client pronunciation"}, stt_prompt="", stt_hotwords=[]
        ),
    )
    socket, task = await connect()
    if request_dictionary is not None:
        socket.send("session.update", session={"pronunciation_replacements": request_dictionary})
    submit(socket, "API is ready.", "saved")
    await socket.until("audio.done")
    await close(socket, task)
    assert engine.calls == [expected]


async def test_cancelled_socket_override_does_not_leak_to_another_clients_defaults(engine, monkeypatch):
    monkeypatch.setattr(
        FakeDb,
        "get",
        lambda _self, _model, key_id: SimpleNamespace(
            pronunciation_replacements={"API": "First caller" if key_id == CLIENT.key_id else "Second caller"},
            stt_prompt="",
            stt_hotwords=[],
        ),
    )
    first, first_task = await connect()
    first.send("session.update", session={"pronunciation_replacements": {"API": "Cancelled override"}})
    engine.gate.clear()
    submit(first, "API is active.", "first")
    await eventually(lambda: bool(engine.calls))
    first.send("input_text.cancel", flush_id="first")
    await first.until("audio.cancelled")
    engine.gate.set()
    await eventually(lambda: task_queue.pending_count() == 0)
    other = replace(CLIENT, key_id="other-speech-client")
    monkeypatch.setattr(route, "get_principal", lambda: other)
    second, second_task = await connect()
    submit(second, "API is ready.", "second")
    await second.until("audio.done")
    first.send("session.update", session={"pronunciation_replacements": None})
    submit(first, "API is restored.", "third")
    await first.until("audio.done")
    second.disconnect()
    await asyncio.wait_for(second_task, 2)
    await close(first, first_task)
    assert engine.calls == [
        "Cancelled override is active.",
        "Second caller is ready.",
        "First caller is restored.",
    ]
