"""Request ids, the JSON log formatter and the metrics registry (torch-free)."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.auth.principal import KeyLimits, Principal, principal_var
from backend.observability import logs, metrics
from backend.observability.requestid import RequestIdMiddleware, current_request_id, request_id_var
from backend.observability.trace import flush_id_var, job_id_var, session_id_var
from backend.tests.security_testapp import build_test_app


@pytest.fixture
def harness(tmp_path):
    app, runtime = build_test_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    key = (tmp_path / "api_key").read_text().strip()
    return runtime, client, key


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_every_response_carries_a_request_id_even_from_the_auth_middleware(harness):
    _, client, key = harness
    response = client.get("/profiles")
    assert response.status_code == 401
    generated = response.headers["x-request-id"]
    assert len(generated) == 16

    response = client.get("/profiles", headers=bearer(key))
    assert response.status_code == 200
    assert response.headers["x-request-id"] != generated


def test_a_well_formed_client_request_id_is_echoed_and_a_bad_one_replaced(harness):
    _, client, key = harness
    response = client.get("/profiles", headers={**bearer(key), "X-Request-Id": "trace-abc.123:7"})
    assert response.headers["x-request-id"] == "trace-abc.123:7"

    response = client.get("/profiles", headers={**bearer(key), "X-Request-Id": "bad id with spaces"})
    assert response.headers["x-request-id"] != "bad id with spaces"
    assert len(response.headers["x-request-id"]) == 16


def test_the_error_id_of_a_500_is_the_request_id(harness, caplog):
    _, client, key = harness
    with caplog.at_level(logging.ERROR):
        response = client.get("/boom", headers={**bearer(key), "X-Request-Id": "req-500"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error", "error_id": "req-500"}
    assert response.headers["x-request-id"] == "req-500"
    assert "req-500" in caplog.text


def test_json_formatter_carries_request_and_key_ids():
    formatter = logs.JsonFormatter()
    record = logging.LogRecord("voicebox.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    token = request_id_var.set("req-1")
    principal_token = principal_var.set(
        Principal(key_id="myapp", role="client", via="header", limits=KeyLimits.defaults_for("client"))
    )
    try:
        payload = json.loads(formatter.format(record))
    finally:
        principal_var.reset(principal_token)
        request_id_var.reset(token)
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "voicebox.test"
    assert payload["request_id"] == "req-1"
    assert payload["key_id"] == "myapp"
    assert payload["ts"].endswith("+00:00")

    access = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1", "GET", "/health", "1.1", 200),
        None,
    )
    payload = json.loads(formatter.format(access))
    assert payload["method"] == "GET"
    assert payload["path"] == "/health"
    assert payload["status"] == 200
    assert "request_id" not in payload


def test_json_logging_env_and_apply():
    assert not logs.json_logging_enabled({})
    assert logs.json_logging_enabled({"VOICEBOX_LOG_FORMAT": "JSON"})
    probe = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    probe.addHandler(handler)
    try:
        assert logs.apply_json_logging() >= 1
        assert isinstance(handler.formatter, logs.JsonFormatter)
    finally:
        probe.removeHandler(handler)


@pytest.mark.skipif(not metrics.ENABLED, reason="prometheus_client not installed")
def test_http_metrics_are_recorded_and_rendered(harness):
    _, client, key = harness
    client.get("/profiles", headers=bearer(key))
    client.get("/profiles")
    body, content_type = metrics.render()
    text = body.decode()
    assert content_type.startswith("text/plain")
    assert 'voicebox_http_requests_total{method="GET",route="/profiles",status="200"}' in text
    assert 'voicebox_http_requests_total{method="GET",route="unrouted",status="401"}' in text
    assert "voicebox_http_request_seconds_bucket" in text
    assert "voicebox_queue_pending_jobs" in text
    assert "voicebox_http_requests_in_flight 0.0" in text  # back to zero once the responses are sent


@pytest.mark.skipif(not metrics.ENABLED, reason="prometheus_client not installed")
async def test_running_jobs_gauge_follows_the_lane_worker():
    import asyncio

    from backend.services import task_queue

    task_queue.init_queue(force=True, workers=2)
    release = asyncio.Event()

    async def blocker():
        await release.wait()

    def gauge(lane: str) -> float:
        for line in metrics.render()[0].decode().splitlines():
            if line.startswith(f'voicebox_queue_running_jobs{{lane="{lane}"}}'):
                return float(line.rsplit(" ", 1)[1])
        raise AssertionError(f"no gauge for lane {lane}")

    assert gauge("cpu") == 0.0
    assert gauge("gpu") == 0.0
    task_queue.enqueue_generation("job-1", blocker(), lane="gpu")
    await asyncio.sleep(0.05)
    assert gauge("gpu") == 1.0
    assert gauge("cpu") == 0.0
    release.set()
    await asyncio.sleep(0.05)
    assert gauge("gpu") == 0.0


@pytest.mark.skipif(not metrics.ENABLED, reason="prometheus_client not installed")
def test_rate_limit_rejections_are_counted(tmp_path):
    app, runtime = build_test_app(tmp_path, env={"VOICEBOX_API_KEYS_JSON": str(tmp_path / "keys.json")})
    _record, key = runtime.keystore.create("tiny", "client", {"requests": 1})
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/profiles", headers=bearer(key)).status_code == 200
    assert client.get("/profiles", headers=bearer(key)).status_code == 429
    text = metrics.render()[0].decode()
    assert 'voicebox_rate_limited_total{dimension="requests"}' in text


@pytest.mark.skipif(not metrics.ENABLED, reason="prometheus_client not installed")
def test_socket_trace_covers_handshakes_transcription_and_fixed_close_labels(caplog):
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    seen = []

    @app.websocket("/v1/realtime")
    async def socket(ws: WebSocket):
        seen.append(current_request_id())
        if ws.query_params.get("reject"):
            await ws.close(code=1008, reason="Bearer SECRET")
            return
        await ws.accept()
        await ws.receive_text()
        # Arbitrary application/peer close text must not become a metric label.
        await ws.close(code=1000, reason="user supplied secret reason")

    def sample(name, **labels):
        return metrics.REGISTRY.get_sample_value(name, labels) or 0

    active = sample("voicebox_websocket_sessions", kind="transcription")
    accepted = sample("voicebox_websocket_accepted_total", kind="transcription")
    other = sample("voicebox_websocket_closed_total", kind="transcription", reason="other")
    refused = sample("voicebox_websocket_closed_total", kind="transcription", reason="handshake_rejected")
    with caplog.at_level(logging.INFO), TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?token=SECRET", headers={"X-Request-Id": "socket-123"}) as ws:
            assert dict(ws.extra_headers)[b"x-request-id"] == b"socket-123"
            assert sample("voicebox_websocket_sessions", kind="transcription") == active + 1
            ws.send_text("PRIVATE TRANSCRIPT")
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/v1/realtime?reject=1"):
            pass
    assert seen[0] == "socket-123"
    assert seen[1]
    assert seen[1] != "socket-123"
    assert current_request_id() is None
    assert sample("voicebox_websocket_sessions", kind="transcription") == active
    assert sample("voicebox_websocket_accepted_total", kind="transcription") == accepted + 1
    assert sample("voicebox_websocket_closed_total", kind="transcription", reason="other") == other + 1
    assert sample("voicebox_websocket_closed_total", kind="transcription", reason="handshake_rejected") == refused + 1
    assert "SECRET" not in caplog.text
    assert "PRIVATE TRANSCRIPT" not in caplog.text
    assert "user supplied secret reason" not in caplog.text


async def test_queued_jobs_capture_caller_trace_without_leaking_between_jobs():
    import asyncio

    from backend import lifecycle
    from backend.services import task_queue

    lifecycle.reset()
    task_queue.init_queue(force=True)
    observed = []
    completed = asyncio.Event()

    async def record():
        principal = principal_var.get()
        observed.append(
            (
                current_request_id(),
                session_id_var.get(),
                flush_id_var.get(),
                job_id_var.get(),
                principal.key_id if principal else None,
            )
        )
        if len(observed) == 2:
            completed.set()

    request_token = request_id_var.set("request-one")
    session_token = session_id_var.set("session-one")
    flush_token = flush_id_var.set("flush-one")
    principal_token = principal_var.set(Principal("caller-one", "client", "header", KeyLimits.defaults_for("client")))
    try:
        task_queue.enqueue_generation("job-one", record())
    finally:
        principal_var.reset(principal_token)
        flush_id_var.reset(flush_token)
        session_id_var.reset(session_token)
        request_id_var.reset(request_token)
    task_queue.enqueue_generation("job-two", record())
    try:
        await asyncio.wait_for(completed.wait(), 2)
        assert observed == [
            ("request-one", "session-one", "flush-one", "job-one", "caller-one"),
            (None, None, None, "job-two", None),
        ]
        assert job_id_var.get() is None
    finally:
        await task_queue.shutdown(2)
        lifecycle.reset()


def test_json_formatter_emits_explicit_trace_fields_only():
    record = logging.LogRecord("voicebox.test", logging.INFO, __file__, 1, "Speech accepted", (), None)
    record.session_id = "session-1"
    record.flush_id = "flush-1"
    record.job_id = "job-1"
    record.event = "speech.flush.started"
    record.input = "PRIVATE INPUT"
    record.audio = "PRIVATE AUDIO"
    record.authorization = "PRIVATE KEY"
    payload = json.loads(logs.JsonFormatter().format(record))
    assert payload["session_id"] == "session-1"
    assert payload["flush_id"] == "flush-1"
    assert payload["job_id"] == "job-1"
    assert payload["event"] == "speech.flush.started"
    assert "PRIVATE" not in json.dumps(payload)
