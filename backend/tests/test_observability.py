"""Request ids, the JSON log formatter and the metrics registry (torch-free)."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from backend.auth.principal import KeyLimits, Principal, principal_var
from backend.observability import logs, metrics
from backend.observability.requestid import request_id_var
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


@pytest.mark.skipif(not metrics.ENABLED, reason="prometheus_client not installed")
def test_rate_limit_rejections_are_counted(tmp_path):
    app, runtime = build_test_app(tmp_path, env={"VOICEBOX_API_KEYS_JSON": str(tmp_path / "keys.json")})
    _record, key = runtime.keystore.create("tiny", "client", {"requests": 1})
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/profiles", headers=bearer(key)).status_code == 200
    assert client.get("/profiles", headers=bearer(key)).status_code == 429
    text = metrics.render()[0].decode()
    assert 'voicebox_rate_limited_total{dimension="requests"}' in text
