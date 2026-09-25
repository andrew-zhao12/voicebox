"""Tests for the token buckets, the rate-limit middleware and the body caps."""

import io
import math

import pytest
from starlette.testclient import TestClient

from backend.auth.principal import KeyLimits, Principal
from backend.auth.ratelimit import RateLimited, RateLimiter
from backend.tests.security_testapp import build_test_app

MIB = 1024 * 1024


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_bucket_math_with_a_fake_clock():
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])

    for _ in range(60):
        assert limiter.charge("k", limit=60).allowed
    denied = limiter.charge("k", limit=60)
    assert not denied.allowed
    assert denied.retry_after_s == 1
    assert denied.remaining == 0

    now[0] += 1.0  # refills one token per second at 60/min
    assert limiter.charge("k", limit=60).allowed
    assert not limiter.charge("k", limit=60).allowed

    headers = RateLimiter.headers_for(denied)
    assert headers["RateLimit-Limit"] == "60"
    assert headers["Retry-After"] == "1"


def test_cost_above_capacity_is_rejected_with_a_refill_estimate():
    limiter = RateLimiter()
    decision = limiter.charge("k", limit=60, cost=120)
    assert not decision.allowed
    assert decision.retry_after_s == math.ceil(120 / (60 / 60))


def test_unlimited_and_disabled_limiters_always_allow():
    assert RateLimiter().charge("k", limit=None, cost=10**9).allowed
    off = RateLimiter(enabled=False)
    assert off.charge("k", limit=1).allowed
    assert off.charge("k", limit=1).allowed
    assert RateLimiter.headers_for(off.charge("k", limit=1)) == {}


def test_charge_or_raise_uses_the_principal_limits():
    limiter = RateLimiter()
    principal = Principal("app", "client", "header", KeyLimits.defaults_for("client").merged({"inference": 1}))
    limiter.charge_or_raise(principal, "inference")
    with pytest.raises(RateLimited) as excinfo:
        limiter.charge_or_raise(principal, "inference")
    assert excinfo.value.status_code == 429
    assert "Retry-After" in excinfo.value.headers


def test_sweep_drops_idle_buckets_and_caps_the_table():
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0], idle_s=10, max_buckets=3)
    for i in range(5):
        limiter.charge(f"k{i}", limit=10)
    assert len(limiter._buckets) <= 3
    now[0] += 11
    limiter.sweep()
    assert limiter._buckets == {}


@pytest.fixture
def limited(tmp_path):
    app, runtime = build_test_app(tmp_path)
    client = TestClient(app)
    admin_key = (tmp_path / "api_key").read_text().strip()
    return runtime, client, admin_key


def test_requests_bucket_returns_headers_then_429(limited):
    runtime, client, admin_key = limited
    _, key = runtime.keystore.create("app", "client", {"requests": 3})

    remaining = []
    for _ in range(3):
        response = client.get("/profiles", headers=bearer(key))
        assert response.status_code == 200
        assert response.headers["ratelimit-limit"] == "3"
        remaining.append(int(response.headers["ratelimit-remaining"]))
    assert remaining == [2, 1, 0]

    denied = client.get("/profiles", headers=bearer(key))
    assert denied.status_code == 429
    assert denied.headers["retry-after"]
    assert "requests" in denied.json()["detail"]

    assert client.get("/profiles", headers=bearer(admin_key)).status_code == 200


def test_inference_and_tts_chars_buckets(limited):
    runtime, client, _ = limited
    _, key = runtime.keystore.create("app", "client", {"inference": 1, "tts_chars": 10})

    assert client.post("/generate", json={"text": "short"}, headers=bearer(key)).status_code == 200
    second = client.post("/generate", json={"text": "short"}, headers=bearer(key))
    assert second.status_code == 429
    assert "inference" in second.json()["detail"]

    _, key2 = runtime.keystore.create("app2", "client", {"tts_chars": 10})
    denied = client.post("/generate", json={"text": "x" * 11}, headers=bearer(key2))
    assert denied.status_code == 429
    assert "tts_chars" in denied.json()["detail"]
    assert denied.headers["retry-after"]


def test_upload_bytes_bucket(limited):
    runtime, client, _ = limited
    _, key = runtime.keystore.create("app", "client", {"uploads_bytes": 100})

    files = {"file": ("a.wav", io.BytesIO(b"x" * 500), "audio/wav")}
    denied = client.post("/transcribe", files=files, headers=bearer(key))
    assert denied.status_code == 429
    assert "uploads" in denied.json()["detail"]


def test_rate_limiting_can_be_disabled(tmp_path):
    app, runtime = build_test_app(tmp_path, env={"VOICEBOX_RATE_LIMITING": "0"})
    client = TestClient(app)
    _, key = runtime.keystore.create("app", "client", {"requests": 1})
    for _ in range(3):
        response = client.get("/profiles", headers=bearer(key))
        assert response.status_code == 200
        assert "ratelimit-limit" not in response.headers


def test_body_caps(limited):
    _, client, admin_key = limited
    big_json = b'{"text": "' + b"x" * (2 * MIB) + b'"}'
    rejected = client.post(
        "/generate", content=big_json, headers={"Content-Type": "application/json", **bearer(admin_key)}
    )
    assert rejected.status_code == 413

    files = {"file": ("a.wav", io.BytesIO(b"x" * (3 * MIB)), "audio/wav")}
    accepted = client.post("/transcribe", files=files, headers=bearer(admin_key))
    assert accepted.status_code == 200
    assert accepted.json()["bytes"] == 3 * MIB
