"""``VOICEBOX_ALLOWED_HOSTS``: Host-header allowlisting against DNS rebinding (torch-free)."""

from fastapi.testclient import TestClient

from backend.auth.middleware import host_allowed, normalize_host
from backend.auth.settings import SecuritySettings
from backend.tests.security_testapp import build_test_app


def test_settings_parse_and_normalize_hosts():
    settings = SecuritySettings.from_env(
        environ={"VOICEBOX_ALLOWED_HOSTS": " Voice.Example.com ,*.Internal.example.com,, "}
    )
    assert settings.allowed_hosts == ("voice.example.com", "*.internal.example.com")
    assert SecuritySettings.from_env(environ={}).allowed_hosts == ()


def test_host_matching_rules():
    allowed = ("voice.example.com", "*.internal.example.com")
    assert host_allowed("voice.example.com", allowed)
    assert host_allowed("api.internal.example.com", allowed)
    assert host_allowed("deep.api.internal.example.com", allowed)
    assert not host_allowed("internal.example.com", allowed)  # the wildcard needs a subdomain
    assert not host_allowed("voice.example.com.evil.net", allowed)
    assert not host_allowed("", allowed)
    for loopback in ("localhost", "127.0.0.1", "::1"):
        assert host_allowed(loopback, ())
    assert normalize_host("Voice.Example.com:17493") == "voice.example.com"
    assert normalize_host("[::1]:17493") == "::1"
    assert normalize_host("[2001:db8::1]") == "2001:db8::1"
    assert normalize_host(None) == ""


def test_requests_for_other_hosts_are_refused_except_probes(tmp_path):
    app, _runtime = build_test_app(tmp_path, env={"VOICEBOX_ALLOWED_HOSTS": "voice.example.com,*.internal.example.com"})
    key = (tmp_path / "api_key").read_text().strip()
    client = TestClient(app, raise_server_exceptions=False)
    auth = {"Authorization": f"Bearer {key}"}

    assert client.get("/profiles", headers={**auth, "Host": "voice.example.com"}).status_code == 200
    assert client.get("/profiles", headers={**auth, "Host": "voice.example.com:8443"}).status_code == 200
    assert client.get("/profiles", headers={**auth, "Host": "api.internal.example.com"}).status_code == 200
    assert client.get("/profiles", headers={**auth, "Host": "127.0.0.1:17493"}).status_code == 200

    refused = client.get("/profiles", headers={**auth, "Host": "evil.example.org"})
    assert refused.status_code == 400
    assert refused.json() == {"detail": "Host not allowed"}
    assert refused.headers["x-request-id"]
    assert client.get("/profiles", headers=auth).status_code == 400  # TestClient's default host "testserver"

    envelope = client.post("/v1/audio/speech", json={}, headers={**auth, "Host": "evil.example.org"})
    assert envelope.status_code == 400
    assert envelope.json()["error"]["message"] == "Host not allowed"

    # Probes are reached by address, so they pass with any Host.
    assert client.get("/health", headers={"Host": "10.0.0.7:17493"}).status_code == 200
    assert client.get("/health/ready", headers={"Host": "10.0.0.7:17493"}).status_code == 200


def test_without_the_variable_every_host_passes(tmp_path):
    app, _runtime = build_test_app(tmp_path)
    key = (tmp_path / "api_key").read_text().strip()
    client = TestClient(app, raise_server_exceptions=False)
    assert (
        client.get("/profiles", headers={"Authorization": f"Bearer {key}", "Host": "anything.example"}).status_code
        == 200
    )
