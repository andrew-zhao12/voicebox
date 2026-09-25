"""End-to-end tests for the auth middleware on a torch-free mini app."""

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.tests.security_testapp import build_test_app


@pytest.fixture
def harness(tmp_path):
    app, runtime = build_test_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    key = (tmp_path / "api_key").read_text().strip()
    return runtime, client, key


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_missing_key_is_401_with_a_challenge(harness):
    _, client, _ = harness
    response = client.get("/profiles")
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}
    assert response.headers["www-authenticate"] == 'Bearer realm="voicebox"'


def test_invalid_key_is_401(harness):
    _, client, _ = harness
    response = client.get("/profiles", headers=bearer("vbx_wrong"))
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API key"}
    assert 'error="invalid_token"' in response.headers["www-authenticate"]


def test_valid_key_reaches_the_route_with_a_principal(harness):
    _, client, key = harness
    response = client.get("/who", headers=bearer(key))
    assert response.status_code == 200
    assert response.json() == {"key_id": "local", "role": "admin", "via": "header"}


def test_health_is_minimal_without_a_key_and_full_with_one(harness):
    _, client, key = harness
    anonymous = client.get("/health")
    assert anonymous.status_code == 200
    assert anonymous.json() == {"status": "healthy", "service": "voicebox"}

    authed = client.get("/health", headers=bearer(key))
    assert authed.json()["model_loaded"] is False


def test_public_paths_need_no_key(harness):
    _, client, _ = harness
    assert client.get("/cloud/callback").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_docs_can_be_disabled(tmp_path):
    app, _ = build_test_app(tmp_path, env={"VOICEBOX_DISABLE_DOCS": "1"})
    client = TestClient(app)
    assert client.get("/docs").status_code == 401
    assert client.get("/openapi.json").status_code == 401


def test_preflight_bypasses_auth_and_401s_carry_cors_headers(harness):
    _, client, _ = harness
    preflight = client.options(
        "/profiles",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"

    denied = client.get("/profiles", headers={"Origin": "http://localhost:5173"})
    assert denied.status_code == 401
    assert denied.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "www-authenticate" in denied.headers["access-control-expose-headers"].lower()


def test_client_keys_are_limited_to_the_allowlist(harness):
    runtime, client, _ = harness
    _, client_key = runtime.keystore.create("app", "client")

    assert client.get("/profiles", headers=bearer(client_key)).status_code == 200
    assert client.post("/generate", json={"text": "hi"}, headers=bearer(client_key)).status_code == 200
    assert client.get("/mcp/", headers=bearer(client_key)).status_code == 200

    forbidden = client.post("/shutdown", headers=bearer(client_key))
    assert forbidden.status_code == 403
    assert forbidden.json() == {"detail": "Admin key required"}
    assert client.get("/captures", headers=bearer(client_key)).status_code == 403
    assert client.get("/mcp/bindings", headers=bearer(client_key)).status_code == 403
    assert client.get("/nope", headers=bearer(client_key)).status_code == 403
    assert client.get("/nope").status_code == 401


def test_media_tokens_work_only_as_a_query_param_on_get_media_paths(harness):
    _, client, key = harness
    token = client.post("/auth/media-token", headers=bearer(key)).json()["token"]

    assert client.get(f"/audio/x?token={token}").text == "audio:x"
    status = client.get(f"/generate/x/status?token={token}")
    assert status.status_code == 200
    assert "completed" in status.text

    assert client.get(f"/profiles?token={token}").status_code == 401
    assert client.post(f"/generate?token={token}", json={"text": "x"}).status_code == 401
    assert client.get("/audio/x", headers=bearer(token)).status_code == 401

    bad = client.get("/audio/x?token=nope")
    assert bad.status_code == 401
    assert bad.json() == {"detail": "Invalid or expired media token"}


def test_media_token_carries_the_live_role_and_dies_with_its_key(harness):
    runtime, client, _ = harness
    _, client_key = runtime.keystore.create("app", "client")
    token = client.post("/auth/media-token", headers=bearer(client_key)).json()["token"]

    assert client.get(f"/audio/x?token={token}").status_code == 200
    assert client.get(f"/captures/x/audio?token={token}").status_code == 403  # admin-only media path

    runtime.keystore.revoke("app")
    assert client.get(f"/audio/x?token={token}").status_code == 401


def test_bearer_failures_lock_the_ip_out_but_not_a_valid_key(harness):
    _, client, key = harness
    for _ in range(30):
        assert client.get("/profiles", headers=bearer("vbx_bad")).status_code == 401
    locked = client.get("/profiles", headers=bearer("vbx_bad"))
    assert locked.status_code == 429
    assert "retry-after" in locked.headers

    assert client.get("/profiles", headers=bearer(key)).status_code == 200
    # Media loads without a token never count and are never locked out.
    assert client.get("/audio/x").status_code == 401


def test_websocket_scopes_are_closed(harness):
    _, client, _ = harness
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws"):
        pass


def test_anonymous_browser_navigations_get_the_spa(tmp_path):
    frontend = tmp_path / "frontend"
    (frontend / "assets").mkdir(parents=True)
    (frontend / "index.html").write_text("<html>spa</html>")
    (frontend / "vite.svg").write_text("<svg/>")
    app, _ = build_test_app(tmp_path, frontend_dir=frontend)
    client = TestClient(app)
    key = (tmp_path / "api_key").read_text().strip()

    navigation = client.get("/captures", headers={"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"})
    assert navigation.status_code == 200
    assert navigation.text == "<html>spa</html>"
    assert client.get("/vite.svg").text == "<svg/>"
    assert client.get("/captures", headers={"Accept": "*/*"}).status_code == 401
    assert client.get("/captures", headers={"Accept": "application/json"}).status_code == 401
    assert client.get("/../etc/passwd", headers={"Accept": "text/html"}).status_code == 200  # index, not a file

    authed = client.get("/captures", headers={"Accept": "text/html", **bearer(key)})
    assert authed.json() == {"items": []}
    # /assets passes through to the router (a real build mounts StaticFiles there).
    assert client.get("/assets/app.js").status_code == 404


def test_unhandled_exceptions_are_masked_with_an_error_id(harness, caplog):
    _, client, key = harness
    response = client.get("/boom", headers=bearer(key))
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert "/secret/path" not in response.text
    assert body["error_id"] in caplog.text


def test_security_headers_are_added(harness):
    _, client, key = harness
    response = client.get("/profiles", headers=bearer(key))
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"
