"""Tests for /auth/whoami, /auth/media-token and key management."""

import pytest
from starlette.testclient import TestClient

from backend.tests.security_testapp import build_test_app


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def harness(tmp_path):
    app, runtime = build_test_app(tmp_path)
    return runtime, TestClient(app), (tmp_path / "api_key").read_text().strip()


def test_whoami_reports_role_and_limits(harness):
    runtime, client, admin_key = harness
    me = client.get("/auth/whoami", headers=bearer(admin_key)).json()
    assert me["key_id"] == "local"
    assert me["role"] == "admin"
    assert me["via"] == "header"
    assert me["limits"]["max_pending_jobs"] is None

    _, key = runtime.keystore.create("app", "client", {"requests": 7})
    me = client.get("/auth/whoami", headers=bearer(key)).json()
    assert (me["key_id"], me["role"], me["limits"]["requests"]) == ("app", "client", 7)


def test_media_token_endpoint(harness):
    _, client, admin_key = harness
    body = client.post("/auth/media-token", headers=bearer(admin_key)).json()
    assert body["token"].count(".") == 1
    assert body["expires_in"] == 1800
    assert client.post("/auth/media-token").status_code == 401


def test_key_management_is_admin_only(harness):
    runtime, client, admin_key = harness

    created = client.post("/auth/keys", json={"id": "myapp", "role": "client"}, headers=bearer(admin_key))
    assert created.status_code == 201
    secret = created.json()["key"]
    assert secret.startswith("vbx_")
    assert created.json()["limits"]["max_pending_jobs"] == 4

    listing = client.get("/auth/keys", headers=bearer(admin_key)).json()["keys"]
    assert [k["id"] for k in listing] == ["local", "myapp"]
    assert all("key" not in k and "sha256" not in k for k in listing)

    assert client.get("/auth/keys", headers=bearer(secret)).status_code == 403
    assert client.post("/auth/keys", json={"id": "x"}, headers=bearer(secret)).status_code == 403

    assert client.post("/auth/keys", json={"id": "myapp"}, headers=bearer(admin_key)).status_code == 409
    assert client.post("/auth/keys", json={"id": "Bad Id"}, headers=bearer(admin_key)).status_code == 422
    assert client.post("/auth/keys", json={"id": "local"}, headers=bearer(admin_key)).status_code == 400
    assert (
        client.post("/auth/keys", json={"id": "z", "limits": {"bogus": 1}}, headers=bearer(admin_key)).status_code
        == 400
    )

    assert client.delete("/auth/keys/local", headers=bearer(admin_key)).status_code == 400
    assert client.delete("/auth/keys/missing", headers=bearer(admin_key)).status_code == 404
    assert client.delete("/auth/keys/myapp", headers=bearer(admin_key)).status_code == 204
    assert client.get("/profiles", headers=bearer(secret)).status_code == 401
