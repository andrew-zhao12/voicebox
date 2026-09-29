"""``VOICEBOX_MCP_STATELESS``: MCP tool calls without a session, through the security stack.

Builds a small app with only the MCP mount (FastMCP, the tool modules and
the profile service import librosa, so this runs under the project venv).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import config, database
from backend.auth.install import install_security
from backend.auth.settings import SecuritySettings
from backend.mcp_server.context import ClientIdMiddleware
from backend.mcp_server.server import build_mcp_app, build_mcp_server, compose_lifespan, mcp_stateless

MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def test_env_flag():
    assert mcp_stateless({}) is False
    assert mcp_stateless({"VOICEBOX_MCP_STATELESS": "1"}) is True
    assert mcp_stateless({"VOICEBOX_MCP_STATELESS": "off"}) is False


@pytest.fixture
def make_api(tmp_path):
    previous = config.get_data_dir()
    config.set_data_dir(tmp_path)
    database.init_db()
    clients: list[TestClient] = []

    def factory(*, stateless: bool) -> SimpleNamespace:
        mcp_app = build_mcp_app(build_mcp_server(), stateless=stateless)

        @asynccontextmanager
        async def lifespan(app):
            async with compose_lifespan(mcp_app.router.lifespan_context)(app):
                yield

        settings = SecuritySettings.from_env(
            frontend_dir=None,
            environ={
                "VOICEBOX_API_KEY_FILE": str(tmp_path / "api_key"),
                "VOICEBOX_API_KEYS_JSON": str(tmp_path / "api_keys.json"),
            },
        )
        app = FastAPI(lifespan=lifespan)
        app.add_middleware(ClientIdMiddleware)
        runtime = install_security(app, settings)
        app.mount("/mcp", mcp_app)
        runtime.startup()
        _record, key = runtime.keystore.create(f"app{len(clients)}", "client", None)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return SimpleNamespace(client=client, key=key)

    try:
        yield factory
    finally:
        for client in clients:
            client.__exit__(None, None, None)
        config.set_data_dir(previous)


def _call_list_profiles(api) -> tuple[int, dict | None]:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "voicebox.list_profiles", "arguments": {}},
    }
    response = api.client.post("/mcp/", json=body, headers={**MCP_HEADERS, "Authorization": f"Bearer {api.key}"})
    if response.status_code != 200:
        return response.status_code, None
    text = response.text
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        payloads = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
        text = payloads[-1]
    return response.status_code, json.loads(text)


def test_stateless_tool_call_needs_no_session(make_api):
    api = make_api(stateless=True)
    status, message = _call_list_profiles(api)
    assert status == 200, message
    assert "result" in message, message
    assert message["result"].get("isError") is not True
    content = message["result"].get("structuredContent") or json.loads(message["result"]["content"][0]["text"])
    assert content == {"profiles": []}

    anonymous = api.client.post("/mcp/", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, headers=MCP_HEADERS)
    assert anonymous.status_code == 401


def test_stateful_transport_still_requires_a_session(make_api):
    api = make_api(stateless=False)
    status, message = _call_list_profiles(api)
    assert status in (400, 404), (status, message)
