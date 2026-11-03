"""Unit tests for the ClientIdMiddleware path predicate.

Locks down which endpoints advance ``last_seen_at`` on the
``MCPClientBinding`` row. Getting this wrong is silent: the Settings UI
just shows a stale "last heard from" timestamp and bindings never get
auto-created for new REST callers.
"""

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.auth.principal import ANONYMOUS, KeyLimits, Principal
from backend.mcp_server import context
from backend.mcp_server.context import _is_stamped_path


@pytest.mark.parametrize(
    "path",
    [
        "/mcp",
        "/mcp/",
        "/mcp/tools/call",
        "/mcp/bindings",  # admin REST; benign — frontend never sets the header
        "/speak",
        "/speak/",
    ],
)
def test_mcp_semantic_paths_are_stamped(path: str) -> None:
    assert _is_stamped_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/health",
        "/generate",
        "/captures",
        "/profiles",
        "/profiles/abc/compose",
        "/events/speak",
        "/tasks/active",
        "/llm/generate",
        # Prefix overlap should not match — /speakers is a hypothetical
        # future endpoint that shouldn't leak the stamp.
        "/speakers",
        # Same for anything starting with /mcpfoo.
        "/mcpfoo",
    ],
)
def test_other_paths_are_not_stamped(path: str) -> None:
    assert _is_stamped_path(path) is False


CLIENT = Principal(key_id="app", role="client", via="header", limits=KeyLimits.defaults_for("client"))


class SetPrincipal:
    """Stands in for the auth middleware, which stores the caller on the scope."""

    def __init__(self, app: ASGIApp, principal: Principal) -> None:
        self.app = app
        self.principal = principal

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope.setdefault("state", {})["principal"] = self.principal
        await self.app(scope, receive, send)


@pytest.mark.parametrize(
    ("path", "principal", "header", "stamps"),
    [
        ("/speak", CLIENT, "cursor", ["cursor"]),
        ("/health", CLIENT, "cursor", []),  # not a bindings path
        ("/speak", ANONYMOUS, "cursor", []),  # the auth middleware let it through unauthenticated
        ("/speak", CLIENT, None, []),
    ],
)
def test_the_middleware_sets_the_context_and_stamps_binding_calls(monkeypatch, path, principal, header, stamps):
    stamped: list[str] = []
    monkeypatch.setattr(context, "_enqueue_stamp", stamped.append)
    seen: list[tuple[str | None, str | None]] = []

    async def endpoint(request):
        seen.append((context.current_client_id.get(), context.current_remote_addr.get()))
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route(path, endpoint, methods=["POST"])])
    app = SetPrincipal(context.ClientIdMiddleware(app), principal)
    headers = {context.CLIENT_ID_HEADER: header} if header else {}
    response = TestClient(app).post(path, headers=headers)

    assert response.text == "ok"
    assert seen == [(header, "testclient")]
    assert stamped == stamps
    assert context.current_client_id.get() is None
