"""Every route of the real app must be classified in ``backend/auth/policy.py``.

Requires the full app (torch); runs under ``just test``.
"""

import pytest

pytest.importorskip("torch")

from fastapi.routing import iter_route_contexts  # after the torch skip guard
from starlette.routing import Mount  # after the torch skip guard

from backend.app import app  # after the torch skip guard
from backend.auth import policy  # after the torch skip guard
from backend.auth.middleware import (  # after the torch skip guard
    AuthMiddleware,
    BodyLimitMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from backend.mcp_server.context import ClientIdMiddleware  # after the torch skip guard


def _registered() -> set[tuple[str, str]]:
    """Every (method, template) the app serves, with router prefixes applied."""
    found: set[tuple[str, str]] = set()
    for context in iter_route_contexts(app.routes):
        original = context.original_route
        if isinstance(original, Mount):
            found.add(("GET", original.path))
            found.add(("POST", original.path))
            continue
        if context.path is None:
            continue
        for method in context.methods or {"GET"}:
            found.add((method, context.path))
    return found


def test_every_route_is_classified():
    registered = _registered()
    assert len(registered) > 100, "route enumeration looks broken"
    unclassified = sorted(
        (method, path) for method, path in registered if policy.classify_template(method, path) is None
    )
    assert not unclassified, f"Add these routes to backend/auth/policy.py: {unclassified}"


def test_every_policy_template_exists():
    templates = {path for _, path in _registered()}
    for rules in (policy.INFERENCE_RULES, policy.MEDIA_TOKEN_RULES):
        for candidate in rules:
            assert candidate.template in templates, candidate.template
    for method, template, _ in policy.all_templates():
        if template in ("/mcp/", "/assets/{path:path}", "/{full_path:path}"):
            continue  # mount sub-path and Docker-only routes
        assert (method, template) in _registered() or template == "/mcp", (method, template)


def test_middleware_stack_order_and_error_handler():
    from fastapi.middleware.cors import CORSMiddleware

    assert [m.cls for m in app.user_middleware] == [
        CORSMiddleware,
        SecurityHeadersMiddleware,
        AuthMiddleware,
        RateLimitMiddleware,
        BodyLimitMiddleware,
        ClientIdMiddleware,
    ]
    assert Exception in app.exception_handlers
    assert app.state.security.settings.rate_limiting is True
