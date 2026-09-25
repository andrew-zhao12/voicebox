"""Tests for the route classification table."""

import pytest

from backend.auth import policy


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("GET", "/health", "public"),
        ("HEAD", "/health", "public"),
        ("GET", "/", "public"),
        ("GET", "/cloud/callback", "public"),
        ("GET", "/docs", "public"),
        ("GET", "/openapi.json", "public"),
        ("POST", "/generate", "client"),
        ("POST", "/generate/stream", "client"),
        ("POST", "/speak", "client"),
        ("POST", "/transcribe", "client"),
        ("GET", "/generate/abc/status", "client"),
        ("POST", "/generate/abc/cancel", "client"),
        ("POST", "/generate/abc/retry", "admin"),
        ("GET", "/history/abc", "client"),
        ("GET", "/history/stats", "admin"),
        ("GET", "/history", "admin"),
        ("GET", "/audio/abc", "client"),
        ("GET", "/audio/version/abc", "client"),
        ("GET", "/samples/abc", "admin"),
        ("GET", "/profiles", "client"),
        ("POST", "/profiles", "admin"),
        ("GET", "/profiles/abc", "client"),
        ("GET", "/profiles/abc/avatar", "client"),
        ("POST", "/profiles/abc/avatar", "admin"),
        ("GET", "/profiles/presets/kokoro", "client"),
        ("GET", "/models/status", "client"),
        ("POST", "/models/abc/unload", "admin"),
        ("GET", "/effects/available", "client"),
        ("GET", "/effects/presets", "admin"),
        ("GET", "/tasks/active", "admin"),
        ("GET", "/captures", "admin"),
        ("POST", "/mcp/", "client"),
        ("GET", "/mcp", "client"),
        ("DELETE", "/mcp/", "client"),
        ("GET", "/mcp/bindings", "admin"),
        ("DELETE", "/mcp/bindings/x", "admin"),
        ("GET", "/events/speak", "admin"),
        ("POST", "/shutdown", "admin"),
        ("GET", "/auth/whoami", "client"),
        ("POST", "/auth/media-token", "client"),
        ("GET", "/auth/keys", "admin"),
        ("GET", "/nope", "admin"),
        ("GET", "/mcp/anything", "admin"),
    ],
)
def test_classify(method, path, expected):
    assert policy.classify(method, path) == expected


def test_docs_become_admin_when_disabled():
    assert policy.classify("GET", "/docs", docs_enabled=False) == "admin"
    assert policy.is_public("GET", "/openapi.json", docs_enabled=False) is False


def test_allows_by_role():
    assert policy.allows("admin", "POST", "/shutdown")
    assert policy.allows("client", "POST", "/generate")
    assert policy.allows("client", "GET", "/health")
    assert not policy.allows("client", "POST", "/shutdown")
    assert not policy.allows("client", "GET", "/captures")
    assert policy.allows("anonymous", "GET", "/health")
    assert not policy.allows("anonymous", "GET", "/profiles")


def test_media_tokens_are_get_only_and_allowlisted():
    assert policy.token_allowed("GET", "/audio/x")
    assert policy.token_allowed("HEAD", "/audio/version/x")
    assert policy.token_allowed("GET", "/generate/x/status")
    assert policy.token_allowed("GET", "/events/speak")
    assert not policy.token_allowed("POST", "/audio/x")
    assert not policy.token_allowed("GET", "/history")
    assert not policy.token_allowed("GET", "/profiles")


def test_inference_and_upload_rules():
    assert policy.is_inference("POST", "/generate")
    assert policy.is_inference("POST", "/captures/x/refine")
    assert not policy.is_inference("GET", "/profiles")
    assert policy.is_upload("POST", "/transcribe")
    assert policy.is_upload("POST", "/mcp/")
    assert not policy.is_upload("POST", "/generate")


def test_body_limits():
    kw = {"default": 2, "multipart_default": 256}
    assert policy.body_limit_for("POST", "/transcribe", "multipart/form-data", **kw) == 200 * policy.MIB
    assert policy.body_limit_for("POST", "/mcp/", "application/json", **kw) == 64 * policy.MIB
    assert policy.body_limit_for("POST", "/profiles/x/avatar", "multipart/form-data", **kw) == 10 * policy.MIB
    assert policy.body_limit_for("POST", "/generate", "application/json", **kw) == 2
    assert policy.body_limit_for("POST", "/unknown", "multipart/form-data; boundary=x", **kw) == 256


def test_classify_template_is_exact():
    assert policy.classify_template("GET", "/history/{generation_id}") == "client"
    assert policy.classify_template("HEAD", "/history/{generation_id}") == "client"
    assert policy.classify_template("DELETE", "/history/{generation_id}") == "admin"
    assert policy.classify_template("GET", "/{full_path:path}") == "public"
    assert policy.classify_template("GET", "/assets/{path:path}") == "public"
    assert policy.classify_template("POST", "/mcp") == "client"
    assert policy.classify_template("GET", "/unknown/{x}") is None
    assert policy.classify_template("PATCH", "/history/{generation_id}") is None


def test_every_template_is_listed_once_per_method():
    seen = set()
    for method, template, _access in policy.all_templates():
        assert (method, template) not in seen, f"{method} {template} listed twice"
        seen.add((method, template))
