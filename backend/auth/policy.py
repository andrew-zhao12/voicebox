"""Which caller may reach which route: the single place routes are classified.

Templates use FastAPI's syntax (``/audio/{generation_id}``).  ``classify``
answers for a concrete request path, ``classify_template`` for a registered
route template so a test can prove every route is listed here.  Anything not
listed is admin-only.  ``HEAD`` is treated as ``GET`` because Starlette adds
it to every GET route.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Access = Literal["public", "client", "admin"]

MIB = 1024 * 1024

_PARAM_RE = re.compile(r"{([^}:]+)(?::([^}]+))?}")


@dataclass(frozen=True)
class Rule:
    methods: frozenset[str]
    template: str
    regex: re.Pattern[str]
    wildcards: int

    def matches(self, method: str, path: str) -> bool:
        return ("*" in self.methods or method in self.methods) and self.regex.fullmatch(path) is not None


def _normalize_method(method: str) -> str:
    upper = method.upper()
    return "GET" if upper == "HEAD" else upper


def _compile(template: str) -> tuple[re.Pattern[str], int]:
    pattern = ""
    last = 0
    wildcards = 0
    for match in _PARAM_RE.finditer(template):
        pattern += re.escape(template[last : match.start()])
        pattern += ".*" if match.group(2) == "path" else "[^/]+"
        wildcards += 1
        last = match.end()
    pattern += re.escape(template[last:])
    return re.compile(pattern), wildcards


def rule(methods: str, template: str) -> Rule:
    """``rule("GET POST", "/profiles/{profile_id}")``; ``"*"`` matches every method."""
    regex, wildcards = _compile(template)
    normalized = frozenset(_normalize_method(m) for m in methods.split())
    return Rule(methods=normalized, template=template, regex=regex, wildcards=wildcards)


DOCS_PATHS = ("/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json")

PUBLIC_RULES: tuple[Rule, ...] = (
    rule("GET", "/"),
    rule("GET", "/health"),
    rule("GET", "/health/ready"),
    rule("GET", "/cloud/callback"),
    rule("GET", "/docs"),
    rule("GET", "/docs/oauth2-redirect"),
    rule("GET", "/redoc"),
    rule("GET", "/openapi.json"),
)

# Served by the auth middleware for anonymous browser navigations when the
# built frontend is present (Docker).  Not runtime rules: they exist so the
# coverage test can classify the catch-all routes.
SPA_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("GET", "/assets/{path:path}"),
    ("GET", "/{full_path:path}"),
)

CLIENT_RULES: tuple[Rule, ...] = (
    rule("POST", "/generate"),
    rule("POST", "/generate/stream"),
    rule("POST", "/speak"),
    rule("POST", "/generate/{generation_id}/cancel"),
    rule("GET", "/generate/{generation_id}/status"),
    rule("POST", "/transcribe"),
    rule("GET", "/history/{generation_id}"),
    rule("GET", "/audio/{generation_id}"),
    rule("GET", "/audio/version/{version_id}"),
    rule("GET", "/profiles"),
    rule("GET", "/profiles/{profile_id}"),
    rule("GET", "/profiles/{profile_id}/avatar"),
    rule("GET", "/profiles/presets/{engine}"),
    rule("GET", "/models/status"),
    rule("GET", "/effects/available"),
    rule("GET", "/auth/whoami"),
    rule("POST", "/auth/media-token"),
    rule("*", "/mcp"),
    rule("*", "/mcp/"),
)

ADMIN_RULES: tuple[Rule, ...] = (
    rule("POST", "/shutdown"),
    rule("POST", "/watchdog/disable"),
    rule("GET", "/health/filesystem"),
    rule("POST", "/maintenance/prune"),
    rule("POST", "/profiles"),
    rule("POST", "/profiles/import"),
    rule("PUT DELETE", "/profiles/{profile_id}"),
    rule("POST GET", "/profiles/{profile_id}/samples"),
    rule("DELETE PUT", "/profiles/samples/{sample_id}"),
    rule("POST DELETE", "/profiles/{profile_id}/avatar"),
    rule("GET", "/profiles/{profile_id}/export"),
    rule("GET PUT", "/profiles/{profile_id}/channels"),
    rule("PUT", "/profiles/{profile_id}/effects"),
    rule("POST", "/profiles/{profile_id}/compose"),
    rule("GET POST", "/channels"),
    rule("GET PUT DELETE", "/channels/{channel_id}"),
    rule("GET PUT", "/channels/{channel_id}/voices"),
    rule("POST", "/generate/{generation_id}/retry"),
    rule("POST", "/generate/{generation_id}/regenerate"),
    rule("POST", "/generate/import"),
    rule("GET", "/history"),
    rule("GET", "/history/stats"),
    rule("POST", "/history/import"),
    rule("DELETE", "/history/failed"),
    rule("POST", "/history/{generation_id}/favorite"),
    rule("DELETE", "/history/{generation_id}"),
    rule("GET", "/history/{generation_id}/export"),
    rule("GET", "/history/{generation_id}/export-audio"),
    rule("POST", "/llm/generate"),
    rule("POST GET", "/captures"),
    rule("GET DELETE", "/captures/{capture_id}"),
    rule("GET", "/captures/{capture_id}/audio"),
    rule("POST", "/captures/{capture_id}/refine"),
    rule("POST", "/captures/{capture_id}/retranscribe"),
    rule("GET", "/capture/readiness"),
    rule("GET POST", "/stories"),
    rule("GET PUT DELETE", "/stories/{story_id}"),
    rule("POST", "/stories/{story_id}/items"),
    rule("DELETE", "/stories/{story_id}/items/{item_id}"),
    rule("PUT", "/stories/{story_id}/items/times"),
    rule("PUT", "/stories/{story_id}/items/reorder"),
    rule("PUT", "/stories/{story_id}/items/{item_id}/move"),
    rule("PUT", "/stories/{story_id}/items/{item_id}/trim"),
    rule("PUT", "/stories/{story_id}/items/{item_id}/volume"),
    rule("POST", "/stories/{story_id}/items/{item_id}/split"),
    rule("POST", "/stories/{story_id}/items/{item_id}/duplicate"),
    rule("PUT", "/stories/{story_id}/items/{item_id}/version"),
    rule("GET", "/stories/{story_id}/export-audio"),
    rule("POST", "/effects/preview/{generation_id}"),
    rule("GET POST", "/effects/presets"),
    rule("GET PUT DELETE", "/effects/presets/{preset_id}"),
    rule("GET", "/generations/{generation_id}/versions"),
    rule("POST", "/generations/{generation_id}/versions/apply-effects"),
    rule("PUT", "/generations/{generation_id}/versions/{version_id}/set-default"),
    rule("DELETE", "/generations/{generation_id}/versions/{version_id}"),
    rule("GET", "/samples/{sample_id}"),
    rule("POST", "/models/load"),
    rule("POST", "/models/unload"),
    rule("POST", "/models/{model_name}/unload"),
    rule("GET", "/models/progress/{model_name}"),
    rule("GET", "/models/cache-dir"),
    rule("POST", "/models/migrate"),
    rule("GET", "/models/migrate/progress"),
    rule("POST", "/models/download"),
    rule("POST", "/models/download/cancel"),
    rule("DELETE", "/models/{model_name}"),
    rule("GET PUT", "/settings/captures"),
    rule("GET PUT", "/settings/generation"),
    rule("POST", "/tasks/clear"),
    rule("POST", "/cache/clear"),
    rule("GET", "/tasks/active"),
    rule("GET", "/backend/cuda-status"),
    rule("POST", "/backend/download-cuda"),
    rule("DELETE", "/backend/cuda"),
    rule("GET", "/backend/cuda-progress"),
    rule("GET", "/backend/rocm-status"),
    rule("POST", "/backend/download-rocm"),
    rule("DELETE", "/backend/rocm"),
    rule("GET", "/backend/rocm-progress"),
    rule("GET PUT", "/mcp/bindings"),
    rule("DELETE", "/mcp/bindings/{client_id}"),
    rule("GET", "/events/speak"),
    rule("POST", "/cloud/login/start"),
    rule("GET", "/cloud/status"),
    rule("POST", "/cloud/disconnect"),
    rule("GET POST", "/auth/keys"),
    rule("DELETE", "/auth/keys/{key_id}"),
)

# ``?token=`` is honoured only here, and only for GET/HEAD.
MEDIA_TOKEN_RULES: tuple[Rule, ...] = (
    rule("GET", "/generate/{generation_id}/status"),
    rule("GET", "/models/progress/{model_name}"),
    rule("GET", "/models/migrate/progress"),
    rule("GET", "/backend/cuda-progress"),
    rule("GET", "/backend/rocm-progress"),
    rule("GET", "/events/speak"),
    rule("GET", "/audio/{generation_id}"),
    rule("GET", "/audio/version/{version_id}"),
    rule("GET", "/samples/{sample_id}"),
    rule("GET", "/profiles/{profile_id}/avatar"),
    rule("GET", "/captures/{capture_id}/audio"),
    rule("GET", "/history/{generation_id}/export"),
    rule("GET", "/history/{generation_id}/export-audio"),
    rule("GET", "/profiles/{profile_id}/export"),
    rule("GET", "/stories/{story_id}/export-audio"),
)

# Charged to the per-key ``inference`` budget by the rate-limit middleware.
# MCP tool calls charge it themselves (the JSON-RPC body decides the tool).
INFERENCE_RULES: tuple[Rule, ...] = (
    rule("POST", "/generate"),
    rule("POST", "/generate/stream"),
    rule("POST", "/generate/{generation_id}/retry"),
    rule("POST", "/generate/{generation_id}/regenerate"),
    rule("POST", "/speak"),
    rule("POST", "/transcribe"),
    rule("POST", "/captures"),
    rule("POST", "/captures/{capture_id}/refine"),
    rule("POST", "/captures/{capture_id}/retranscribe"),
    rule("POST", "/llm/generate"),
    rule("POST", "/profiles/{profile_id}/compose"),
    rule("POST", "/models/load"),
    rule("POST", "/models/download"),
)

# Request bodies charged to ``uploads_bytes`` and capped individually.
BODY_LIMIT_RULES: tuple[tuple[Rule, int], ...] = (
    (rule("POST", "/transcribe"), 200 * MIB),
    (rule("POST", "/captures"), 200 * MIB),
    (rule("POST", "/generate/import"), 200 * MIB),
    (rule("POST", "/profiles/import"), 100 * MIB),
    (rule("POST", "/history/import"), 50 * MIB),
    (rule("POST", "/profiles/{profile_id}/samples"), 50 * MIB),
    (rule("POST", "/profiles/{profile_id}/avatar"), 10 * MIB),
    (rule("*", "/mcp"), 64 * MIB),
    (rule("*", "/mcp/"), 64 * MIB),
)

_CLASSES: tuple[tuple[Access, tuple[Rule, ...]], ...] = (
    ("public", PUBLIC_RULES),
    ("client", CLIENT_RULES),
    ("admin", ADMIN_RULES),
)


def classify(method: str, path: str, *, docs_enabled: bool = True) -> Access:
    """Access class of a concrete request; the most specific matching rule wins."""
    normalized = _normalize_method(method)
    best: tuple[int, Access] | None = None
    for access, rules in _CLASSES:
        for candidate in rules:
            if candidate.matches(normalized, path) and (best is None or candidate.wildcards < best[0]):
                best = (candidate.wildcards, access)
    if best is None:
        return "admin"
    if best[1] == "public" and not docs_enabled and path in DOCS_PATHS:
        return "admin"
    return best[1]


def is_public(method: str, path: str, *, docs_enabled: bool = True) -> bool:
    return classify(method, path, docs_enabled=docs_enabled) == "public"


def allows(role: str, method: str, path: str, *, docs_enabled: bool = True) -> bool:
    """Whether a caller with ``role`` (admin, client or anonymous) may reach the route."""
    if role == "admin":
        return True
    access = classify(method, path, docs_enabled=docs_enabled)
    if role == "client":
        return access in ("public", "client")
    return access == "public"


def _any(rules: tuple[Rule, ...], method: str, path: str) -> bool:
    normalized = _normalize_method(method)
    return any(candidate.matches(normalized, path) for candidate in rules)


def token_allowed(method: str, path: str) -> bool:
    return _normalize_method(method) == "GET" and _any(MEDIA_TOKEN_RULES, method, path)


def is_inference(method: str, path: str) -> bool:
    return _any(INFERENCE_RULES, method, path)


def is_upload(method: str, path: str) -> bool:
    normalized = _normalize_method(method)
    return any(candidate.matches(normalized, path) for candidate, _ in BODY_LIMIT_RULES)


def body_limit_for(method: str, path: str, content_type: str, *, default: int, multipart_default: int) -> int:
    normalized = _normalize_method(method)
    for candidate, cap in BODY_LIMIT_RULES:
        if candidate.matches(normalized, path):
            return cap
    if content_type.lower().startswith("multipart/"):
        return multipart_default
    return default


def classify_template(method: str, template: str) -> Access | None:
    """Exact lookup of a registered route template; ``None`` means unclassified."""
    normalized = _normalize_method(method)
    if (normalized, template) in SPA_TEMPLATES:
        return "public"
    for access, rules in _CLASSES:
        for candidate in rules:
            if candidate.template == template and ("*" in candidate.methods or normalized in candidate.methods):
                return access
    return None


def all_templates() -> list[tuple[str, str, Access]]:
    """Every ``(method, template, access)`` this policy knows about."""
    out: list[tuple[str, str, Access]] = []
    for access, rules in _CLASSES:
        for candidate in rules:
            for method in sorted(candidate.methods):
                out.append((method, candidate.template, access))
    return out
