"""Wire authentication, rate limiting and the defensive middlewares into a FastAPI app."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .errors import unhandled_exception_handler
from .keystore import KeyStore
from .logfilter import install_access_log_redaction
from .middleware import AuthMiddleware, BodyLimitMiddleware, RateLimitMiddleware, SecurityHeadersMiddleware
from .principal import get_principal
from .ratelimit import RateLimiter
from .routes import router as auth_router
from .settings import SecuritySettings
from .tokens import MediaTokenSigner

logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:17493",
    "http://127.0.0.1:17493",
    "tauri://localhost",  # Tauri webview (macOS)
    "https://tauri.localhost",  # Tauri webview (Windows/Linux)
    "http://tauri.localhost",  # Tauri webview (Windows, some builds)
]

EXPOSED_HEADERS = [
    # Streaming metadata set by /generate/stream.
    "X-Voicebox-Sample-Rate",
    "X-Voicebox-Channels",
    "X-Voicebox-Sample-Format",
    "X-Voicebox-Stream-Mode",
    "X-Voicebox-Job-Id",
    # Auth and rate-limit feedback.
    "WWW-Authenticate",
    "Retry-After",
    "RateLimit-Limit",
    "RateLimit-Remaining",
    "RateLimit-Reset",
]


@dataclass
class SecurityRuntime:
    settings: SecuritySettings
    keystore: KeyStore
    tokens: MediaTokenSigner
    limiter: RateLimiter

    def startup(self) -> None:
        """Load (and if needed create) the keys; runs in the lifespan, after the data dir is final."""
        self.keystore.ensure_loaded()
        logger.info("API keys: %s", self.keystore.describe())
        if not self.settings.rate_limiting:
            logger.warning("Rate limiting is disabled (VOICEBOX_RATE_LIMITING=0)")


_runtime: SecurityRuntime | None = None


def get_runtime() -> SecurityRuntime | None:
    """The runtime installed on the application, or ``None`` outside a configured app (tests)."""
    return _runtime


def charge(dimension: str, cost: float = 1.0) -> None:
    """Charge the in-flight caller's ``dimension`` bucket; raises ``RateLimited`` (429) when exhausted."""
    runtime = _runtime
    if runtime is None:
        return
    runtime.limiter.charge_or_raise(get_principal(), dimension, cost)


def build_runtime(settings: SecuritySettings) -> SecurityRuntime:
    return SecurityRuntime(
        settings=settings,
        keystore=KeyStore(env_key=settings.api_key_env, key_file=settings.key_file, keys_json=settings.keys_json),
        tokens=MediaTokenSigner(ttl_s=settings.media_token_ttl_s),
        limiter=RateLimiter(enabled=settings.rate_limiting),
    )


def add_cors(app: FastAPI, settings: SecuritySettings) -> None:
    """CORS with local-first defaults plus ``VOICEBOX_CORS_ORIGINS``."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEFAULT_CORS_ORIGINS + list(settings.cors_extra_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=EXPOSED_HEADERS,
    )


def _add_bearer_scheme(app: FastAPI) -> None:
    """Declare the API key in the OpenAPI schema so Swagger's Authorize button works."""
    original = app.openapi

    def openapi_with_bearer() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = original()
        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {})["bearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "description": "Voicebox API key (vbx_...)",
        }
        schema.setdefault("security", [{"bearerAuth": []}])
        app.openapi_schema = schema
        return schema

    app.openapi = openapi_with_bearer  # type: ignore[method-assign]  # FastAPI documents this override


def install_security(app: FastAPI, settings: SecuritySettings) -> SecurityRuntime:
    """Add the security middlewares, the ``/auth`` routes and the error handler.

    Starlette's ``add_middleware`` inserts at the outside, so after this call
    the stack is CORS → SecurityHeaders → Auth → RateLimit → BodyLimit → the
    middlewares the caller added before → router.
    """
    global _runtime
    runtime = build_runtime(settings)
    _runtime = runtime
    app.add_middleware(BodyLimitMiddleware, settings=settings)
    app.add_middleware(RateLimitMiddleware, runtime=runtime)
    app.add_middleware(AuthMiddleware, runtime=runtime)
    app.add_middleware(SecurityHeadersMiddleware)
    add_cors(app, settings)
    app.include_router(auth_router)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    if not settings.disable_docs:
        _add_bearer_scheme(app)
    install_access_log_redaction()
    app.state.security = runtime
    return runtime
