"""Security configuration read from the environment.

Paths are callables because the application object is built at import time,
before ``config.set_data_dir`` runs from the CLI entry points; resolving
``{data}/api_key`` eagerly would create the key in the wrong directory.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .. import config

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

DEFAULT_MAX_QUEUE_DEPTH = 32
DEFAULT_MEDIA_TOKEN_TTL_S = 30 * 60
DEFAULT_BODY_LIMIT = 2 * MIB
DEFAULT_MULTIPART_LIMIT = 256 * MIB


def _env_flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _media_token_secret(raw: str) -> bytes | None:
    """A 32-byte HMAC key derived from ``VOICEBOX_MEDIA_TOKEN_SECRET``, or ``None`` for a per-process secret.

    Any string of at least 32 characters works (``openssl rand -base64 32``);
    every replica given the same value verifies the others' media tokens.
    """
    if not raw:
        return None
    if len(raw) < 32:
        logger.warning("VOICEBOX_MEDIA_TOKEN_SECRET is shorter than 32 characters; ignoring it")
        return None
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", name, raw)
        return default


@dataclass(frozen=True)
class SecuritySettings:
    """Knobs for authentication, rate limiting and request caps."""

    api_key_env: str | None
    key_file: Callable[[], Path]
    keys_json: Callable[[], Path]
    disable_docs: bool
    rate_limiting: bool
    max_queue_depth: int
    media_token_ttl_s: int
    media_token_secret: bytes | None
    body_limit_default: int
    body_limit_multipart: int
    cors_extra_origins: tuple[str, ...]
    frontend_dir: Path | None

    @classmethod
    def from_env(
        cls,
        *,
        frontend_dir: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> SecuritySettings:
        env = os.environ if environ is None else environ
        api_key = env.get("VOICEBOX_API_KEY", "").strip() or None
        key_file_override = env.get("VOICEBOX_API_KEY_FILE", "").strip()
        keys_json_override = env.get("VOICEBOX_API_KEYS_JSON", "").strip()

        def key_file() -> Path:
            if key_file_override:
                return Path(key_file_override).expanduser().resolve()
            return config.get_data_dir() / "api_key"

        def keys_json() -> Path:
            if keys_json_override:
                return Path(keys_json_override).expanduser().resolve()
            return config.get_data_dir() / "api_keys.json"

        origins = tuple(o.strip() for o in env.get("VOICEBOX_CORS_ORIGINS", "").split(",") if o.strip())
        multipart_mb = _env_int(env, "VOICEBOX_MAX_BODY_MB", DEFAULT_MULTIPART_LIMIT // MIB)
        return cls(
            api_key_env=api_key,
            key_file=key_file,
            keys_json=keys_json,
            disable_docs=_env_flag(env, "VOICEBOX_DISABLE_DOCS", False),
            rate_limiting=_env_flag(env, "VOICEBOX_RATE_LIMITING", True),
            max_queue_depth=max(1, _env_int(env, "VOICEBOX_MAX_QUEUE_DEPTH", DEFAULT_MAX_QUEUE_DEPTH)),
            media_token_ttl_s=max(60, _env_int(env, "VOICEBOX_MEDIA_TOKEN_TTL", DEFAULT_MEDIA_TOKEN_TTL_S)),
            media_token_secret=_media_token_secret(env.get("VOICEBOX_MEDIA_TOKEN_SECRET", "").strip()),
            body_limit_default=DEFAULT_BODY_LIMIT,
            body_limit_multipart=max(DEFAULT_BODY_LIMIT, multipart_mb * MIB),
            cors_extra_origins=origins,
            frontend_dir=frontend_dir,
        )
