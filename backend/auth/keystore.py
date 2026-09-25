"""API key storage: the env key, the local key file and the hashed multi-key store.

Keys are ``vbx_`` + 32 random bytes (base64url).  Only SHA-256 digests are
kept for keys in ``api_keys.json``; the local key file holds the plaintext
because the desktop shell reads it to authenticate its own webviews.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .principal import LIMIT_FIELDS, KeyLimits

logger = logging.getLogger(__name__)

KEY_PREFIX = "vbx_"
RESERVED_IDS = frozenset({"env", "local"})
KEY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_PRESENTED_LEN = 256
STORE_VERSION = 1
ROLES = ("admin", "client")

Source = Literal["env", "local", "file"]


@dataclass(frozen=True)
class KeyRecord:
    id: str
    role: Literal["admin", "client"]
    sha256: str
    created_at: str
    limits: dict[str, int | None]
    source: Source

    def key_limits(self) -> KeyLimits:
        return KeyLimits.defaults_for(self.role).merged(self.limits)


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def validate_key_id(key_id: str) -> None:
    if not KEY_ID_RE.match(key_id or ""):
        raise ValueError("Key id must be 1-32 lowercase letters, digits, '-' or '_', starting with a letter or digit")
    if key_id in RESERVED_IDS:
        raise ValueError(f"Key id '{key_id}' is reserved")


def validate_limits(limits: Mapping[str, object] | None) -> dict[str, int | None]:
    """Return a clean copy of per-key limit overrides (``None`` = unlimited)."""
    if not limits:
        return {}
    clean: dict[str, int | None] = {}
    for name, value in limits.items():
        if name not in LIMIT_FIELDS:
            raise ValueError(f"Unknown limit '{name}'; expected one of {', '.join(LIMIT_FIELDS)}")
        if value is None:
            clean[name] = None
        elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Limit '{name}' must be a non-negative integer or null")
        else:
            clean[name] = value
    return clean


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_private(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` atomically with mode 0600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _warn_if_world_readable(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return
    if mode & 0o077:
        logger.warning("API key file %s is readable by other users (mode %o); consider chmod 600", path, mode)


def ensure_local_key_file(path: Path) -> str:
    """Return the key stored in ``path``, creating a fresh one (mode 0600) when absent."""
    if path.exists():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            _warn_if_world_readable(path)
            return existing
        key = generate_key()
        _write_private(path, key + "\n")
        logger.info("Created local API key file %s", path)
        return key

    path.parent.mkdir(parents=True, exist_ok=True)
    key = generate_key()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Lost a race with another writer (the desktop shell creates the same file).
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
        _write_private(path, key + "\n")
        return key
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    logger.info("Created local API key file %s", path)
    return key


class KeyStore:
    """Merged view of the env key, the local key file and ``api_keys.json``.

    ``api_keys.json`` is re-read when its mtime, size or inode changes (stat
    at most once per second), so keys created by the CLI or an admin route
    take effect without a restart.
    """

    def __init__(
        self,
        *,
        env_key: str | None,
        key_file: Callable[[], Path],
        keys_json: Callable[[], Path],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._env_key = env_key or None
        self._key_file = key_file
        self._keys_json = keys_json
        self._clock = clock
        self._lock = threading.RLock()
        self._static: dict[str, KeyRecord] = {}
        self._file: dict[str, KeyRecord] = {}
        self._loaded = False
        self._json_sig: tuple[int, int, int] | None = None
        self._last_stat = float("-inf")

    @property
    def local_key_path(self) -> Path | None:
        """Where the plaintext local key lives, or ``None`` when the env key is in use."""
        return None if self._env_key else self._key_file()

    @property
    def keys_json_path(self) -> Path:
        return self._keys_json()

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._load_static()
            self._load_json(force=True)
            self._last_stat = self._clock()
            self._loaded = True

    def describe(self) -> str:
        """One log line about where keys come from; never includes a key."""
        self.ensure_loaded()
        local = "disabled" if self._env_key else str(self._key_file())
        return (
            f"env={'yes' if self._env_key else 'no'}, local={local}, file={len(self._file)} keys ({self._keys_json()})"
        )

    def lookup(self, presented: str) -> KeyRecord | None:
        """Find the record whose digest matches ``presented`` (constant-time compares)."""
        if not presented or len(presented) > MAX_PRESENTED_LEN:
            return None
        with self._lock:
            self.ensure_loaded()
            self._maybe_reload()
            digest = hash_key(presented)
            match = None
            for record in self._records():
                if hmac.compare_digest(record.sha256, digest):
                    match = record
            return match

    def get(self, key_id: str) -> KeyRecord | None:
        with self._lock:
            self.ensure_loaded()
            self._maybe_reload()
            return self._static.get(key_id) or self._file.get(key_id)

    def list_keys(self) -> list[KeyRecord]:
        with self._lock:
            self.ensure_loaded()
            self._maybe_reload()
            return list(self._records())

    def create(self, key_id: str, role: str, limits: Mapping[str, object] | None = None) -> tuple[KeyRecord, str]:
        """Add a key to ``api_keys.json``; returns the record and the plaintext (shown once)."""
        validate_key_id(key_id)
        if role not in ROLES:
            raise ValueError(f"Role must be one of {', '.join(ROLES)}")
        clean_limits = validate_limits(limits)
        with self._lock:
            self.ensure_loaded()
            self._load_json()
            if key_id in self._static or key_id in self._file:
                raise KeyError(f"Key id '{key_id}' already exists")
            key = generate_key()
            record = KeyRecord(
                id=key_id,
                role=role,  # type: ignore[arg-type]  # validated against ROLES above
                sha256=hash_key(key),
                created_at=_now_iso(),
                limits=clean_limits,
                source="file",
            )
            self._file[key_id] = record
            self._save_json()
            return record, key

    def revoke(self, key_id: str) -> bool:
        """Remove a key from ``api_keys.json``; ``False`` when it was not there."""
        if key_id in RESERVED_IDS:
            raise ValueError(f"Key '{key_id}' cannot be revoked; unset VOICEBOX_API_KEY or delete the key file instead")
        with self._lock:
            self.ensure_loaded()
            self._load_json()
            if key_id not in self._file:
                return False
            del self._file[key_id]
            self._save_json()
            return True

    def _records(self) -> Iterator[KeyRecord]:
        yield from self._static.values()
        yield from self._file.values()

    def _load_static(self) -> None:
        if self._env_key:
            self._static = {
                "env": KeyRecord(
                    id="env", role="admin", sha256=hash_key(self._env_key), created_at="", limits={}, source="env"
                )
            }
            return
        key = ensure_local_key_file(self._key_file())
        self._static = {
            "local": KeyRecord(id="local", role="admin", sha256=hash_key(key), created_at="", limits={}, source="local")
        }

    def _stat_json(self) -> tuple[int, int, int] | None:
        try:
            st = self._keys_json().stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    def _maybe_reload(self) -> None:
        now = self._clock()
        if now - self._last_stat < 1.0:
            return
        self._last_stat = now
        self._load_json()

    def _load_json(self, force: bool = False) -> None:
        signature = self._stat_json()
        if not force and signature == self._json_sig:
            return
        if signature is None:
            self._file = {}
            self._json_sig = None
            return
        path = self._keys_json()
        try:
            parsed = self._parse(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Ignoring unreadable API key store %s: %s", path, e)
            return
        self._file = parsed
        self._json_sig = signature

    @staticmethod
    def _parse(data: object) -> dict[str, KeyRecord]:
        if not isinstance(data, dict):
            raise ValueError("top level must be an object")
        version = data.get("version", STORE_VERSION)
        if version != STORE_VERSION:
            raise ValueError(f"unsupported version {version!r}")
        entries = data.get("keys", [])
        if not isinstance(entries, list):
            raise ValueError("'keys' must be a list")
        records: dict[str, KeyRecord] = {}
        for entry in entries:
            try:
                if not isinstance(entry, dict):
                    raise ValueError("entry is not an object")
                key_id = str(entry["id"])
                validate_key_id(key_id)
                role = entry.get("role", "client")
                if role not in ROLES:
                    raise ValueError(f"bad role {role!r}")
                digest = str(entry["sha256"]).lower()
                if not SHA256_RE.match(digest):
                    raise ValueError("sha256 is not a hex digest")
                records[key_id] = KeyRecord(
                    id=key_id,
                    role=role,
                    sha256=digest,
                    created_at=str(entry.get("created_at", "")),
                    limits=validate_limits(entry.get("limits") or {}),
                    source="file",
                )
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(
                    "Skipping invalid API key entry %r: %s", entry.get("id") if isinstance(entry, dict) else entry, e
                )
        return records

    def _save_json(self) -> None:
        payload = {
            "version": STORE_VERSION,
            "keys": [
                {
                    "id": record.id,
                    "role": record.role,
                    "sha256": record.sha256,
                    "created_at": record.created_at,
                    "limits": record.limits,
                }
                for record in self._file.values()
            ],
        }
        _write_private(self._keys_json(), json.dumps(payload, indent=2) + "\n")
        self._json_sig = self._stat_json()
