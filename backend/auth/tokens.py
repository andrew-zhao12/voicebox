"""Short-lived media tokens for browser loads that cannot send headers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from .principal import Principal


@dataclass(frozen=True)
class TokenClaims:
    key_id: str
    role: str
    exp: int


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


class MediaTokenSigner:
    """HMAC-SHA256 tokens ``base64url(key_id|role|exp).base64url(signature)``.

    The secret is generated per process, so a restart invalidates every token;
    clients simply request a new one.
    """

    def __init__(
        self,
        secret: bytes | None = None,
        *,
        ttl_s: int = 1800,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret = secret or secrets.token_bytes(32)
        self.ttl_s = int(ttl_s)
        self._clock = clock

    def issue(self, principal: Principal) -> tuple[str, int]:
        """Return ``(token, expires_in_seconds)`` for ``principal``."""
        exp = int(self._clock()) + self.ttl_s
        payload = f"{principal.key_id}|{principal.role}|{exp}".encode()
        signature = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return f"{_b64encode(payload)}.{_b64encode(signature)}", self.ttl_s

    def verify(self, token: str) -> TokenClaims | None:
        """Return the claims when ``token`` is well formed, signed by us and unexpired."""
        if not token or len(token) > 512 or token.count(".") != 1:
            return None
        payload_part, signature_part = token.split(".", 1)
        try:
            payload = _b64decode(payload_part)
            signature = _b64decode(signature_part)
        except (binascii.Error, ValueError):
            return None
        expected = hmac.new(self._secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            return None
        try:
            key_id, role, exp_text = payload.decode("utf-8").split("|")
            exp = int(exp_text)
        except ValueError:
            return None
        if exp < self._clock():
            return None
        return TokenClaims(key_id=key_id, role=role, exp=exp)
