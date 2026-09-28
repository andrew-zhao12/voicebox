"""Tests for media tokens."""

from backend.auth.principal import KeyLimits, Principal
from backend.auth.settings import SecuritySettings
from backend.auth.tokens import MediaTokenSigner

ADMIN = Principal(key_id="local", role="admin", via="header", limits=KeyLimits.defaults_for("admin"))


def test_issue_and_verify_round_trip():
    now = [1_000_000.0]
    signer = MediaTokenSigner(b"secret" * 6, ttl_s=60, clock=lambda: now[0])

    token, expires_in = signer.issue(ADMIN)
    assert expires_in == 60
    claims = signer.verify(token)
    assert (claims.key_id, claims.role, claims.exp) == ("local", "admin", 1_000_060)


def test_expired_token_is_rejected():
    now = [1_000_000.0]
    signer = MediaTokenSigner(b"secret" * 6, ttl_s=60, clock=lambda: now[0])
    token, _ = signer.issue(ADMIN)
    now[0] += 61
    assert signer.verify(token) is None


def test_tampering_and_foreign_secrets_are_rejected():
    signer = MediaTokenSigner(b"secret" * 6, ttl_s=60)
    token, _ = signer.issue(ADMIN)
    payload, signature = token.split(".")

    assert signer.verify(f"{payload}x.{signature}") is None
    assert signer.verify(f"{payload}.{signature[:-2]}AA") is None
    assert MediaTokenSigner(b"other" * 8, ttl_s=60).verify(token) is None
    assert signer.verify("") is None
    assert signer.verify("no-dot") is None
    assert signer.verify("a.b.c") is None
    assert signer.verify("!!!.???") is None


def test_shared_secret_from_the_environment_verifies_across_signers(caplog):
    env = {"VOICEBOX_MEDIA_TOKEN_SECRET": "K7fXo2Vq9mZp4Lr8Tb1Yw6Nc3Hd5Js0Ae="}
    first = SecuritySettings.from_env(environ=env).media_token_secret
    second = SecuritySettings.from_env(environ=env).media_token_secret
    assert first == second
    assert len(first) == 32

    token, _ = MediaTokenSigner(first, ttl_s=60).issue(ADMIN)
    assert MediaTokenSigner(second, ttl_s=60).verify(token) is not None
    assert MediaTokenSigner(ttl_s=60).verify(token) is None  # a per-process secret does not

    assert SecuritySettings.from_env(environ={}).media_token_secret is None
    assert SecuritySettings.from_env(environ={"VOICEBOX_MEDIA_TOKEN_SECRET": "too-short"}).media_token_secret is None
    assert "shorter than 32" in caplog.text
