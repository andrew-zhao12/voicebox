"""Tests for the API key store: env key, local key file and the hashed json store."""

import json
import os
import stat

import pytest

from backend.auth.keystore import KEY_PREFIX, KeyStore, ensure_local_key_file, generate_key, hash_key, validate_limits


def make_store(tmp_path, env_key=None, clock=None):
    kwargs = {"clock": clock} if clock else {}
    return KeyStore(
        env_key=env_key,
        key_file=lambda: tmp_path / "api_key",
        keys_json=lambda: tmp_path / "api_keys.json",
        **kwargs,
    )


def test_env_key_is_an_admin_record_and_suppresses_the_local_file(tmp_path):
    store = make_store(tmp_path, env_key="vbx_env")
    store.ensure_loaded()

    record = store.lookup("vbx_env")
    assert record is not None
    assert (record.id, record.role, record.source) == ("env", "admin", "env")
    assert store.local_key_path is None
    assert not (tmp_path / "api_key").exists()


def test_local_key_is_created_once_with_private_permissions(tmp_path):
    store = make_store(tmp_path)
    store.ensure_loaded()

    path = tmp_path / "api_key"
    key = path.read_text().strip()
    assert key.startswith(KEY_PREFIX)
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert store.lookup(key).id == "local"
    assert store.lookup(key).role == "admin"

    again = make_store(tmp_path)
    again.ensure_loaded()
    assert again.lookup(key) is not None


def test_existing_key_file_is_read_and_stripped(tmp_path):
    (tmp_path / "api_key").write_text("  vbx_existing \n")
    assert ensure_local_key_file(tmp_path / "api_key") == "vbx_existing"


def test_empty_key_file_is_replaced(tmp_path):
    (tmp_path / "api_key").write_text("")
    key = ensure_local_key_file(tmp_path / "api_key")
    assert key.startswith(KEY_PREFIX)
    assert (tmp_path / "api_key").read_text().strip() == key


def test_lookup_rejects_unknown_empty_and_oversized_keys(tmp_path):
    store = make_store(tmp_path, env_key="vbx_env")
    assert store.lookup("vbx_nope") is None
    assert store.lookup("") is None
    assert store.lookup("x" * 300) is None


def test_create_list_revoke_round_trip(tmp_path):
    store = make_store(tmp_path, env_key="vbx_env")

    record, key = store.create("myapp", "client", {"requests": 10, "tts_chars": None})
    assert record.role == "client"
    assert record.sha256 == hash_key(key)
    assert store.lookup(key).key_limits().requests == 10
    assert store.lookup(key).key_limits().tts_chars is None
    assert store.lookup(key).key_limits().inference == 30  # role default kept

    data = json.loads((tmp_path / "api_keys.json").read_text())
    assert data["version"] == 1
    assert data["keys"][0]["id"] == "myapp"
    assert key not in (tmp_path / "api_keys.json").read_text()
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "api_keys.json").stat().st_mode) == 0o600

    assert [r.id for r in store.list_keys()] == ["env", "myapp"]
    assert store.revoke("myapp") is True
    assert store.lookup(key) is None
    assert store.revoke("myapp") is False


def test_reserved_and_invalid_ids_are_refused(tmp_path):
    store = make_store(tmp_path, env_key="vbx_env")
    with pytest.raises(ValueError, match="reserved"):
        store.create("env", "client")
    with pytest.raises(ValueError, match="Key id must be"):
        store.create("Bad Id!", "client")
    with pytest.raises(ValueError, match="Role must be"):
        store.create("ok", "superuser")
    store.create("dup", "client")
    with pytest.raises(KeyError):
        store.create("dup", "client")
    with pytest.raises(ValueError, match="cannot be revoked"):
        store.revoke("local")


def test_json_changes_are_picked_up_after_the_stat_interval(tmp_path):
    now = [1000.0]
    store = make_store(tmp_path, env_key="vbx_env", clock=lambda: now[0])
    store.ensure_loaded()

    other = make_store(tmp_path, env_key="vbx_env")
    _, key = other.create("late", "client")

    assert store.lookup(key) is None  # stat throttled: just loaded
    now[0] += 2.0
    assert store.lookup(key) is not None


def test_corrupt_json_keeps_the_last_good_set(tmp_path):
    now = [0.0]
    store = make_store(tmp_path, env_key="vbx_env", clock=lambda: now[0])
    _, key = store.create("keep", "client")

    (tmp_path / "api_keys.json").write_text("{not json")
    now[0] += 2.0
    assert store.lookup(key) is not None


def test_invalid_entries_are_skipped_but_valid_ones_load(tmp_path):
    good = generate_key()
    (tmp_path / "api_keys.json").write_text(
        json.dumps(
            {
                "version": 1,
                "keys": [
                    {"id": "good", "role": "client", "sha256": hash_key(good), "created_at": "", "limits": {}},
                    {"id": "BAD", "role": "client", "sha256": "zz"},
                    {"id": "norole", "role": "root", "sha256": hash_key("x")},
                ],
            }
        )
    )
    store = make_store(tmp_path, env_key="vbx_env")
    assert store.lookup(good).id == "good"
    assert store.get("BAD") is None


def test_validate_limits():
    assert validate_limits(None) == {}
    assert validate_limits({"requests": 5, "inference": None}) == {"requests": 5, "inference": None}
    with pytest.raises(ValueError, match="Unknown limit"):
        validate_limits({"bogus": 1})
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_limits({"requests": -1})
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_limits({"requests": True})
