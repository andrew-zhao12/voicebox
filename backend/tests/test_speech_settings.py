"""Reusable speech defaults remain bounded, isolated by key, and backward compatible."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.auth.install import install_security
from backend.auth.principal import KeyLimits, Principal
from backend.auth.settings import SecuritySettings
from backend.database import Base, SpeechSettings, get_db
from backend.database.migrations import run_migrations
from backend.routes.speech_settings import router
from backend.services import speech_settings as service
from backend.speech_settings_models import SpeechSettingsUpdate


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def api(tmp_path, monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr("backend.database.session.SessionLocal", factory)

    def sessions():
        with factory() as db:
            yield db

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = sessions
    settings = SecuritySettings.from_env(
        frontend_dir=None,
        environ={
            "VOICEBOX_API_KEY_FILE": str(tmp_path / "api_key"),
            "VOICEBOX_API_KEYS_JSON": str(tmp_path / "api_keys.json"),
        },
    )
    runtime = install_security(app, settings)
    runtime.startup()
    _, first = runtime.keystore.create("first", "client")
    _, second = runtime.keystore.create("second", "client")
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            first=first,
            second=second,
            admin=(tmp_path / "api_key").read_text().strip(),
            factory=factory,
            runtime=runtime,
        )
    engine.dispose()


def test_defaults_are_empty_and_read_does_not_create_rows(api):
    response = api.client.get("/speech/settings", headers=bearer(api.first))
    assert response.status_code == 200
    assert response.json() == {"pronunciation_replacements": {}, "stt_prompt": "", "stt_hotwords": []}
    with api.factory() as db:
        assert db.query(SpeechSettings).count() == 0


def test_partial_updates_null_reset_and_json_export_import(api):
    saved = {
        "pronunciation_replacements": {"API": "A P I"},
        "stt_prompt": "Voice software",
        "stt_hotwords": ["Voicebox"],
    }
    response = api.client.put("/speech/settings", headers=bearer(api.first), json=saved)
    assert response.status_code == 200
    assert response.json() == saved
    exported = api.client.get("/speech/settings", headers=bearer(api.first)).json()
    imported = api.client.put("/speech/settings", headers=bearer(api.second), json=exported)
    assert imported.json() == saved
    reset = api.client.put("/speech/settings", headers=bearer(api.first), json={"stt_hotwords": None})
    assert reset.json() == {**saved, "stt_hotwords": []}
    assert api.client.put("/speech/settings", headers=bearer(api.first), json={}).json() == reset.json()
    assert api.client.get("/speech/settings", headers=bearer(api.second)).json() == saved


def test_clients_cannot_read_or_write_another_keys_settings(api):
    assert api.client.get("/speech/settings").status_code == 401
    for method in ("get", "put"):
        kwargs = {"json": {"stt_prompt": "stolen"}} if method == "put" else {}
        response = getattr(api.client, method)("/speech/settings/second", headers=bearer(api.first), **kwargs)
        assert response.status_code == 403
    assert api.client.get("/speech/settings", headers=bearer(api.second)).json()["stt_prompt"] == ""


def test_admin_can_edit_existing_keys_but_not_unknown_or_revoked_ones(api):
    response = api.client.put("/speech/settings/first", headers=bearer(api.admin), json={"stt_prompt": "Acme"})
    assert response.status_code == 200
    assert api.client.get("/speech/settings", headers=bearer(api.first)).json()["stt_prompt"] == "Acme"
    assert api.client.get("/speech/settings/first", headers=bearer(api.admin)).json() == response.json()
    api.runtime.keystore.revoke("second")
    for key_id in ("missing", "second", "Invalid.Key"):
        target = f"/speech/settings/{key_id}"
        assert api.client.get(target, headers=bearer(api.admin)).status_code == 404
        assert api.client.put(target, headers=bearer(api.admin), json={"stt_prompt": "No"}).status_code == 404
    assert api.client.get("/speech/settings", headers=bearer(api.second)).status_code == 401
    with api.factory() as db:
        assert [row.key_id for row in db.query(SpeechSettings).all()] == ["first"]


@pytest.mark.parametrize(
    "payload",
    [
        {"stt_prompt": "x" * 4097},
        {"stt_hotwords": [""]},
        {"stt_hotwords": ["x" * 101]},
        {"stt_hotwords": ["x"] * 101},
        {"stt_hotwords": ["x" * 100] * 42},
        {"pronunciation_replacements": {"": "spoken"}},
        {"pronunciation_replacements": {"x": "y" * 257}},
        {"pronunciation_replacements": {str(i): "x" * 100 for i in range(100)}},
        {"owner_key_id": "second"},
    ],
)
def test_malformed_or_oversized_settings_are_rejected(api, payload):
    assert api.client.put("/speech/settings", headers=bearer(api.first), json=payload).status_code == 422
    with api.factory() as db:
        assert db.query(SpeechSettings).count() == 0


def test_dictionary_precedence_and_non_recursive_resolution(api):
    principal = Principal("first", "client", "header", KeyLimits.defaults_for("client"))
    with api.factory() as db:
        service.update_settings(db, "first", SpeechSettingsUpdate(pronunciation_replacements={"API": "client"}))
        voice = SimpleNamespace(pronunciation_replacements=None)
        assert service.resolve_pronunciations(db, voice, principal) == {"API": "client"}
        voice.pronunciation_replacements = {"API": "voice"}
        assert service.resolve_pronunciations(db, voice, principal) == {"API": "voice"}
        assert service.resolve_pronunciations(db, voice, principal, {"API": "request"}) == {"API": "request"}
        assert service.resolve_pronunciations(db, voice, principal, {}) == {}
        voice.pronunciation_replacements = {}
        assert service.resolve_pronunciations(db, voice, principal) == {}


def test_transcription_hints_respect_explicit_empty_overrides(api):
    principal = Principal("first", "client", "header", KeyLimits.defaults_for("client"))
    with api.factory() as db:
        service.update_settings(db, "first", SpeechSettingsUpdate(stt_prompt="Names", stt_hotwords=["Acme"]))
    assert service.resolve_transcription_hints(None, None, principal=principal) == ("Names", ["Acme"])
    assert service.resolve_transcription_hints("", [], principal=principal) == ("", [])
    assert service.resolve_transcription_hints("Custom", None, principal=principal) == ("Custom", ["Acme"])
    other = Principal("second", "client", "header", KeyLimits.defaults_for("client"))
    assert service.resolve_transcription_hints(None, None, principal=other) == ("", [])


def test_profile_migration_is_idempotent_and_keeps_existing_rows():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE profiles (id VARCHAR PRIMARY KEY, name VARCHAR)"))
        conn.execute(text("INSERT INTO profiles VALUES ('old', 'Old voice')"))
    run_migrations(engine)
    run_migrations(engine)
    assert "pronunciation_replacements" in {column["name"] for column in inspect(engine).get_columns("profiles")}
    with engine.connect() as conn:
        row = conn.execute(text("SELECT name, pronunciation_replacements FROM profiles WHERE id='old'")).one()
        assert tuple(row) == ("Old voice", None)
    engine.dispose()


def test_concurrent_first_partial_updates_keep_both_fields(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'settings.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    ready = threading.Barrier(2)

    @event.listens_for(engine, "before_cursor_execute")
    def synchronize_inserts(_conn, _cursor, statement, _params, _context, _many):
        if statement.startswith("INSERT INTO speech_settings"):
            ready.wait(timeout=5)

    def update(patch):
        with factory() as db:
            return service.update_settings(db, "first", patch)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(update, SpeechSettingsUpdate(stt_prompt="Names")),
            executor.submit(update, SpeechSettingsUpdate(stt_hotwords=["Acme"])),
        ]
        for future in futures:
            future.result(timeout=10)
    with factory() as db:
        assert db.query(SpeechSettings).count() == 1
        assert service.get_settings(db, "first").model_dump() == {
            "pronunciation_replacements": {},
            "stt_prompt": "Names",
            "stt_hotwords": ["Acme"],
        }
    engine.dispose()
