"""Boot-time voice seeding: directory scanning, idempotence and the readiness step (torch-free)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import config, database, lifecycle
from backend.services import seed_profiles


def test_configured_dir_reads_the_environment():
    assert seed_profiles.configured_dir({}) is None
    assert seed_profiles.configured_dir({"VOICEBOX_SEED_PROFILES": "  "}) is None
    assert seed_profiles.configured_dir({"VOICEBOX_SEED_PROFILES": "/seed"}) == Path("/seed")


def test_bundle_paths_and_directory_bundles(tmp_path):
    (tmp_path / "b.voicebox.zip").write_bytes(b"zip")
    (tmp_path / "a.zip").write_bytes(b"zip")
    (tmp_path / ".hidden.zip").write_bytes(b"zip")
    (tmp_path / "notes.txt").write_text("ignored")
    unpacked = tmp_path / "c-unpacked"
    (unpacked / "samples").mkdir(parents=True)
    (unpacked / "manifest.json").write_text("{}")
    (unpacked / "samples.json").write_text("{}")
    (unpacked / "samples" / "x.wav").write_bytes(b"RIFF")
    (tmp_path / "plain-dir").mkdir()

    found = seed_profiles.bundle_paths(tmp_path)
    assert [p.name for p in found] == ["a.zip", "b.voicebox.zip", "c-unpacked"]
    assert seed_profiles.bundle_paths(tmp_path / "missing") == []

    data = seed_profiles.read_bundle(unpacked)
    with zipfile.ZipFile(__import__("io").BytesIO(data)) as zip_file:
        assert sorted(zip_file.namelist()) == ["manifest.json", "samples.json", "samples/x.wav"]
    assert seed_profiles.read_bundle(tmp_path / "a.zip") == b"zip"


@pytest.fixture
def data_dir(tmp_path):
    previous = config.get_data_dir()
    config.set_data_dir(tmp_path / "data")
    database.init_db()
    lifecycle.reset()
    yield tmp_path
    lifecycle.reset()
    config.set_data_dir(previous)


async def test_apply_reports_per_bundle_and_never_raises(data_dir):
    seed = data_dir / "seed"
    seed.mkdir()
    for name in ("one.zip", "two.zip", "bad.zip"):
        (seed / name).write_bytes(name.encode())

    async def fake_import(data, db, *, on_conflict):
        assert on_conflict == "skip"
        if data == b"bad.zip":
            raise ValueError("manifest missing")
        outcome = "skipped" if data == b"two.zip" else "created"
        return SimpleNamespace(outcome=outcome, name=data.decode().removesuffix(".zip"), profile=None)

    report = await seed_profiles.apply(seed, SimpleNamespace(rollback=lambda: None), importer=fake_import)
    assert report.created == ["one"]
    assert report.skipped == ["two"]
    assert report.failed == {"bad.zip": "manifest missing"}
    assert not report.ok
    assert report.summary() == "1 created, 1 already present, 1 failed"


async def test_run_gates_readiness_on_the_seed(data_dir, monkeypatch):
    seed = data_dir / "seed"
    seed.mkdir()

    async def ok(directory, db, *, on_conflict="skip"):
        return seed_profiles.SeedReport(created=["Narrator"])

    monkeypatch.setattr(seed_profiles, "apply", ok)
    lifecycle.step_begin(seed_profiles.STEP_NAME, "importing")
    assert not lifecycle.readiness(True)[0]
    report = await seed_profiles.run(seed)
    assert report.ok
    ready, body = lifecycle.readiness(True)
    assert ready
    assert body["startup"]["done"] == [seed_profiles.STEP_NAME]

    async def broken(directory, db, *, on_conflict="skip"):
        return seed_profiles.SeedReport(failed={"x.zip": "bad"})

    monkeypatch.setattr(seed_profiles, "apply", broken)
    await seed_profiles.run(seed)
    ready, body = lifecycle.readiness(True)
    assert not ready
    assert body["startup"]["failed"] == [seed_profiles.STEP_NAME]

    await seed_profiles.run(data_dir / "nowhere")
    assert not lifecycle.readiness(True)[0]
