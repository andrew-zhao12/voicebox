"""Voice profile bundles: the 1.1 manifest round trip, 1.0 compatibility, conflict modes and atomicity.

Imports validate samples through librosa, so this runs under the project
venv (``just test``), not in the torch-free CI job.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from backend import config, database
from backend.database import ProfileSample, VoiceProfile
from backend.models import VoiceProfileCreate
from backend.services import export_import, profiles

SR = 24000


def wav_bytes(seconds: float = 2.5) -> bytes:
    t = np.arange(int(SR * seconds)) / SR
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def bundle(profile: dict, samples: dict[str, bytes], texts: dict[str, str], version: str = "1.1") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        zip_file.writestr("manifest.json", json.dumps({"version": version, "profile": profile, "has_avatar": False}))
        zip_file.writestr("samples.json", json.dumps(texts))
        for name, data in samples.items():
            zip_file.writestr(f"samples/{name}", data)
    return buffer.getvalue()


@pytest.fixture
def db(tmp_path):
    previous = config.get_data_dir()
    config.set_data_dir(tmp_path)
    database.init_db()
    session = next(database.get_db())
    try:
        yield session
    finally:
        session.close()
        config.set_data_dir(previous)


async def make_cloned(db, name="Narrator", tmp_path=None):
    created = await profiles.create_profile(VoiceProfileCreate(name=name, language="en", default_engine="qwen"), db)
    sample_path = tmp_path / f"{name}.wav"
    sample_path.write_bytes(wav_bytes())
    await profiles.add_profile_sample(created.id, str(sample_path), "A reference sentence.", db)
    return created


async def test_round_trip_keeps_every_field_of_a_preset_profile(db):
    created = await profiles.create_profile(
        VoiceProfileCreate(
            name="Heart",
            description="Warm narrator",
            language="en",
            voice_type="preset",
            preset_engine="kokoro",
            preset_voice_id="af_heart",
            personality="Speaks warmly.",
        ),
        db,
    )
    row = db.query(VoiceProfile).filter_by(id=created.id).first()
    row.effects_chain = json.dumps([{"type": "reverb", "enabled": True, "params": {"room_size": 0.4}}])
    db.commit()

    data = export_import.export_profile_to_zip(created.id, db)
    with zipfile.ZipFile(io.BytesIO(data)) as zip_file:
        manifest = json.loads(zip_file.read("manifest.json"))
    assert manifest["version"] == "1.1"
    assert manifest["profile"]["preset_voice_id"] == "af_heart"
    assert manifest["profile"]["effects_chain"][0]["type"] == "reverb"

    await profiles.delete_profile(created.id, db)
    result = await export_import.import_profile_bundle(data, db, on_conflict="skip")
    assert result.outcome == "created"
    imported = result.profile
    assert imported.name == "Heart"
    assert imported.voice_type == "preset"
    assert imported.preset_engine == "kokoro"
    assert imported.preset_voice_id == "af_heart"
    assert imported.default_engine == "kokoro"
    assert imported.description == "Warm narrator"
    assert imported.personality == "Speaks warmly."
    assert [e.type for e in imported.effects_chain] == ["reverb"]
    assert imported.id == created.id  # the id travels with the bundle when it is free


async def test_round_trip_of_cloned_and_designed_profiles(db, tmp_path):
    cloned = await make_cloned(db, tmp_path=tmp_path)
    designed = await profiles.create_profile(
        VoiceProfileCreate(name="Designed", language="en", voice_type="designed", design_prompt="A calm baritone."),
        db,
    )
    cloned_zip = export_import.export_profile_to_zip(cloned.id, db)
    designed_zip = export_import.export_profile_to_zip(designed.id, db)
    await profiles.delete_profile(cloned.id, db)
    await profiles.delete_profile(designed.id, db)

    restored = (await export_import.import_profile_bundle(cloned_zip, db)).profile
    assert restored.voice_type == "cloned"
    assert restored.default_engine == "qwen"
    samples = db.query(ProfileSample).filter_by(profile_id=restored.id).all()
    assert len(samples) == 1
    assert samples[0].reference_text == "A reference sentence."
    assert config.resolve_storage_path(samples[0].audio_path).exists()

    restored_designed = (await export_import.import_profile_bundle(designed_zip, db)).profile
    assert restored_designed.voice_type == "designed"
    assert restored_designed.design_prompt == "A calm baritone."


def test_export_refuses_a_cloned_profile_without_samples(db):
    row = VoiceProfile(name="Empty", language="en")
    db.add(row)
    db.commit()
    with pytest.raises(ValueError, match="no samples"):
        export_import.export_profile_to_zip(row.id, db)


async def test_manifest_1_0_imports_as_a_cloned_profile(db):
    data = bundle({"name": "Legacy", "language": "en"}, {"a.wav": wav_bytes()}, {"a.wav": "Old text."}, version="1.0")
    result = await export_import.import_profile_bundle(data, db)
    assert result.outcome == "created"
    assert result.profile.voice_type == "cloned"
    assert db.query(ProfileSample).filter_by(profile_id=result.profile.id).count() == 1


async def test_conflict_modes_skip_rename_and_replace(db, tmp_path):
    original = await make_cloned(db, name="Twin", tmp_path=tmp_path)
    data = export_import.export_profile_to_zip(original.id, db)

    skipped = await export_import.import_profile_bundle(data, db, on_conflict="skip")
    assert skipped.outcome == "skipped"
    assert skipped.profile.id == original.id

    renamed = await export_import.import_profile_bundle(data, db, on_conflict="rename")
    assert renamed.outcome == "created"
    assert renamed.name == "Twin (1)"
    assert renamed.profile.id != original.id  # the bundle's id is taken, so a new one is issued

    original_samples = {s.id for s in db.query(ProfileSample).filter_by(profile_id=original.id).all()}
    replaced = await export_import.import_profile_bundle(data, db, on_conflict="replace")
    assert replaced.outcome == "replaced"
    assert replaced.profile.name == "Twin"
    assert replaced.profile.id == original.id  # recreated under the same id
    new_samples = {s.id for s in db.query(ProfileSample).filter_by(profile_id=original.id).all()}
    assert new_samples
    assert not new_samples & original_samples
    assert {p.name for p in db.query(VoiceProfile).all()} == {"Twin", "Twin (1)"}

    with pytest.raises(ValueError, match="on_conflict"):
        await export_import.import_profile_bundle(data, db, on_conflict="overwrite")


async def test_a_bad_sample_leaves_no_profile_behind(db):
    data = bundle({"name": "Broken", "language": "en"}, {"a.wav": wav_bytes(0.4)}, {"a.wav": "Too short."})
    with pytest.raises(ValueError, match=r"Invalid sample a\.wav"):
        await export_import.import_profile_bundle(data, db)
    assert db.query(VoiceProfile).filter_by(name="Broken").first() is None

    missing = bundle({"name": "Missing", "language": "en"}, {}, {"gone.wav": "No file."})
    with pytest.raises(ValueError, match="not found in ZIP"):
        await export_import.import_profile_bundle(missing, db)

    bad_effects = bundle({"name": "Effects", "language": "en", "effects_chain": [{"type": "nope"}]}, {}, {})
    with pytest.raises(ValueError, match="effects chain"):
        await export_import.import_profile_bundle(bad_effects, db)
    assert db.query(VoiceProfile).count() == 0

    with pytest.raises(ValueError, match="Invalid ZIP"):
        await export_import.import_profile_bundle(b"not a zip", db)


async def test_bundle_ids_must_be_uuids(db):
    evil = bundle({"id": "../../escape", "name": "Evil", "language": "en"}, {"a.wav": wav_bytes()}, {"a.wav": "Hi."})
    result = await export_import.import_profile_bundle(evil, db)
    assert result.profile.id != "../../escape"
    assert config.resolve_storage_path(
        db.query(ProfileSample).filter_by(profile_id=result.profile.id).first().audio_path
    ).is_relative_to(config.get_profiles_dir())

    fixed = "0b7c7f64-5d1e-4c1a-9b83-2f7d9f6a1e20"
    ok = bundle({"id": fixed.upper(), "name": "Fixed", "language": "en"}, {"a.wav": wav_bytes()}, {"a.wav": "Hi."})
    assert (await export_import.import_profile_bundle(ok, db)).profile.id == fixed
