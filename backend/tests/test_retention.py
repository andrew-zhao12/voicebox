"""Retention sweep against a temporary SQLite database and data directory (torch-free)."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

import pytest

from backend import config, database
from backend.database import (
    Capture as DBCapture,
    Generation as DBGeneration,
    GenerationVersion as DBGenerationVersion,
    Story as DBStory,
    StoryItem as DBStoryItem,
    VoiceProfile as DBVoiceProfile,
    session as db_session,
)
from backend.services import retention


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    database.init_db()
    yield tmp_path
    db_session.engine.dispose()


def _old(days: int) -> datetime:
    return datetime.utcnow() - timedelta(days=days)


def _touch(path, *, days_old: int, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    stamp = (datetime.utcnow() - timedelta(days=days_old)).timestamp()
    os.utime(path, (stamp, stamp))


def _generation(db, profile_id: str, *, days_old: int, data_dir, **fields) -> DBGeneration:
    gen_id = str(uuid.uuid4())
    audio = data_dir / "generations" / f"{gen_id}.wav"
    _touch(audio, days_old=days_old)
    gen = DBGeneration(
        id=gen_id,
        profile_id=profile_id,
        text="hello",
        language="en",
        audio_path=config.to_storage_path(audio),
        duration=1.0,
        created_at=_old(days_old),
        **{"status": "completed", **fields},
    )
    db.add(gen)
    db.commit()
    return gen


def test_configured_days_parsing(caplog):
    assert retention.configured_days({}) is None
    assert retention.configured_days({"VOICEBOX_RETENTION_DAYS": ""}) is None
    assert retention.configured_days({"VOICEBOX_RETENTION_DAYS": "30"}) == 30
    assert retention.configured_days({"VOICEBOX_RETENTION_DAYS": "0"}) == 0
    assert retention.configured_days({"VOICEBOX_RETENTION_DAYS": "-1"}) is None
    assert retention.configured_days({"VOICEBOX_RETENTION_DAYS": "soon"}) is None
    assert "retention disabled" in caplog.text


async def test_prune_keeps_recent_favourite_story_and_active_rows(data_dir):
    db = db_session.SessionLocal()
    try:
        profile = DBVoiceProfile(id=str(uuid.uuid4()), name="p", language="en")
        db.add(profile)
        db.commit()

        old = _generation(db, profile.id, days_old=40, data_dir=data_dir)
        old_version_file = data_dir / "generations" / f"{old.id}_take.wav"
        _touch(old_version_file, days_old=40)
        db.add(
            DBGenerationVersion(
                id=str(uuid.uuid4()),
                generation_id=old.id,
                label="take-2",
                audio_path=config.to_storage_path(old_version_file),
                is_default=True,
            )
        )
        recent = _generation(db, profile.id, days_old=2, data_dir=data_dir)
        favourite = _generation(db, profile.id, days_old=40, data_dir=data_dir, is_favorited=True)
        in_story = _generation(db, profile.id, days_old=40, data_dir=data_dir)
        story = DBStory(id=str(uuid.uuid4()), name="s")
        db.add(story)
        db.commit()
        db.add(DBStoryItem(id=str(uuid.uuid4()), story_id=story.id, generation_id=in_story.id))
        active = _generation(db, profile.id, days_old=40, data_dir=data_dir, status="generating")
        db.commit()

        old_capture_file = data_dir / "captures" / "old.wav"
        _touch(old_capture_file, days_old=40)
        recent_capture_file = data_dir / "captures" / "new.wav"
        _touch(recent_capture_file, days_old=1)
        for name, path, days in (("old", old_capture_file, 40), ("new", recent_capture_file, 1)):
            db.add(
                DBCapture(
                    id=name,
                    audio_path=config.to_storage_path(path),
                    transcript_raw="t",
                    created_at=_old(days),
                )
            )
        db.commit()

        orphan_old = data_dir / "generations" / "orphan.wav"
        _touch(orphan_old, days_old=3, size=1000)
        orphan_fresh = data_dir / "generations" / "in-flight.wav"
        _touch(orphan_fresh, days_old=0)
        capture_orphan = data_dir / "captures" / "stray.wav"
        _touch(capture_orphan, days_old=3)

        report = await retention.prune(db, days=30)

        assert report.generations == 1
        assert report.captures == 1
        assert report.orphan_files == 2
        assert report.errors == 0
        assert report.freed_bytes == 16 + 16 + 16 + 1000 + 16

        remaining = {row.id for row in db.query(DBGeneration).all()}
        assert remaining == {recent.id, favourite.id, in_story.id, active.id}
        assert not (data_dir / "generations" / f"{old.id}.wav").exists()
        assert not old_version_file.exists()
        assert db.query(DBGenerationVersion).filter_by(generation_id=old.id).count() == 0
        assert {row.id for row in db.query(DBCapture).all()} == {"new"}
        assert not old_capture_file.exists()
        assert recent_capture_file.exists()
        assert not orphan_old.exists()
        assert orphan_fresh.exists()
        assert not capture_orphan.exists()
    finally:
        db.close()


async def test_prune_with_zero_days_still_spares_files_younger_than_a_day(data_dir):
    db = db_session.SessionLocal()
    try:
        profile = DBVoiceProfile(id=str(uuid.uuid4()), name="p", language="en")
        db.add(profile)
        db.commit()
        gen = _generation(db, profile.id, days_old=0, data_dir=data_dir)
        db.query(DBGeneration).filter_by(id=gen.id).update({"created_at": _old(1)})
        db.commit()
        fresh_orphan = data_dir / "generations" / "writing.wav"
        _touch(fresh_orphan, days_old=0)

        report = await retention.prune(db, days=0)

        assert report.generations == 1
        assert report.orphan_files == 0
        assert fresh_orphan.exists()
    finally:
        db.close()
