"""Profile names are unique: creating or renaming onto an existing name fails with a clear message."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import Base
from backend.models import VoiceProfileCreate
from backend.services.profiles import create_profile, update_profile


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    monkeypatch.setattr(config, "get_profiles_dir", lambda: tmp_path / "profiles")
    try:
        yield session
    finally:
        session.close()


async def test_create_profile_duplicate_name_raises_error(test_db):
    first = await create_profile(VoiceProfileCreate(name="Test Profile", description="First", language="en"), test_db)
    assert first.name == "Test Profile"

    with pytest.raises(ValueError, match="already exists") as exc_info:
        await create_profile(VoiceProfileCreate(name="Test Profile", description="Second", language="en"), test_db)
    message = str(exc_info.value)
    assert "already exists" in message
    assert "Test Profile" in message
    assert "choose a different name" in message.lower()


async def test_create_profile_different_names_succeeds(test_db):
    one = await create_profile(VoiceProfileCreate(name="Profile One", language="en"), test_db)
    two = await create_profile(VoiceProfileCreate(name="Profile Two", language="en"), test_db)
    assert one.name == "Profile One"
    assert two.name == "Profile Two"
    assert one.id != two.id


async def test_update_profile_to_duplicate_name_raises_error(test_db):
    await create_profile(VoiceProfileCreate(name="Profile A", language="en"), test_db)
    second = await create_profile(VoiceProfileCreate(name="Profile B", language="en"), test_db)

    with pytest.raises(ValueError, match="already exists") as exc_info:
        await update_profile(second.id, VoiceProfileCreate(name="Profile A", language="en"), test_db)
    assert "already exists" in str(exc_info.value)
    assert "Profile A" in str(exc_info.value)


async def test_update_profile_keep_same_name_succeeds(test_db):
    profile = await create_profile(
        VoiceProfileCreate(name="My Profile", description="Original", language="en"), test_db
    )
    updated = await update_profile(
        profile.id, VoiceProfileCreate(name="My Profile", description="Updated", language="en"), test_db
    )
    assert updated is not None
    assert updated.id == profile.id
    assert updated.description == "Updated"


async def test_update_profile_to_new_unique_name_succeeds(test_db):
    profile = await create_profile(VoiceProfileCreate(name="Original Name", language="en"), test_db)
    updated = await update_profile(profile.id, VoiceProfileCreate(name="New Unique Name", language="en"), test_db)
    assert updated is not None
    assert updated.id == profile.id
    assert updated.name == "New Unique Name"


async def test_case_sensitive_names_allowed(test_db):
    """The unique constraint is case-sensitive, so 'test profile' and 'Test Profile' coexist."""
    lower = await create_profile(VoiceProfileCreate(name="test profile", language="en"), test_db)
    title = await create_profile(VoiceProfileCreate(name="Test Profile", language="en"), test_db)
    assert lower.id != title.id
