"""Voice profile bundles (``.voicebox.zip``) and generation exports.

A profile bundle is a ZIP with ``manifest.json``, ``samples.json`` (filename
to reference text), ``samples/<id>.wav`` and an optional ``avatar.<ext>``.
Manifest 1.0 carried only ``name``, ``description`` and ``language``, so a
bundle always imported as a cloned profile.  Manifest 1.1 adds the voice
type and everything that makes preset, designed and cloned profiles behave
the same on another server: ``voice_type``, ``preset_engine``,
``preset_voice_id``, ``design_prompt``, ``default_engine``, ``effects_chain``
and ``personality``, plus the profile ``id``, which an import keeps when no
other profile has it.  ``import_profile_bundle`` reads both versions,
validates every sample before it writes a row, and lets the caller decide
what a name clash means (``rename`` for the UI, ``skip`` for boot-time
seeding, ``replace`` to update a catalog in place).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy.orm import Session

from .. import config
from ..database import (
    Generation as DBGeneration,
    GenerationVersion as DBGenerationVersion,
    ProfileSample as DBProfileSample,
    VoiceProfile as DBVoiceProfile,
)
from ..models import VoiceProfileCreate, VoiceProfileResponse
from ..utils.clock import utcnow
from .profiles import add_profile_sample, create_profile, delete_profile

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "1.1"
OnConflict = Literal["rename", "skip", "replace"]
ON_CONFLICT_MODES: tuple[str, ...] = ("rename", "skip", "replace")

# Profile columns that travel in a 1.1 manifest, in addition to name/description/language.
_PROFILE_FIELDS = ("voice_type", "preset_engine", "preset_voice_id", "design_prompt", "default_engine", "personality")


@dataclass(frozen=True)
class ImportResult:
    """What ``import_profile_bundle`` did: the profile (``None`` when skipped) and how."""

    profile: VoiceProfileResponse | None
    outcome: Literal["created", "skipped", "replaced"]
    name: str


def _get_unique_profile_name(name: str, db: Session) -> str:
    """Append ``(n)`` until the name is free."""
    base_name = name
    counter = 1
    while db.query(DBVoiceProfile).filter_by(name=name).first() is not None:
        name = f"{base_name} ({counter})"
        counter += 1
    return name


def _effects_chain_of(profile: DBVoiceProfile) -> list | None:
    if not profile.effects_chain:
        return None
    try:
        chain = json.loads(profile.effects_chain)
    except ValueError:
        return None
    return chain if isinstance(chain, list) else None


def export_profile_to_zip(profile_id: str, db: Session) -> bytes:
    """Export a voice profile as a bundle (manifest 1.1).

    Raises ``ValueError`` when the profile does not exist, when a cloned
    profile has no samples, or when a sample file is missing on disk.
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    voice_type = getattr(profile, "voice_type", None) or "cloned"
    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()
    if voice_type == "cloned" and not samples:
        raise ValueError(f"Profile {profile_id} has no samples")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        has_avatar = False
        if profile.avatar_path:
            avatar_path = config.resolve_storage_path(profile.avatar_path)
            if avatar_path is not None and avatar_path.exists():
                has_avatar = True
                zip_file.write(avatar_path, f"avatar{avatar_path.suffix}")

        profile_data = {
            "id": profile.id,
            "name": profile.name,
            "description": profile.description,
            "language": profile.language,
            "voice_type": voice_type,
            "preset_engine": getattr(profile, "preset_engine", None),
            "preset_voice_id": getattr(profile, "preset_voice_id", None),
            "design_prompt": getattr(profile, "design_prompt", None),
            "default_engine": getattr(profile, "default_engine", None),
            "personality": getattr(profile, "personality", None),
            "effects_chain": _effects_chain_of(profile),
        }
        manifest = {"version": MANIFEST_VERSION, "profile": profile_data, "has_avatar": has_avatar}
        zip_file.writestr("manifest.json", json.dumps(manifest, indent=2))

        samples_data: dict[str, str] = {}
        for sample in samples:
            audio_path = config.resolve_storage_path(sample.audio_path)
            if audio_path is None or not audio_path.exists():
                raise ValueError(f"Audio file not found: {sample.audio_path}")
            zip_file.write(audio_path, f"samples/{audio_path.name}")
            samples_data[audio_path.name] = sample.reference_text
        zip_file.writestr("samples.json", json.dumps(samples_data, indent=2))

    zip_buffer.seek(0)
    return zip_buffer.read()


def _read_bundle(zip_file: zipfile.ZipFile) -> tuple[dict, dict[str, str]]:
    """``(profile fields, samples mapping)`` after structural validation."""
    namelist = set(zip_file.namelist())
    if "manifest.json" not in namelist:
        raise ValueError("ZIP archive missing manifest.json")
    if "samples.json" not in namelist:
        raise ValueError("ZIP archive missing samples.json")

    manifest = json.loads(zip_file.read("manifest.json"))
    if not isinstance(manifest, dict) or "version" not in manifest:
        raise ValueError("Invalid manifest.json: missing version")
    profile_data = manifest.get("profile")
    if not isinstance(profile_data, dict):
        raise ValueError("Invalid manifest.json: missing profile")

    samples_data = json.loads(zip_file.read("samples.json"))
    if not isinstance(samples_data, dict):
        raise ValueError("Invalid samples.json: must be a dictionary")
    for filename, reference_text in samples_data.items():
        if not filename.endswith(".wav"):
            raise ValueError(f"Invalid sample filename: {filename} (must be .wav)")
        if f"samples/{filename}" not in namelist:
            raise ValueError(f"Sample file not found in ZIP: samples/{filename}")
        if not isinstance(reference_text, str):
            raise ValueError(f"Invalid samples.json: reference text for {filename} must be a string")
    return profile_data, samples_data


def _profile_create_from(profile_data: dict, name: str) -> VoiceProfileCreate:
    fields = {
        "name": name,
        "description": profile_data.get("description"),
        "language": profile_data.get("language") or "en",
    }
    for field in _PROFILE_FIELDS:
        value = profile_data.get(field)
        if value is not None:
            fields[field] = value
    try:
        return VoiceProfileCreate(**fields)
    except ValueError as e:
        raise ValueError(f"Invalid profile in manifest.json: {e}") from e


def _validate_effects_chain(chain: object) -> str | None:
    if chain is None:
        return None
    if not isinstance(chain, list):
        raise ValueError("Invalid manifest.json: effects_chain must be a list")
    from ..utils.effects import validate_effects_chain  # lazy: imports pedalboard

    error = validate_effects_chain(chain)
    if error:
        raise ValueError(f"Invalid effects chain in manifest.json: {error}")
    return json.dumps(chain)


async def import_profile_bundle(file_bytes: bytes, db: Session, *, on_conflict: OnConflict = "rename") -> ImportResult:
    """Import a bundle; every sample is validated before the profile row is written.

    ``on_conflict`` decides what happens when a profile with the bundle's
    name exists: ``rename`` creates ``name (n)``, ``skip`` returns the
    existing state untouched, ``replace`` deletes the existing profile
    first.  Raises ``ValueError`` for a bad archive; a failure after the row
    was created removes the partial profile again.
    """
    if on_conflict not in ON_CONFLICT_MODES:
        raise ValueError(f"on_conflict must be one of {', '.join(ON_CONFLICT_MODES)}")
    from ..utils.audio import validate_and_load_reference_audio  # lazy: librosa

    try:
        zip_file = zipfile.ZipFile(io.BytesIO(file_bytes), "r")
    except zipfile.BadZipFile as e:
        raise ValueError("Invalid ZIP file") from e

    with zip_file, tempfile.TemporaryDirectory(prefix="voicebox-import-") as tmp_dir:
        try:
            profile_data, samples_data = _read_bundle(zip_file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in archive: {e}") from e

        original_name = str(profile_data.get("name") or "Imported Profile")
        effects_json = _validate_effects_chain(profile_data.get("effects_chain"))

        # Samples first: a bad file must not leave a half-imported profile behind.
        staged: list[tuple[Path, str]] = []
        for index, (filename, reference_text) in enumerate(samples_data.items()):
            path = Path(tmp_dir) / f"sample-{index}.wav"
            path.write_bytes(zip_file.read(f"samples/{filename}"))
            is_valid, error, _audio, _sr = await asyncio.to_thread(validate_and_load_reference_audio, str(path))
            if not is_valid:
                raise ValueError(f"Invalid sample {filename}: {error}")
            staged.append((path, reference_text))

        existing = db.query(DBVoiceProfile).filter_by(name=original_name).first()
        outcome: Literal["created", "skipped", "replaced"] = "created"
        name = original_name
        if existing is not None:
            if on_conflict == "skip":
                from .profiles import _profile_to_response

                return ImportResult(_profile_to_response(existing), "skipped", original_name)
            if on_conflict == "replace":
                await delete_profile(existing.id, db)
                outcome = "replaced"
            else:
                name = _get_unique_profile_name(original_name, db)

        # Keeping the bundle's id (when free) gives every replica seeded from
        # the same bundles the same ids, so a voice id works on any of them.
        profile = await create_profile(_profile_create_from(profile_data, name), db, profile_id=profile_data.get("id"))
        try:
            if effects_json is not None:
                row = db.query(DBVoiceProfile).filter_by(id=profile.id).first()
                row.effects_chain = effects_json
                row.updated_at = utcnow()
                db.commit()

            avatar_files = [f for f in zip_file.namelist() if f.startswith("avatar.")]
            if avatar_files:
                avatar_path = Path(tmp_dir) / avatar_files[0]
                avatar_path.write_bytes(zip_file.read(avatar_files[0]))
                try:
                    from .profiles import upload_avatar

                    await upload_avatar(profile.id, str(avatar_path), db)
                except Exception as e:  # the avatar is decoration; the voice still imports
                    logger.warning("Skipping the avatar of imported profile %r: %s", name, e)

            for path, reference_text in staged:
                await add_profile_sample(profile.id, str(path), reference_text, db)
        except Exception:
            await delete_profile(profile.id, db)
            raise

        from .profiles import get_profile

        return ImportResult(await get_profile(profile.id, db), outcome, name)


async def import_profile_from_zip(file_bytes: bytes, db: Session) -> VoiceProfileResponse:
    """The UI's import: a name clash creates ``name (n)``."""
    result = await import_profile_bundle(file_bytes, db, on_conflict="rename")
    assert result.profile is not None  # rename never skips
    return result.profile


def export_generation_to_zip(generation_id: str, db: Session) -> bytes:
    """Export a generation (all versions) to a ZIP archive."""
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not generation:
        raise ValueError(f"Generation {generation_id} not found")

    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()
    if not profile:
        raise ValueError(f"Profile {generation.profile_id} not found")

    versions = (
        db.query(DBGenerationVersion)
        .filter_by(generation_id=generation_id)
        .order_by(DBGenerationVersion.created_at)
        .all()
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        version_entries = []
        for v in versions:
            v_path = config.resolve_storage_path(v.audio_path)
            effects_chain = json.loads(v.effects_chain) if v.effects_chain else None
            version_entries.append(
                {
                    "id": v.id,
                    "label": v.label,
                    "is_default": v.is_default,
                    "effects_chain": effects_chain,
                    "filename": v_path.name if v_path is not None else None,
                }
            )

        manifest = {
            "version": "1.0",
            "generation": {
                "id": generation.id,
                "text": generation.text,
                "language": generation.language,
                "duration": generation.duration,
                "seed": generation.seed,
                "instruct": generation.instruct,
                "created_at": generation.created_at.isoformat(),
            },
            "profile": {
                "id": profile.id,
                "name": profile.name,
                "description": profile.description,
                "language": profile.language,
            },
            "versions": version_entries,
        }
        zip_file.writestr("manifest.json", json.dumps(manifest, indent=2))

        for v in versions:
            v_path = config.resolve_storage_path(v.audio_path)
            if v_path is not None and v_path.exists():
                zip_file.write(v_path, f"audio/{v_path.name}")

        if not versions:
            audio_path = config.resolve_storage_path(generation.audio_path)
            if audio_path is not None and audio_path.exists():
                zip_file.write(audio_path, f"audio/{audio_path.name}")

    zip_buffer.seek(0)
    return zip_buffer.read()


async def import_generation_from_zip(file_bytes: bytes, db: Session) -> dict:
    """Import a generation from a ZIP archive, attaching it to the named or any profile."""
    import shutil
    import uuid

    try:
        zip_file = zipfile.ZipFile(io.BytesIO(file_bytes), "r")
    except zipfile.BadZipFile as e:
        raise ValueError("Invalid ZIP file") from e

    with zip_file:
        namelist = zip_file.namelist()
        if "manifest.json" not in namelist:
            raise ValueError("ZIP archive missing manifest.json")
        try:
            manifest_data = json.loads(zip_file.read("manifest.json"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in archive: {e}") from e
        if "version" not in manifest_data:
            raise ValueError("Invalid manifest.json: missing version")
        if "generation" not in manifest_data:
            raise ValueError("Invalid manifest.json: missing generation data")

        generation_data = manifest_data["generation"]
        profile_data = manifest_data.get("profile", {})
        for field in ("text", "language", "duration"):
            if field not in generation_data:
                raise ValueError(f"Invalid manifest.json: missing generation.{field}")

        audio_files = [f for f in namelist if f.startswith("audio/") and f.endswith(".wav")]
        if not audio_files:
            raise ValueError("No audio file found in ZIP archive")

        profile_id = None
        profile_name = profile_data.get("name", "Unknown Profile")
        if profile_name and profile_name != "Unknown Profile":
            existing_profile = db.query(DBVoiceProfile).filter_by(name=profile_name).first()
            if existing_profile:
                profile_id = existing_profile.id
        if not profile_id:
            any_profile = db.query(DBVoiceProfile).first()
            if not any_profile:
                raise ValueError("No voice profiles found. Please create a profile before importing generations.")
            profile_id = any_profile.id
            profile_name = any_profile.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(zip_file.read(audio_files[0]))
            tmp_path = tmp.name
        try:
            generations_dir = config.get_generations_dir()
            generations_dir.mkdir(parents=True, exist_ok=True)
            new_generation_id = str(uuid.uuid4())
            audio_dest = generations_dir / f"{new_generation_id}.wav"
            shutil.copy(tmp_path, audio_dest)

            db_generation = DBGeneration(
                id=new_generation_id,
                profile_id=profile_id,
                text=generation_data["text"],
                language=generation_data["language"],
                audio_path=config.to_storage_path(audio_dest),
                duration=generation_data["duration"],
                seed=generation_data.get("seed"),
                instruct=generation_data.get("instruct"),
                created_at=utcnow(),
            )
            db.add(db_generation)
            db.commit()
            db.refresh(db_generation)
            return {
                "id": db_generation.id,
                "profile_id": profile_id,
                "profile_name": profile_name,
                "text": db_generation.text,
                "message": f"Generation imported successfully (assigned to profile: {profile_name})",
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)
