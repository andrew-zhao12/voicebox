"""Seed voice profiles at boot from a directory of bundles.

``VOICEBOX_SEED_PROFILES=/seed`` names a directory of ``.voicebox.zip``
bundles (``GET /profiles/{id}/export``, ``python -m backend.voices export``)
or unpacked bundle directories (a folder with ``manifest.json``).  At
startup every bundle whose profile name is not present yet is imported, so
replicas that share nothing but this directory offer the same voices; a
bundle whose name already exists is skipped, which makes restarts and
rolling updates idempotent.  ``GET /health/ready`` answers 503 until the
seed has been applied, and stays 503 when a bundle is invalid, so a broken
catalog is caught by the rollout instead of by an application.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .. import lifecycle

logger = logging.getLogger(__name__)

SEED_ENV = "VOICEBOX_SEED_PROFILES"
STEP_NAME = "seed_profiles"


def configured_dir(environ: Mapping[str, str] = os.environ) -> Path | None:
    """The seed directory from the environment, or ``None`` when unset."""
    raw = environ.get(SEED_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


@dataclass
class SeedReport:
    created: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        parts = [f"{len(self.created)} created"]
        if self.replaced:
            parts.append(f"{len(self.replaced)} replaced")
        parts.append(f"{len(self.skipped)} already present")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)


def bundle_paths(directory: Path) -> list[Path]:
    """Bundles in *directory*: ``*.zip`` files and folders holding a ``manifest.json``, sorted by name."""
    if not directory.is_dir():
        return []
    found: list[Path] = []
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith("."):
            continue
        is_zip = entry.is_file() and entry.suffix.lower() == ".zip"
        is_unpacked = entry.is_dir() and (entry / "manifest.json").is_file()
        if is_zip or is_unpacked:
            found.append(entry)
    return found


def read_bundle(path: Path) -> bytes:
    """The bundle's bytes; an unpacked directory is zipped in memory."""
    if path.is_file():
        return path.read_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for member in sorted(path.rglob("*")):
            if member.is_file():
                zip_file.write(member, member.relative_to(path).as_posix())
    return buffer.getvalue()


async def apply(directory: Path, db, *, on_conflict: str = "skip") -> SeedReport:
    """Import every bundle in *directory* into *db*; never raises for a single bad bundle."""
    from .export_import import import_profile_bundle  # lazy: pulls in librosa through the profile service

    report = SeedReport()
    for path in bundle_paths(directory):
        try:
            data = await asyncio.to_thread(read_bundle, path)
            result = await import_profile_bundle(data, db, on_conflict=on_conflict)  # type: ignore[arg-type]  # validated inside
        except Exception as e:
            db.rollback()
            report.failed[path.name] = str(e)
            logger.error("Seed bundle %s failed: %s", path.name, e)
            continue
        getattr(report, result.outcome).append(result.name)
    return report


async def run(directory: Path) -> SeedReport:
    """Apply the seed at startup and reflect the result in readiness."""
    from ..database import get_db

    lifecycle.step_begin(STEP_NAME, "importing")
    db = next(get_db())
    try:
        if not directory.is_dir():
            raise FileNotFoundError(f"{SEED_ENV}={directory} is not a directory")
        report = await apply(directory, db)
    except Exception as e:
        logger.error("Seeding voice profiles from %s failed: %s", directory, e)
        lifecycle.step_failed(STEP_NAME, str(e))
        return SeedReport(failed={str(directory): str(e)})
    finally:
        db.close()

    if report.ok:
        logger.info("Seeded voice profiles from %s: %s", directory, report.summary())
        lifecycle.step_done(STEP_NAME)
    else:
        logger.error("Seeded voice profiles from %s: %s", directory, report.summary())
        lifecycle.step_failed(STEP_NAME, "; ".join(f"{name}: {error}" for name, error in report.failed.items()))
    return report
