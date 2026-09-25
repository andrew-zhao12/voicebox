"""Delete old generations, captures and orphaned audio (``VOICEBOX_RETENTION_DAYS``).

Off unless the variable is set, so the desktop app keeps everything.  With
it set, a daily sweep removes generations older than N days that are not
favourited and not placed in a story, captures older than N days, and files
under ``generations/`` and ``captures/`` that no row references (only when
they are at least one day old, so an in-flight write is never touched).
``POST /maintenance/prune`` runs the same sweep on demand.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import or_
from sqlalchemy.orm import Session

from .. import config
from ..database import (
    Capture as DBCapture,
    Generation as DBGeneration,
    GenerationVersion as DBGenerationVersion,
    StoryItem as DBStoryItem,
    get_db,
)

logger = logging.getLogger(__name__)

ENV_VAR = "VOICEBOX_RETENTION_DAYS"
INTERVAL_S = 24 * 3600.0
FIRST_RUN_DELAY_S = 60.0
ACTIVE_STATUSES = ("generating", "loading_model")


@dataclass
class PruneReport:
    days: int
    generations: int = 0
    captures: int = 0
    orphan_files: int = 0
    freed_bytes: int = 0
    errors: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def configured_days(environ: Mapping[str, str] = os.environ) -> int | None:
    """Retention in days, or None when unset (keep everything)."""
    raw = environ.get(ENV_VAR)
    if raw is None or not raw.strip():
        return None
    try:
        days = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; retention disabled", ENV_VAR, raw)
        return None
    if days < 0:
        logger.warning("%s=%r is negative; retention disabled", ENV_VAR, raw)
        return None
    return days


def _size(path: Path | None) -> int:
    try:
        return path.stat().st_size if path is not None else 0
    except OSError:
        return 0


def _referenced(db: Session, rows) -> set[Path]:
    paths: set[Path] = set()
    for (stored,) in rows:
        resolved = config.resolve_storage_path(stored)
        if resolved is not None:
            paths.add(resolved)
    return paths


def _delete_orphans(directory: Path, referenced: set[Path], older_than: datetime, report: PruneReport) -> None:
    if not directory.is_dir():
        return
    cutoff_ts = older_than.timestamp()
    for entry in directory.iterdir():
        try:
            if not entry.is_file() or entry.resolve() in referenced:
                continue
            if entry.stat().st_mtime >= cutoff_ts:
                continue
            size = entry.stat().st_size
            entry.unlink()
        except OSError:
            logger.exception("Could not remove orphaned file %s", entry)
            report.errors += 1
            continue
        report.orphan_files += 1
        report.freed_bytes += size


async def prune(db: Session, *, days: int, now: datetime | None = None) -> PruneReport:
    """Run one sweep and report what was removed."""
    from . import history

    now = now or datetime.utcnow()
    cutoff = now - timedelta(days=days)
    # Unreferenced files are invisible to the app, so they go after one day
    # whatever the retention: an in-flight generation writes its file before
    # its row points at it, and a day covers any such write.
    orphan_cutoff = now - timedelta(days=1)
    report = PruneReport(days=days)

    in_stories = {gen_id for (gen_id,) in db.query(DBStoryItem.generation_id).distinct()}
    old_generations = (
        db.query(DBGeneration)
        .filter(
            DBGeneration.created_at < cutoff,
            or_(DBGeneration.is_favorited.is_(False), DBGeneration.is_favorited.is_(None)),
        )
        .all()
    )
    for gen in old_generations:
        if gen.id in in_stories or (gen.status or "completed") in ACTIVE_STATUSES:
            continue
        freed = _size(config.resolve_storage_path(gen.audio_path))
        for version in db.query(DBGenerationVersion).filter_by(generation_id=gen.id).all():
            freed += _size(config.resolve_storage_path(version.audio_path))
        try:
            if await history.delete_generation(gen.id, db):
                report.generations += 1
                report.freed_bytes += freed
        except Exception:
            db.rollback()
            logger.exception("Could not delete generation %s", gen.id)
            report.errors += 1

    old_captures = db.query(DBCapture).filter(DBCapture.created_at < cutoff).all()
    for capture in old_captures:
        path = config.resolve_storage_path(capture.audio_path)
        freed = _size(path)
        try:
            if path is not None and path.exists():
                path.unlink()
            db.delete(capture)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Could not delete capture %s", capture.id)
            report.errors += 1
            continue
        report.captures += 1
        report.freed_bytes += freed

    generation_files = _referenced(db, db.query(DBGeneration.audio_path).filter(DBGeneration.audio_path != ""))
    generation_files |= _referenced(db, db.query(DBGenerationVersion.audio_path))
    _delete_orphans(config.get_generations_dir(), generation_files, orphan_cutoff, report)
    capture_files = _referenced(db, db.query(DBCapture.audio_path))
    _delete_orphans(config.get_captures_dir(), capture_files, orphan_cutoff, report)
    return report


async def run_loop(days: int) -> None:
    """Background task: sweep shortly after boot, then once a day."""
    await asyncio.sleep(FIRST_RUN_DELAY_S)
    while True:
        db = next(get_db())
        try:
            report = await prune(db, days=days)
            logger.info("Retention sweep: %s", report.as_dict())
        except Exception:
            logger.exception("Retention sweep failed")
        finally:
            db.close()
        await asyncio.sleep(INTERVAL_S)
