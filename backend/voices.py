"""Export and import voice profile bundles from the command line.

Run from the repository root with the backend virtualenv::

    python -m backend.voices export ./seed             # every profile, one .voicebox.zip each
    python -m backend.voices export ./seed --name Narrator --name "Support Bot"
    python -m backend.voices import ./seed             # what VOICEBOX_SEED_PROFILES does at boot
    python -m backend.voices import bundle.voicebox.zip --on-conflict replace
    python -m backend.voices list

Pass ``--data-dir`` when the server does not use ``<cwd>/data``.  The
bundles are the same files ``GET /profiles/{id}/export`` produces, so a
catalog designed in the UI can be exported here, committed next to the
deployment and applied to every replica through ``VOICEBOX_SEED_PROFILES``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def _safe_filename(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in name.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "profile"


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", help="the server's data directory (default: <cwd>/data or VOICEBOX_DATA_DIR)")
    parser = argparse.ArgumentParser(prog="python -m backend.voices", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", parents=[common], help="write one bundle per profile into a directory")
    export.add_argument("directory")
    export.add_argument("--name", action="append", default=[], help="only this profile (repeatable)")

    imp = sub.add_parser("import", parents=[common], help="import a bundle or every bundle in a directory")
    imp.add_argument("path")
    imp.add_argument(
        "--on-conflict",
        choices=("skip", "rename", "replace"),
        default="skip",
        help="when a profile with the bundle's name exists (default: skip)",
    )

    sub.add_parser("list", parents=[common], help="print the profiles in the database")
    return parser


def _open_db(data_dir: str | None):
    from . import config

    if data_dir:
        config.set_data_dir(data_dir)
    from .database import get_db, init_db

    init_db()
    return next(get_db())


def _export(args: argparse.Namespace) -> int:
    from .database import VoiceProfile
    from .services.export_import import export_profile_to_zip

    db = _open_db(args.data_dir)
    try:
        query = db.query(VoiceProfile).order_by(VoiceProfile.name)
        profiles = query.all()
        if args.name:
            wanted = {n.lower() for n in args.name}
            profiles = [p for p in profiles if p.name.lower() in wanted]
            missing = wanted - {p.name.lower() for p in profiles}
            for name in sorted(missing):
                print(f"no profile named {name!r}", file=sys.stderr)
        out_dir = Path(args.directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        failures = 0
        for profile in profiles:
            try:
                data = export_profile_to_zip(profile.id, db)
            except ValueError as e:
                print(f"skipped {profile.name!r}: {e}", file=sys.stderr)
                failures += 1
                continue
            target = out_dir / f"{_safe_filename(profile.name)}.voicebox.zip"
            target.write_bytes(data)
            print(f"{profile.name}: {target} ({len(data)} bytes)")
        return 1 if failures or (args.name and missing) else 0
    finally:
        db.close()


def _import(args: argparse.Namespace) -> int:
    from .services import seed_profiles
    from .services.export_import import import_profile_bundle

    db = _open_db(args.data_dir)
    try:
        path = Path(args.path)
        if path.is_file():
            result = asyncio.run(import_profile_bundle(path.read_bytes(), db, on_conflict=args.on_conflict))
            print(f"{result.outcome}: {result.name}")
            return 0
        report = asyncio.run(seed_profiles.apply(path, db, on_conflict=args.on_conflict))
        for name in report.created:
            print(f"created: {name}")
        for name in report.replaced:
            print(f"replaced: {name}")
        for name in report.skipped:
            print(f"already present: {name}")
        for bundle, error in report.failed.items():
            print(f"failed: {bundle}: {error}", file=sys.stderr)
        print(report.summary())
        return 0 if report.ok else 1
    finally:
        db.close()


def _list(args: argparse.Namespace) -> int:
    from .database import ProfileSample, VoiceProfile

    db = _open_db(args.data_dir)
    try:
        for profile in db.query(VoiceProfile).order_by(VoiceProfile.name).all():
            samples = db.query(ProfileSample).filter_by(profile_id=profile.id).count()
            kind = getattr(profile, "voice_type", None) or "cloned"
            detail = getattr(profile, "preset_voice_id", None) or f"{samples} sample(s)"
            owner = getattr(profile, "owner_key_id", None)
            print(f"{profile.name:32} {kind:8} {detail:20} {profile.id}{f'  owner={owner}' if owner else ''}")
        return 0
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    args = _parser().parse_args(argv)
    if args.command == "export":
        return _export(args)
    if args.command == "import":
        return _import(args)
    return _list(args)


if __name__ == "__main__":
    sys.exit(main())
