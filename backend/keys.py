"""Manage API keys from the command line.

Run from the repository root with the backend virtualenv::

    python -m backend.keys create --id myapp --role client
    python -m backend.keys list
    python -m backend.keys revoke --id myapp
    python -m backend.keys local      # print the local admin key
    python -m backend.keys path       # where the key files live

Pass ``--data-dir`` when the server does not use ``<cwd>/data``.  Only
``backend.config`` and ``backend.auth`` are imported, so this runs without torch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import config
from .auth.keystore import KeyStore, ensure_local_key_file, validate_limits
from .auth.settings import SecuritySettings


def _store(args: argparse.Namespace) -> KeyStore:
    if args.data_dir:
        config.set_data_dir(args.data_dir)
    settings = SecuritySettings.from_env()
    return KeyStore(env_key=settings.api_key_env, key_file=settings.key_file, keys_json=settings.keys_json)


def _parse_limits(items: Sequence[str]) -> dict[str, int | None]:
    limits: dict[str, int | None] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--limit expects name=value, got {item!r}")
        value = value.strip().lower()
        try:
            limits[name.strip()] = None if value in ("null", "none", "unlimited") else int(value)
        except ValueError:
            raise SystemExit(f"--limit {name}: expected an integer or 'unlimited'") from None
    try:
        return validate_limits(limits)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def cmd_create(args: argparse.Namespace) -> int:
    store = _store(args)
    try:
        record, key = store.create(args.id, args.role, _parse_limits(args.limit))
    except (KeyError, ValueError) as e:
        raise SystemExit(str(e.args[0] if e.args else e)) from None
    print(f"Created {record.role} key '{record.id}' in {store.keys_json_path}", file=sys.stderr)
    print("Store it now; it will not be shown again:", file=sys.stderr)
    print(key)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    store = _store(args)
    for record in store.list_keys():
        limits = record.key_limits().as_dict()
        summary = ", ".join(f"{name}={'unlimited' if value is None else value}" for name, value in limits.items())
        print(f"{record.id:<20} {record.role:<7} {record.source:<6} {record.created_at or '-':<26} {summary}")
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    store = _store(args)
    try:
        revoked = store.revoke(args.id)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if not revoked:
        raise SystemExit(f"No key with id '{args.id}' in {store.keys_json_path}")
    print(f"Revoked '{args.id}'", file=sys.stderr)
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    store = _store(args)
    print(f"local key file: {store.local_key_path or '(disabled: VOICEBOX_API_KEY is set)'}")
    print(f"key store:      {store.keys_json_path}")
    return 0


def cmd_local(args: argparse.Namespace) -> int:
    store = _store(args)
    path = store.local_key_path
    if path is None:
        raise SystemExit("VOICEBOX_API_KEY is set, so there is no local key file")
    print("This is the admin key for this data directory; keep it private.", file=sys.stderr)
    print(ensure_local_key_file(path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m backend.keys", description="Manage Voicebox API keys")
    parser.add_argument("--data-dir", help="Data directory of the server (default: ./data)")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create a key and print it once")
    create.add_argument("--id", required=True, help="key id: lowercase letters, digits, '-' or '_'")
    create.add_argument("--role", choices=("admin", "client"), default="client")
    create.add_argument(
        "--limit",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="per-minute override, e.g. requests=600 or inference=unlimited (repeatable)",
    )
    create.set_defaults(func=cmd_create)

    commands.add_parser("list", help="list keys (never prints secrets)").set_defaults(func=cmd_list)

    revoke = commands.add_parser("revoke", help="remove a key from the store")
    revoke.add_argument("--id", required=True)
    revoke.set_defaults(func=cmd_revoke)

    commands.add_parser("path", help="print where the key files live").set_defaults(func=cmd_path)
    commands.add_parser("local", help="print the local admin key").set_defaults(func=cmd_local)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
