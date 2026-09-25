"""Download models ahead of time: ``python -m backend.preload kokoro whisper-turbo``.

Loads each model exactly the way the server would (downloading whatever is
missing into the HuggingFace cache, see ``VOICEBOX_MODELS_DIR``), unloads
it again and exits.  Meant for baking a Docker image or an init container so
``VOICEBOX_PRELOAD_MODELS`` finds everything on disk at boot.  ``--list``
prints the registry names; ``--from-env`` reads ``VOICEBOX_PRELOAD_MODELS``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m backend.preload", description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="model names from the registry (see --list)")
    parser.add_argument("--list", action="store_true", help="print every model name and exit")
    parser.add_argument("--from-env", action="store_true", help="also load the names in VOICEBOX_PRELOAD_MODELS")
    return parser


async def _load_all(names: list[str]) -> int:
    from .backends import get_model_config, get_model_load_func, unload_model_by_config  # lazy: heavy import

    failures = 0
    for name in names:
        cfg = get_model_config(name)
        if cfg is None:
            print(f"unknown model: {name}", file=sys.stderr)
            failures += 1
            continue
        print(f"loading {name} ({cfg.hf_repo_id})...", flush=True)
        try:
            result = get_model_load_func(cfg)()
            if asyncio.iscoroutine(result):
                await result
            unload_model_by_config(cfg)
            print(f"ready: {name}", flush=True)
        except Exception as e:
            print(f"failed: {name}: {e}", file=sys.stderr)
            failures += 1
    return failures


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    args = _parser().parse_args(argv)
    from .backends import get_all_model_configs  # lazy: heavy import
    from .services.preload import parse_model_names

    if args.list:
        for cfg in get_all_model_configs():
            print(f"{cfg.model_name:24} {cfg.display_name} ({cfg.hf_repo_id})")
        return 0
    names = list(args.names)
    if args.from_env:
        for name in parse_model_names(os.environ.get("VOICEBOX_PRELOAD_MODELS")):
            if name not in names:
                names.append(name)
    if not names:
        _parser().error("give at least one model name, --from-env or --list")
    failures = asyncio.run(_load_all(names))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
