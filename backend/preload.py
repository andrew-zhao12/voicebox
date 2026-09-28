"""Download models ahead of time: ``python -m backend.preload kokoro whisper-turbo``.

Loads each model exactly the way the server would (downloading whatever is
missing into the HuggingFace cache, see ``VOICEBOX_MODELS_DIR``), fetches
the files an engine would otherwise download lazily after loading (Kokoro's
voice files), unloads it again and exits.  Meant for baking a Docker image
or filling a shared model volume, so replicas started with
``HF_HUB_OFFLINE=1`` find everything on disk.  ``--list`` prints the registry
names; ``--from-env`` reads ``VOICEBOX_PRELOAD_MODELS``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

# Imported first on purpose: ``backend.config`` maps VOICEBOX_MODELS_DIR onto
# HF_HUB_CACHE at import time, and huggingface_hub reads that variable once,
# when it is first imported (by the engine modules below).
from . import config  # noqa: F401 -- side effect: applies VOICEBOX_MODELS_DIR

logger = logging.getLogger(__name__)

# Files an engine downloads after load_model(), on first use, keyed by engine:
# (HuggingFace repo, glob patterns).  A replica running with HF_HUB_OFFLINE=1
# needs them in the cache as well.
LAZY_ASSETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "kokoro": ("hexgrad/Kokoro-82M", ("voices/*.pt",)),
}


def prefetch_lazy_assets(engine: str) -> int:
    """Download the lazily fetched files of *engine* into the cache; returns how many patterns were fetched."""
    assets = LAZY_ASSETS.get(engine)
    if assets is None:
        return 0
    from huggingface_hub import snapshot_download  # lazy: heavy import

    repo, patterns = assets
    snapshot_download(repo_id=repo, allow_patterns=list(patterns))
    return len(patterns)


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
            if prefetch_lazy_assets(cfg.engine):
                print(f"fetched the on-demand files of {cfg.engine}", flush=True)
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
