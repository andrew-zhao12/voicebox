"""Load models at startup so the first request is served warm (``VOICEBOX_PRELOAD_MODELS``).

Each model is loaded as a job on the generation queue, so a preload never
overlaps a generation and cannot double-load with one; Whisper and the LLM
additionally take their inference slot.  ``GET /health/ready`` stays 503
until every configured model is resident.  A failed load is retried every
``RETRY_S`` seconds, so a transient download failure heals without a
restart.  Loading downloads whatever is missing through the backend's own
progress-tracked path, exactly like ``POST /models/download``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Mapping

from .. import lifecycle
from .inference_slots import llm_slot, whisper_slot
from .task_queue import PRELOAD_JOB_PREFIX, QueueFullError, enqueue_generation

logger = logging.getLogger(__name__)

ENV_VAR = "VOICEBOX_PRELOAD_MODELS"
RETRY_S = 120.0


def parse_model_names(value: str | None) -> list[str]:
    """Split a comma- or whitespace-separated list, dropping blanks and duplicates."""
    if not value:
        return []
    names: list[str] = []
    for part in value.replace(",", " ").split():
        if part not in names:
            names.append(part)
    return names


def configured_models(
    environ: Mapping[str, str] = os.environ,
    *,
    lookup: Callable[[str], object | None] | None = None,
) -> list[str]:
    """Model names from the environment that exist in the registry; unknown names are logged and skipped."""
    names = parse_model_names(environ.get(ENV_VAR))
    if not names:
        return []
    if lookup is None:
        from ..backends import get_model_config  # lazy: heavy import

        lookup = get_model_config
    known: list[str] = []
    unknown: list[str] = []
    for name in names:
        (known if lookup(name) is not None else unknown).append(name)
    if unknown:
        logger.warning("%s: ignoring unknown model name(s): %s", ENV_VAR, ", ".join(unknown))
    return known


async def _await_maybe(result) -> None:
    if asyncio.iscoroutine(result):
        await result


async def _load_one(name: str, done: asyncio.Event) -> None:
    """Queue job: load *name* (downloading first when needed) and record the outcome."""
    from ..backends import check_model_loaded, get_model_config, get_model_load_func  # lazy: heavy import

    try:
        cfg = get_model_config(name)
        if cfg is None:
            raise ValueError(f"unknown model {name}")
        if check_model_loaded(cfg):
            lifecycle.preload_ready(name)
            return
        lifecycle.preload_phase(name, "loading")
        load = get_model_load_func(cfg)
        if cfg.engine == "whisper":
            slot = whisper_slot
        elif cfg.engine == "qwen_llm":
            slot = llm_slot
        else:
            slot = None
        if slot is None:
            await _await_maybe(load())
        else:
            async with slot.acquire():
                await _await_maybe(load())
        lifecycle.preload_ready(name)
        logger.info("Preloaded %s", name)
    except asyncio.CancelledError:
        lifecycle.preload_failed(name, "cancelled")
        raise
    except Exception as e:
        logger.warning("Preload of %s failed: %s", name, e)
        lifecycle.preload_failed(name, str(e))
    finally:
        done.set()


async def run(names: list[str]) -> None:
    """Background task started from the lifespan.

    Returns once every model is resident, or when the server starts draining.
    """
    if not names:
        return
    lifecycle.preload_expect(names)
    remaining = list(names)
    attempt = 0
    while remaining and not lifecycle.is_draining():
        events: list[asyncio.Event] = []
        for name in remaining:
            done = asyncio.Event()
            try:
                enqueue_generation(f"{PRELOAD_JOB_PREFIX}{name}-{attempt}", _load_one(name, done))
            except QueueFullError as e:
                lifecycle.preload_failed(name, str(e))
                done.set()
            events.append(done)
        await asyncio.gather(*(event.wait() for event in events))
        failed = lifecycle.preload_state()["failed"]
        remaining = [name for name in remaining if name in failed]
        if remaining and not lifecycle.is_draining():
            attempt += 1
            logger.info("Retrying %d preload(s) in %.0f s: %s", len(remaining), RETRY_S, ", ".join(remaining))
            await asyncio.sleep(RETRY_S)
