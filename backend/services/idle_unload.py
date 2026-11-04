"""Unload models nobody has used for a while (``VOICEBOX_MODEL_IDLE_UNLOAD_S``).

Off unless the variable is set.  Every sweep unloads TTS engines whose last
job finished more than the idle time ago and that have no job queued or
running, plus Whisper and the LLM when their slot has been idle as long.
Engines behind ``VOICEBOX_PRELOAD_MODELS`` are never unloaded, so the
readiness promise holds.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable, Mapping

from . import task_queue
from .inference_slots import InferenceBusyError, InferenceSlot, llm_slot, whisper_slot

logger = logging.getLogger(__name__)

ENV_VAR = "VOICEBOX_MODEL_IDLE_UNLOAD_S"
MIN_IDLE_S = 30.0


def configured_idle_s(environ: Mapping[str, str] = os.environ) -> float | None:
    raw = environ.get(ENV_VAR)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; idle unload disabled", ENV_VAR, raw)
        return None
    if value < MIN_IDLE_S:
        logger.warning("%s=%r is below %.0f s; idle unload disabled", ENV_VAR, raw, MIN_IDLE_S)
        return None
    return value


async def sweep(
    idle_s: float,
    protected: set[str],
    *,
    now: float | None = None,
    tts_backends: Callable[[str], object] | None = None,
    tts_engines: set[str] | None = None,
    stt_backends: Mapping[str, object] | None = None,
    llm_backend: object | None = None,
) -> list[str]:
    """Unload idle models once; returns the engine names unloaded.

    The keyword arguments exist so tests can inject fakes; the defaults
    reach the real registry (a heavy import, hence lazy).
    """
    now = time.monotonic() if now is None else now
    unloaded: list[str] = []

    if tts_backends is None or tts_engines is None:
        from ..backends import TTS_ENGINES, get_tts_backend_for_engine  # lazy: heavy import

        tts_backends = get_tts_backend_for_engine
        tts_engines = set(TTS_ENGINES)
    for engine, last_used in task_queue.engines_last_used().items():
        if engine in protected or engine not in tts_engines:
            continue
        if now - last_used < idle_s or task_queue.engine_in_use(engine):
            continue
        backend = tts_backends(engine)
        if backend.is_loaded():
            backend.unload_model()
            unloaded.append(engine)

    if stt_backends is None:
        from ..backends import STT_ENGINES, get_stt_backend_for_engine  # lazy: heavy import

        stt_backends = {engine: get_stt_backend_for_engine(engine) for engine in STT_ENGINES}
    if llm_backend is None:
        from . import llm as llm_service  # lazy: heavy import

        llm_backend = llm_service.get_llm_model()

    unloaded.extend(await _unload_slot(whisper_slot, stt_backends, protected, idle_s, now))
    unloaded.extend(await _unload_slot(llm_slot, {"qwen_llm": llm_backend}, protected, idle_s, now))

    if unloaded:
        logger.info("Idle unload: %s (unused for %.0f s)", ", ".join(unloaded), idle_s)
    return unloaded


async def _unload_slot(
    slot: InferenceSlot, backends: Mapping[str, object], protected: set[str], idle_s: float, now: float
) -> list[str]:
    if slot.busy or slot.last_used is None or now - slot.last_used < idle_s:
        return []

    async def unload() -> list[str]:
        # A request may have won the slot while we were waiting to acquire it.
        if slot.last_used is None or now - slot.last_used < idle_s:
            return []
        names = []
        for engine, backend in backends.items():
            # Native streams retain model state between inference calls, when the slot is idle.
            if engine in protected or getattr(backend, "active_streams", 0):
                continue
            if backend.is_loaded():
                backend.unload_model()
                names.append(engine)
        return names

    try:
        return await slot.run(unload)
    except InferenceBusyError:
        return []


async def run_loop(idle_s: float, protected: set[str]) -> None:
    """Background task: sweep at a fraction of the idle time, forever."""
    interval = max(15.0, min(60.0, idle_s / 2))
    while True:
        await asyncio.sleep(interval)
        try:
            await sweep(idle_s, protected)
        except Exception:
            logger.exception("Idle unload sweep failed")
