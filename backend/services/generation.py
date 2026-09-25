"""
Unified TTS generation orchestration.

Replaces the three near-identical closures (_run_generation, _run_retry,
_run_regenerate) that lived in main.py with a single ``run_generation()``
function parameterized by *mode*.

Mode differences:
  - "generate"   : full pipeline -- save clean version, optionally apply
                    effects and create a processed version.
  - "retry"      : re-runs a failed generation with the same seed.
                    No effects, no version creation.
  - "regenerate" : re-runs with seed=None for variation.  Creates a new
                    version with an auto-incremented "take-N" label.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from .. import config, models
from ..auth import charge
from ..auth.principal import Principal
from . import history, personality, profiles, task_queue
from .inference_slots import InferenceBusyError, llm_slot
from .task_queue import QueueFullError, cancel_generation, enqueue_generation, ensure_capacity
from ..database import get_db
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)


class GenerationRefused(Exception):
    """A request cannot be served; carries the HTTP status the route should answer with."""

    def __init__(self, status_code: int, detail: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


def resolve_engine(data: models.GenerationRequest, profile) -> str:
    """The request's engine, else the profile's default or preset engine, else qwen."""
    return data.engine or getattr(profile, "default_engine", None) or getattr(profile, "preset_engine", None) or "qwen"


async def rewrite_for_personality(data: models.GenerationRequest, profile) -> tuple[str, str]:
    """Return ``(text, source)``, rewriting through the profile's personality LLM when asked."""
    if not (data.personality and getattr(profile, "personality", None)):
        return data.text, "manual"
    try:
        async with llm_slot.acquire():
            llm_result = await personality.rewrite_as_profile(profile.personality, data.text)
    except InferenceBusyError as e:
        raise GenerationRefused(429, str(e), {"Retry-After": str(e.retry_after_s)}) from e
    except ValueError as e:
        raise GenerationRefused(400, str(e)) from e
    text = llm_result.text.strip()
    if not text:
        raise GenerationRefused(500, "LLM produced empty output; nothing to speak.")
    return text, "personality_speak"


def resolve_effects_chain(data: models.GenerationRequest, db) -> list | None:
    """Effects from the request, else the profile's saved default chain, else ``None``."""
    if data.effects_chain is not None:
        return [e.model_dump() for e in data.effects_chain]

    import json as _json

    from ..database import VoiceProfile as DBVoiceProfile

    profile_obj = db.query(DBVoiceProfile).filter_by(id=data.profile_id).first()
    if profile_obj and profile_obj.effects_chain:
        try:
            return _json.loads(profile_obj.effects_chain)
        except Exception:
            return None
    return None


@dataclass
class EnginePrep:
    """A loaded engine plus everything a generation needs from it."""

    backend: object
    voice_prompt: dict
    trim_fn: Callable | None
    runaway_detector: Callable | None
    runaway_cut_fn: Callable | None


async def prepare_engine(
    engine: str,
    model_size: str | None,
    profile_id: str,
    db,
    *,
    on_loading: Callable[[], Awaitable[None]] | None = None,
) -> EnginePrep:
    """Load the engine, build the profile's voice prompt and pick its post-processing.

    Shared by every generation path so model loading and voice-prompt
    creation always happen inside the queued job, serialized with all other
    GPU work.  *on_loading* is awaited before a model that is not resident
    yet gets loaded (the async path reports ``loading_model`` there).
    """
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..utils.audio import find_tts_runaway_cut, has_tts_runaway, trim_tts_output

    tts_model = get_tts_backend_for_engine(engine)
    if on_loading is not None and not tts_model.is_loaded():
        await on_loading()

    await load_engine_model(engine, model_size)

    voice_prompt = await profiles.create_voice_prompt_for_profile(
        profile_id,
        db,
        use_cache=True,
        engine=engine,
    )

    retries_runaway = engine_retries_runaway(engine)
    return EnginePrep(
        backend=tts_model,
        voice_prompt=voice_prompt,
        trim_fn=trim_tts_output if engine_needs_trim(engine) else None,
        runaway_detector=has_tts_runaway if retries_runaway else None,
        runaway_cut_fn=find_tts_runaway_cut if retries_runaway else None,
    )


async def run_generation(
    *,
    generation_id: str,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: Optional[int],
    normalize: bool = False,
    effects_chain: Optional[list] = None,
    instruct: Optional[str] = None,
    mode: Literal["generate", "retry", "regenerate"],
    max_chunk_chars: Optional[int] = None,
    crossfade_ms: Optional[int] = None,
    version_id: Optional[str] = None,
) -> None:
    """Execute TTS inference and persist the result.

    This is the single entry point for all background generation work.
    It is designed to be enqueued via ``services.task_queue.enqueue_generation``.
    """
    from ..utils.chunked_tts import generate_chunked
    from ..utils.audio import normalize_audio, save_audio

    task_manager = get_task_manager()
    bg_db = next(get_db())

    try:

        async def _mark_loading() -> None:
            await history.update_generation_status(generation_id, "loading_model", bg_db)

        prep = await prepare_engine(engine, model_size, profile_id, bg_db, on_loading=_mark_loading)

        await history.update_generation_status(generation_id, "generating", bg_db)

        gen_kwargs: dict = dict(
            language=language,
            seed=seed if mode != "regenerate" else None,
            instruct=instruct,
            trim_fn=prep.trim_fn,
            runaway_detector=prep.runaway_detector,
        )
        if max_chunk_chars is not None:
            gen_kwargs["max_chunk_chars"] = max_chunk_chars
        if crossfade_ms is not None:
            gen_kwargs["crossfade_ms"] = crossfade_ms

        audio, sample_rate = await generate_chunked(prep.backend, text, prep.voice_prompt, **gen_kwargs)

        # --- Normalize (generate and regenerate always; retry skips) -----
        if normalize or mode == "regenerate":
            audio = normalize_audio(audio)

        duration = len(audio) / sample_rate

        # --- Persist audio and update status -----------------------------
        if mode == "generate":
            final_path = _save_generate(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                effects_chain=effects_chain,
                save_audio=save_audio,
                db=bg_db,
            )
        elif mode == "retry":
            final_path = _save_retry(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                save_audio=save_audio,
            )
        elif mode == "regenerate":
            final_path = _save_regenerate(
                generation_id=generation_id,
                version_id=version_id,
                audio=audio,
                sample_rate=sample_rate,
                save_audio=save_audio,
                db=bg_db,
            )

        await history.update_generation_status(
            generation_id=generation_id,
            status="completed",
            db=bg_db,
            audio_path=final_path,
            duration=duration,
        )

    except asyncio.CancelledError:
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error="Generation cancelled",
        )
        _notify_speak_end(generation_id, status="cancelled")
    except Exception as e:
        traceback.print_exc()
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error=str(e),
        )
        _notify_speak_end(generation_id, status="failed")
    else:
        _notify_speak_end(generation_id, status="completed")
    finally:
        task_manager.complete_generation(generation_id)
        bg_db.close()


@dataclass
class StreamSession:
    """Channel between a queued streaming job and the HTTP response consuming it.

    ``frames`` carries ``(audio, sample_rate)`` tuples followed by either
    ``None`` (finished) or an ``Exception`` (failed).  The queue is unbounded
    on purpose: PCM chunks are small, and a slow client must never hold the
    GPU slot that the serial queue hands out.
    """

    job_id: str
    frames: asyncio.Queue = field(default_factory=asyncio.Queue)
    consumer_gone: asyncio.Event = field(default_factory=asyncio.Event)


def new_stream_session() -> StreamSession:
    """Create a session with a fresh ``stream-*`` job id."""
    return StreamSession(job_id=f"{task_queue.STREAM_JOB_PREFIX}{uuid.uuid4()}")


async def run_generation_stream(
    *,
    session: StreamSession,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str | None,
    seed: int | None,
    instruct: str | None,
    normalize: bool,
    effects_chain: list | None,
    max_chunk_chars: int,
    crossfade_ms: int,
    first_chunk_chars: int | None,
) -> None:
    """Queued job behind ``POST /generate/stream``.

    Synthesizes chunk by chunk and pushes playable audio into
    ``session.frames``.  Like :func:`run_generation` it runs inside the serial
    queue, so model loading, voice-prompt creation and inference never
    overlap other GPU work.  It writes no ``generations`` row; failures are
    delivered to the consumer instead of the worker.
    """
    from ..utils.audio import StreamingNormalizer
    from ..utils.chunked_tts import generate_chunked_stream
    from ..utils.effects import StreamingEffects

    frames = session.frames
    try:
        bg_db = next(get_db())
        try:
            prep = await prepare_engine(engine, model_size, profile_id, bg_db)
        finally:
            bg_db.close()

        normalizer = StreamingNormalizer() if normalize else None
        effects = StreamingEffects(effects_chain) if effects_chain else None

        async for audio, sample_rate in generate_chunked_stream(
            prep.backend,
            text,
            prep.voice_prompt,
            language=language,
            seed=seed,
            instruct=instruct,
            max_chunk_chars=max_chunk_chars,
            first_chunk_chars=first_chunk_chars,
            crossfade_ms=crossfade_ms,
            trim_fn=prep.trim_fn,
            runaway_detector=prep.runaway_detector,
            runaway_cut_fn=prep.runaway_cut_fn,
        ):
            if session.consumer_gone.is_set():
                break
            if normalizer is not None or effects is not None:
                audio = await asyncio.to_thread(_post_process_chunk, audio, sample_rate, normalizer, effects)
            frames.put_nowait((audio, sample_rate))
    except asyncio.CancelledError:
        frames.put_nowait(None)
        raise
    except Exception as e:
        logger.exception("Streaming generation %s failed", session.job_id)
        frames.put_nowait(e)
    else:
        frames.put_nowait(None)


def _post_process_chunk(audio, sample_rate: int, normalizer, effects):
    """Normalize then apply effects, in the same order as ``run_generation``."""
    if normalizer is not None:
        audio = normalizer.process(audio)
    if effects is not None:
        audio = effects.process(audio, sample_rate)
    return audio


def _notify_speak_end(generation_id: str, *, status: str) -> None:
    """Publish a speak-end event; the frontend ignores unknown ids."""
    try:
        from ..mcp_server import events as mcp_events

        mcp_events.publish(
            "speak-end",
            {"generation_id": generation_id, "status": status},
        )
    except Exception:
        # Never let event pub/sub break generation completion.
        pass


def _save_generate(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    effects_chain: Optional[list],
    save_audio,
    db,
) -> str:
    """Save clean version and optionally an effects-processed version.

    Returns the final audio path (processed if effects were applied,
    otherwise clean).
    """
    from . import versions as versions_mod

    clean_audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(clean_audio_path), sample_rate)

    has_effects = effects_chain and any(e.get("enabled", True) for e in effects_chain)

    versions_mod.create_version(
        generation_id=generation_id,
        label="original",
        audio_path=config.to_storage_path(clean_audio_path),
        db=db,
        effects_chain=None,
        is_default=not has_effects,
    )

    final_audio_path = str(clean_audio_path)

    if has_effects:
        from ..utils.effects import apply_effects, validate_effects_chain

        assert effects_chain is not None

        error_msg = validate_effects_chain(effects_chain)
        if error_msg:
            import logging
            logging.getLogger(__name__).warning("invalid effects chain, skipping: %s", error_msg)
            versions_mod.set_default_version(
                versions_mod.list_versions(generation_id, db)[0].id, db
            )
        else:
            processed_audio = apply_effects(audio, sample_rate, effects_chain)
            processed_path = config.get_generations_dir() / f"{generation_id}_processed.wav"
            save_audio(processed_audio, str(processed_path), sample_rate)
            final_audio_path = str(processed_path)
            versions_mod.create_version(
                generation_id=generation_id,
                label="version-2",
                audio_path=config.to_storage_path(processed_path),
                db=db,
                effects_chain=effects_chain,
                is_default=True,
            )

    return config.to_storage_path(final_audio_path)


def _save_retry(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    save_audio,
) -> str:
    """Save retry output -- single file, no versions.

    Returns the audio path.
    """
    audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(audio_path), sample_rate)
    return config.to_storage_path(audio_path)


@dataclass
class OpenedStream:
    """A streaming job that has produced its first playable chunk."""

    session: StreamSession
    first_audio: np.ndarray
    sample_rate: int
    engine: str
    text: str


def abandon_stream(session: StreamSession) -> None:
    """Stop a streaming job whose consumer went away (queued jobs are skipped, running ones cancelled)."""
    session.consumer_gone.set()
    cancel_generation(session.job_id)


async def open_stream(data: models.StreamGenerationRequest, db, principal: Principal) -> OpenedStream:
    """Validate a streamed request, queue its job and wait for the first chunk.

    Shared by ``POST /generate/stream`` and ``POST /v1/audio/speech``: caps and
    the ``tts_chars`` charge are applied here, the model must already be on
    disk, and the first chunk (or the failure) arrives before any response
    headers are sent, so early errors keep a real status code.  Raises
    ``GenerationRefused`` for anything the caller must answer with an error.
    """
    from fastapi import HTTPException

    from ..backends import engine_has_model_sizes, ensure_model_cached_or_raise
    from ..utils.effects import validate_effects_chain

    try:
        ensure_capacity(principal.key_id, principal.limits.max_pending_jobs)
    except QueueFullError as e:
        raise _refused_queue(e) from e
    charge("tts_chars", len(data.text))

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise GenerationRefused(404, "Profile not found")

    engine = resolve_engine(data, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise GenerationRefused(400, str(e)) from e

    model_size = (data.model_size or "1.7B") if engine_has_model_sizes(engine) else None
    try:
        await ensure_model_cached_or_raise(engine, model_size or "default")
    except HTTPException as e:
        raise GenerationRefused(e.status_code, str(e.detail)) from e

    effects_chain = resolve_effects_chain(data, db)
    if effects_chain:
        error_msg = validate_effects_chain(effects_chain)
        if error_msg:
            raise GenerationRefused(400, f"Invalid effects chain: {error_msg}")

    text, _source = await rewrite_for_personality(data, profile)

    first_chunk_chars = data.first_chunk_chars
    if first_chunk_chars is not None:
        first_chunk_chars = min(first_chunk_chars, data.max_chunk_chars)

    session = new_stream_session()
    try:
        enqueue_generation(
            session.job_id,
            run_generation_stream(
                session=session,
                profile_id=data.profile_id,
                text=text,
                language=data.language,
                engine=engine,
                model_size=model_size,
                seed=data.seed,
                instruct=data.instruct,
                normalize=data.normalize,
                effects_chain=effects_chain,
                max_chunk_chars=data.max_chunk_chars,
                crossfade_ms=data.crossfade_ms,
                first_chunk_chars=first_chunk_chars,
            ),
            owner=principal.key_id,
            max_pending=principal.limits.max_pending_jobs,
        )
    except QueueFullError as e:
        raise _refused_queue(e) from e

    try:
        first = await session.frames.get()
    except asyncio.CancelledError:
        abandon_stream(session)
        raise
    if first is None:
        raise GenerationRefused(500, "TTS produced no audio")
    if isinstance(first, ValueError):
        raise GenerationRefused(400, str(first))
    if isinstance(first, Exception):
        logger.error("Stream %s failed before the first chunk: %s", session.job_id, first)
        raise GenerationRefused(500, "Speech synthesis failed; see the server log")
    first_audio, sample_rate = first
    return OpenedStream(session=session, first_audio=first_audio, sample_rate=sample_rate, engine=engine, text=text)


def _refused_queue(error: QueueFullError) -> GenerationRefused:
    status = 503 if error.reason == "draining" else 429
    return GenerationRefused(status, str(error), {"Retry-After": str(error.retry_after_s)})


async def stream_frames(opened: OpenedStream) -> AsyncIterator[np.ndarray]:
    """Yield the first chunk, then every later one, abandoning the job if the consumer stops early."""
    session = opened.session
    finished = False
    try:
        yield opened.first_audio
        while True:
            item = await session.frames.get()
            if item is None:
                finished = True
                return
            if isinstance(item, Exception):
                finished = True
                logger.error("Stream %s ended early: %s", session.job_id, item)
                return
            yield item[0]
    finally:
        # Also runs on client disconnect (CancelledError) and generator
        # close, so an abandoned stream stops holding the queue.
        if not finished:
            abandon_stream(session)


def _save_regenerate(
    *,
    generation_id: str,
    version_id: Optional[str],
    audio,
    sample_rate: int,
    save_audio,
    db,
) -> str:
    """Save regeneration output as a new version with auto-label.

    Returns the audio path.
    """
    from . import versions as versions_mod

    import uuid as _uuid

    suffix = _uuid.uuid4().hex[:8]
    audio_path = config.get_generations_dir() / f"{generation_id}_{suffix}.wav"
    save_audio(audio, str(audio_path), sample_rate)

    # Count via DB query rather than list length to avoid TOCTOU race
    from ..database import GenerationVersion as DBGenerationVersion

    count = db.query(DBGenerationVersion).filter_by(generation_id=generation_id).count()
    label = f"take-{count + 1}"

    versions_mod.create_version(
        generation_id=generation_id,
        label=label,
        audio_path=config.to_storage_path(audio_path),
        db=db,
        effects_chain=None,
        is_default=True,
    )

    return config.to_storage_path(audio_path)
