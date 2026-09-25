"""TTS generation endpoints."""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..auth import charge, get_principal
from ..database import Generation as DBGeneration, VoiceProfile as DBVoiceProfile, get_db
from ..services import history, personality, profiles
from ..services.generation import StreamSession, new_stream_session, run_generation, run_generation_stream
from ..services.inference_slots import InferenceBusyError, llm_slot
from ..services.task_queue import (
    QueueFullError,
    cancel_generation as cancel_generation_job,
    enqueue_generation,
    ensure_capacity,
)
from ..utils.audio import load_audio
from ..utils.tasks import get_task_manager
from ..utils.wav_stream import float_to_pcm16_bytes, streaming_wav_header

logger = logging.getLogger(__name__)

router = APIRouter()

IMPORTED_AUDIO_PROFILE_NAME = "Imported Audio"
IMPORT_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
IMPORT_AUDIO_MAX_BYTES = 200 * 1024 * 1024  # 200 MB


def _get_or_create_import_profile(db: Session) -> DBVoiceProfile:
    """Singleton profile every imported audio clip points at — keeps the
    Generation FK happy without making profile_id nullable across the schema."""
    row = db.query(DBVoiceProfile).filter(DBVoiceProfile.name == IMPORTED_AUDIO_PROFILE_NAME).first()
    if row is not None:
        return row
    row = DBVoiceProfile(
        id=str(uuid.uuid4()),
        name=IMPORTED_AUDIO_PROFILE_NAME,
        description="External audio imported into a story timeline.",
        language="en",
        voice_type="import",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _resolve_generation_engine(data: models.GenerationRequest, profile) -> str:
    return data.engine or getattr(profile, "default_engine", None) or getattr(profile, "preset_engine", None) or "qwen"


def _queue_full(error: QueueFullError) -> HTTPException:
    return HTTPException(status_code=429, detail=str(error), headers={"Retry-After": str(error.retry_after_s)})


def _busy(error: InferenceBusyError) -> HTTPException:
    return HTTPException(status_code=429, detail=str(error), headers={"Retry-After": str(error.retry_after_s)})


async def _rewrite_for_personality(data: models.GenerationRequest, profile) -> tuple[str, str]:
    """Return ``(text, source)``, rewriting through the profile's personality LLM when asked."""
    if not (data.personality and getattr(profile, "personality", None)):
        return data.text, "manual"
    try:
        async with llm_slot.acquire():
            llm_result = await personality.rewrite_as_profile(profile.personality, data.text)
    except InferenceBusyError as e:
        raise _busy(e) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    text = llm_result.text.strip()
    if not text:
        raise HTTPException(status_code=500, detail="LLM produced empty output; nothing to speak.")
    return text, "personality_speak"


def _resolve_effects_chain(data: models.GenerationRequest, db: Session) -> list | None:
    """Effects from the request, else the profile's saved default chain, else ``None``."""
    if data.effects_chain is not None:
        return [e.model_dump() for e in data.effects_chain]

    import json as _json

    profile_obj = db.query(DBVoiceProfile).filter_by(id=data.profile_id).first()
    if profile_obj and profile_obj.effects_chain:
        try:
            return _json.loads(profile_obj.effects_chain)
        except Exception:
            return None
    return None


@router.post("/generate", response_model=models.GenerationResponse)
async def generate_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate speech from text using a voice profile."""
    task_manager = get_task_manager()
    generation_id = str(uuid.uuid4())

    principal = get_principal()
    try:
        ensure_capacity(principal.key_id, principal.limits.max_pending_jobs)
    except QueueFullError as e:
        raise _queue_full(e) from e
    charge("tts_chars", len(data.text))

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    from ..backends import engine_has_model_sizes, ensure_model_cached_or_raise

    engine = _resolve_generation_engine(data, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_size = (data.model_size or "1.7B") if engine_has_model_sizes(engine) else None
    if not principal.is_admin:
        # Only admins may trigger a multi-gigabyte model download from the queue worker.
        await ensure_model_cached_or_raise(engine, model_size or "default")

    text, source = await _rewrite_for_personality(data, profile)

    generation = await history.create_generation(
        profile_id=data.profile_id,
        text=text,
        language=data.language,
        audio_path="",
        duration=0,
        seed=data.seed,
        db=db,
        instruct=data.instruct,
        generation_id=generation_id,
        status="generating",
        engine=engine,
        model_size=model_size if engine_has_model_sizes(engine) else None,
        source=source,
    )

    task_manager.start_generation(
        task_id=generation_id,
        profile_id=data.profile_id,
        text=text,
    )

    effects_chain_config = _resolve_effects_chain(data, db)

    try:
        enqueue_generation(
            generation_id,
            run_generation(
                generation_id=generation_id,
                profile_id=data.profile_id,
                text=text,
                language=data.language,
                engine=engine,
                model_size=model_size,
                seed=data.seed,
                normalize=data.normalize,
                effects_chain=effects_chain_config,
                instruct=data.instruct,
                mode="generate",
                max_chunk_chars=data.max_chunk_chars,
                crossfade_ms=data.crossfade_ms,
            ),
            owner=principal.key_id,
            max_pending=principal.limits.max_pending_jobs,
        )
    except QueueFullError as e:
        task_manager.complete_generation(generation_id)
        await history.update_generation_status(generation_id=generation_id, status="failed", db=db, error="Queue full")
        raise _queue_full(e) from e

    return generation


@router.post("/generate/{generation_id}/retry", response_model=models.GenerationResponse)
async def retry_generation(generation_id: str, db: Session = Depends(get_db)):
    """Retry a failed generation using the same parameters."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if (gen.status or "completed") != "failed":
        raise HTTPException(status_code=400, detail="Only failed generations can be retried")

    principal = get_principal()
    try:
        ensure_capacity(principal.key_id, principal.limits.max_pending_jobs)
    except QueueFullError as e:
        raise _queue_full(e) from e
    charge("tts_chars", len(gen.text or ""))

    gen.status = "generating"
    gen.error = None
    gen.audio_path = ""
    gen.duration = 0
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen.profile_id,
        text=gen.text,
    )

    try:
        enqueue_generation(
            generation_id,
            run_generation(
                generation_id=generation_id,
                profile_id=gen.profile_id,
                text=gen.text,
                language=gen.language,
                engine=gen.engine or "qwen",
                model_size=gen.model_size or "1.7B",
                seed=gen.seed,
                instruct=gen.instruct,
                mode="retry",
            ),
            owner=principal.key_id,
            max_pending=principal.limits.max_pending_jobs,
        )
    except QueueFullError as e:
        task_manager.complete_generation(generation_id)
        gen.status = "failed"
        gen.error = "Queue full"
        db.commit()
        raise _queue_full(e) from e

    return models.GenerationResponse.model_validate(gen)


@router.post(
    "/generate/{generation_id}/regenerate",
    response_model=models.GenerationResponse,
)
async def regenerate_generation(generation_id: str, db: Session = Depends(get_db)):
    """Re-run TTS with the same parameters and save the result as a new version."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation must be completed to regenerate")

    principal = get_principal()
    try:
        ensure_capacity(principal.key_id, principal.limits.max_pending_jobs)
    except QueueFullError as e:
        raise _queue_full(e) from e
    charge("tts_chars", len(gen.text or ""))

    gen.status = "generating"
    gen.error = None
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen.profile_id,
        text=gen.text,
    )

    version_id = str(uuid.uuid4())

    try:
        enqueue_generation(
            generation_id,
            run_generation(
                generation_id=generation_id,
                profile_id=gen.profile_id,
                text=gen.text,
                language=gen.language,
                engine=gen.engine or "qwen",
                model_size=gen.model_size or "1.7B",
                seed=gen.seed,
                instruct=gen.instruct,
                mode="regenerate",
                version_id=version_id,
            ),
            owner=principal.key_id,
            max_pending=principal.limits.max_pending_jobs,
        )
    except QueueFullError as e:
        task_manager.complete_generation(generation_id)
        gen.status = "completed"
        db.commit()
        raise _queue_full(e) from e

    return models.GenerationResponse.model_validate(gen)


@router.post("/generate/{generation_id}/cancel")
async def cancel_generation(generation_id: str, db: Session = Depends(get_db)):
    """Cancel a queued or running generation."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if (gen.status or "completed") not in ("loading_model", "generating"):
        raise HTTPException(status_code=400, detail="Only active generations can be cancelled")

    cancellation_state = cancel_generation_job(generation_id)
    if cancellation_state is None:
        # Row says active but the worker is no longer tracking it — the gen
        # coroutine exited without writing a terminal status (most often a
        # SQLite lock racing with the failed-status write inside the worker's
        # exception handler). Fail the row here so the user can move on.
        task_manager = get_task_manager()
        task_manager.complete_generation(generation_id)
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=db,
            error="Generation orphaned by worker",
        )
        return {"message": "Orphaned generation cleared"}

    if cancellation_state == "queued":
        task_manager = get_task_manager()
        task_manager.complete_generation(generation_id)
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=db,
            error="Generation cancelled",
        )
        return {"message": "Queued generation cancelled"}

    return {"message": "Generation cancellation requested"}


@router.get("/generate/{generation_id}/status")
async def get_generation_status(generation_id: str, db: Session = Depends(get_db)):
    """SSE endpoint that streams generation status updates."""
    import json

    async def event_stream():
        try:
            while True:
                db.expire_all()
                gen = db.query(DBGeneration).filter_by(id=generation_id).first()
                if not gen:
                    yield f"data: {json.dumps({'status': 'not_found', 'id': generation_id})}\n\n"
                    return

                payload = {
                    "id": gen.id,
                    "status": gen.status or "completed",
                    "duration": gen.duration,
                    "error": gen.error,
                    # Agent-originated sources ("mcp", "rest") skip main-window
                    # autoplay — the floating pill plays those directly.
                    "source": gen.source,
                }
                yield f"data: {json.dumps(payload)}\n\n"

                if (gen.status or "completed") in ("completed", "failed"):
                    return

                await asyncio.sleep(1)
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("SSE client disconnected for generation %s", generation_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _abandon_stream(session: StreamSession) -> None:
    """Stop a streaming job whose consumer went away (queued jobs are skipped, running ones cancelled)."""
    session.consumer_gone.set()
    cancel_generation_job(session.job_id)


@router.post("/generate/stream")
async def stream_speech(
    data: models.StreamGenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate speech and stream it as soon as the first sentence is ready.

    The text is synthesized one sentence chunk at a time inside the serial
    generation queue, and every chunk is sent as PCM16 the moment it exists.
    ``format=wav`` (default) prefixes a RIFF header whose length fields are
    ``0xFFFFFFFF`` (unknown length); ``format=pcm`` sends raw signed 16-bit
    little-endian mono samples.  The sample rate is reported in the
    ``X-Voicebox-Sample-Rate`` header.  Nothing is written to history.
    """
    from ..backends import engine_has_model_sizes, ensure_model_cached_or_raise
    from ..utils.effects import validate_effects_chain

    principal = get_principal()
    try:
        ensure_capacity(principal.key_id, principal.limits.max_pending_jobs)
    except QueueFullError as e:
        raise _queue_full(e) from e
    charge("tts_chars", len(data.text))

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    engine = _resolve_generation_engine(data, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_size = (data.model_size or "1.7B") if engine_has_model_sizes(engine) else None
    await ensure_model_cached_or_raise(engine, model_size or "default")

    effects_chain = _resolve_effects_chain(data, db)
    if effects_chain:
        error_msg = validate_effects_chain(effects_chain)
        if error_msg:
            raise HTTPException(status_code=400, detail=f"Invalid effects chain: {error_msg}")

    text, _source = await _rewrite_for_personality(data, profile)

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
        raise _queue_full(e) from e

    # Wait for the first playable chunk (or the failure) before sending
    # headers: early errors keep a real status code and the sample rate is
    # known for the WAV header.  The first chunk is the earliest playable
    # audio anyway, so this costs no latency.
    try:
        first = await session.frames.get()
    except asyncio.CancelledError:
        _abandon_stream(session)
        raise
    if first is None:
        raise HTTPException(status_code=500, detail="TTS produced no audio")
    if isinstance(first, ValueError):
        raise HTTPException(status_code=400, detail=str(first))
    if isinstance(first, Exception):
        logger.error("Stream %s failed before the first chunk: %s", session.job_id, first)
        raise HTTPException(status_code=500, detail="Speech synthesis failed; see the server log")
    first_audio, sample_rate = first

    async def body():
        finished = False
        try:
            head = streaming_wav_header(sample_rate) if data.format == "wav" else b""
            yield head + float_to_pcm16_bytes(first_audio)
            while True:
                item = await session.frames.get()
                if item is None:
                    finished = True
                    return
                if isinstance(item, Exception):
                    finished = True
                    logger.error("Stream %s ended early: %s", session.job_id, item)
                    return
                yield float_to_pcm16_bytes(item[0])
        finally:
            # Also runs on client disconnect (CancelledError) and generator
            # close, so an abandoned stream stops holding the queue.
            if not finished:
                _abandon_stream(session)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Voicebox-Stream-Mode": "chunked",
        "X-Voicebox-Job-Id": session.job_id,
        "X-Voicebox-Sample-Rate": str(sample_rate),
        "X-Voicebox-Channels": "1",
        "X-Voicebox-Sample-Format": "s16le",
    }
    if data.format == "wav":
        headers["Content-Disposition"] = 'inline; filename="speech.wav"'
    media_type = "audio/wav" if data.format == "wav" else "audio/pcm"
    return StreamingResponse(body(), media_type=media_type, headers=headers)


@router.post("/generate/import", response_model=models.GenerationResponse)
async def import_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Register an external audio file as a generation row.

    Designed for the story timeline so users can drop in music or other
    non-TTS audio. The row points at a singleton "Imported Audio" profile
    so the existing generation/story plumbing keeps working unchanged."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMPORT_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {sorted(IMPORT_AUDIO_EXTENSIONS)}",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > IMPORT_AUDIO_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {IMPORT_AUDIO_MAX_BYTES // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)
    audio_bytes = b"".join(chunks)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    generation_id = str(uuid.uuid4())
    target = config.get_generations_dir() / f"{generation_id}{suffix}"
    target.write_bytes(audio_bytes)

    try:
        audio, sr = load_audio(str(target))
        duration = float(len(audio) / sr) if sr else 0.0
    except Exception as decode_err:
        try:
            target.unlink()
        except OSError:
            pass
        logger.warning("Rejected audio import %s: %s", generation_id, decode_err)
        raise HTTPException(status_code=400, detail="Could not decode the audio file") from decode_err

    profile = _get_or_create_import_profile(db)
    display_name = Path(file.filename or "Imported audio").stem or "Imported audio"

    return await history.create_generation(
        profile_id=profile.id,
        text=display_name,
        language="en",
        audio_path=config.to_storage_path(target),
        duration=duration,
        seed=None,
        db=db,
        generation_id=generation_id,
        status="completed",
        engine="import",
        model_size=None,
        source="import",
    )
