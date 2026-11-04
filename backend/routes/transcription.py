"""Transcription endpoints."""

import asyncio
import logging
import tempfile
from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import models
from ..auth import get_principal
from ..services import transcribe
from ..services.inference_slots import InferenceBusyError, whisper_slot
from ..services.task_queue import create_background_task
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB

# Same set profiles.py accepts for voice samples. librosa picks its decoder from the
# file extension, so the temp file has to keep the uploaded one.
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".opus"}


@router.post("/transcribe", response_model=models.TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    model: str | None = Form(None),
    prompt: str | None = Form(None),
    hotwords: list[str] | None = Form(None),
):
    """Transcribe audio file to text."""
    text, duration = await transcribe_upload(file, language, model, prompt=prompt, hotwords=hotwords)
    return models.TranscriptionResponse(text=text, duration=duration)


async def _save_upload(file: UploadFile) -> str:
    """Write the upload to a temp file that keeps its extension (librosa picks the decoder from it)."""
    uploaded_ext = Path(file.filename or "").suffix.lower()
    file_suffix = uploaded_ext if uploaded_ext in ALLOWED_AUDIO_EXTS else ".wav"
    with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as tmp:
        try:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                tmp.write(chunk)
            return tmp.name
        except BaseException:
            Path(tmp.name).unlink(missing_ok=True)
            raise


async def transcribe_upload(file: UploadFile, language: str | None, model: str | None, **options) -> tuple[str, float]:
    """Decode an upload, run Whisper and return ``(text, duration_seconds)``.

    Shared by ``POST /transcribe`` and ``POST /v1/audio/transcriptions``;
    raises ``HTTPException`` for bad model names, missing models (409 for
    client keys, 202 while an admin's download runs) and decoder failures.
    """
    tmp_path = await _save_upload(file)
    try:
        return await transcribe_file(tmp_path, language, model, **options)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def transcribe_upload_detailed(file: UploadFile, language: str | None, model: str | None, **options):
    """Like ``transcribe_upload`` but returns a ``utils.subtitles.Transcript`` with segment timestamps."""
    tmp_path = await _save_upload(file)
    try:
        return await transcribe_file_detailed(tmp_path, language, model, **options)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def transcribe_file(tmp_path: str, language: str | None, model: str | None, **options) -> tuple[str, float]:
    """Run Whisper on an audio file already on disk (see ``transcribe_upload`` for the errors)."""
    return await _run_whisper(tmp_path, language, model, detailed=False, **options)


async def transcribe_file_detailed(tmp_path: str, language: str | None, model: str | None, **options):
    """Run Whisper on a file and return a ``Transcript`` with segments (errors as ``transcribe_upload``)."""
    return await _run_whisper(tmp_path, language, model, detailed=True, **options)


async def _run_whisper(
    tmp_path: str,
    language: str | None,
    model: str | None,
    *,
    detailed: bool,
    prompt: str | None = None,
    hotwords: list[str] | None = None,
    word_timestamps: bool = False,
):
    from ..utils.audio import load_audio, save_audio

    stt_path = None
    inference_owned = False
    try:
        try:
            backend, size, cfg = transcribe.resolve_stt(model)
            transcribe.validate_options(cfg, prompt=prompt, hotwords=hotwords, word_timestamps=word_timestamps)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if hasattr(backend, "runtime_available") and not backend.runtime_available():
            raise HTTPException(
                status_code=503,
                detail=f"{cfg.engine} runtime is not installed or configured; see the transcription setup guide",
            )
        already_loaded = backend.is_loaded() and backend.model_size == size
        if not already_loaded and not backend._is_model_cached(size):
            if not get_principal().is_admin:
                raise HTTPException(
                    status_code=409, detail=f"{cfg.model_name} is not downloaded; ask an admin to download it first."
                )
            task_manager = get_task_manager()

            async def download_background():
                try:
                    await whisper_slot.run(getattr(backend, "load_model_async", backend.load_model), size)
                    task_manager.complete_download(cfg.model_name)
                except Exception as e:
                    task_manager.error_download(cfg.model_name, str(e))

            task_manager.start_download(cfg.model_name)
            create_background_task(download_background())
            raise HTTPException(
                status_code=202,
                detail={
                    "message": f"{cfg.model_name} is being downloaded. Please wait and try again.",
                    "model_name": cfg.model_name,
                    "downloading": True,
                },
            )
        audio, sr = await _file_io(load_audio, tmp_path)
        duration = len(audio) / sr
        # Own a decoded WAV for the entire inference. A cancelled upload/socket can
        # delete its input while the shielded model thread is still using this copy.
        with tempfile.NamedTemporaryFile(suffix=".stt.wav", delete=False) as handle:
            stt_path = handle.name
        await _file_io(save_audio, audio, stt_path, sr)
        options = {}
        if prompt:
            options["prompt"] = prompt
        if hotwords:
            options["hotwords"] = hotwords
        if word_timestamps:
            options["word_timestamps"] = True

        async def infer_owned():
            try:
                fn = backend.transcribe_detailed if detailed else backend.transcribe
                return await fn(stt_path, language, size, **options)
            finally:
                Path(stt_path).unlink(missing_ok=True)

        def infer():
            # InferenceSlot calls this synchronously before scheduling the coroutine.
            # Transfer ownership now, including cancellation before its first step.
            nonlocal inference_owned
            inference_owned = True
            return infer_owned()

        try:
            result = await whisper_slot.run(infer)
        except InferenceBusyError as e:
            raise HTTPException(status_code=429, detail=str(e), headers={"Retry-After": str(e.retry_after_s)}) from e
        if detailed:
            return replace(result, duration=duration) if result.duration <= 0 else result
        return result, duration
    except HTTPException:
        raise
    except (ImportError, RuntimeError) as e:
        logger.exception("STT runtime unavailable")
        raise HTTPException(
            status_code=503, detail="STT runtime unavailable; check the server model configuration"
        ) from e
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail="Transcription failed") from e
    finally:
        if stt_path is not None and not inference_owned:
            Path(stt_path).unlink(missing_ok=True)


async def _file_io(fn, *args):
    """Finish file access before cancellation lets its caller remove the source/destination."""
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # File decoding/writing is bounded by the upload cap. Retaining this small
        # IO task avoids deleting a file a thread still needs or recreating a leak.
        try:
            await task
        except Exception:
            logger.debug("Cancelled transcription file IO failed", exc_info=True)
        raise
