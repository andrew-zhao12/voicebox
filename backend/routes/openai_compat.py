"""OpenAI-compatible audio API under ``/v1``.

``POST /v1/audio/speech`` streams synthesis in the format the client asks for,
``POST /v1/audio/transcriptions`` runs Whisper, ``GET /v1/models`` and
``GET /v1/voices`` list what this server can do, and ``POST``/``DELETE
/v1/voices`` let an application manage its own cloned voices.  Any OpenAI
SDK works with ``base_url="https://<host>/v1"`` and a client API key.
Errors under ``/v1`` use OpenAI's ``{"error": {...}}`` envelope (see
``backend/api_errors.py``); the handlers installed by
``install_openai_compat`` re-shape FastAPI's own ``HTTPException`` and
validation responses for these paths only.

A ``voice`` is resolved in this order: a profile id, a profile name, a
built-in preset voice id (``af_heart``, ``Ryan``, or qualified as
``kokoro:af_heart``), then OpenAI's stock names, which select the configured
default voice.  Profiles created by another client key are invisible.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import models
from ..api_errors import is_openai_path, openai_error
from ..auth import get_principal
from ..auth.principal import Principal
from ..database import VoiceProfile as DBVoiceProfile, get_db
from ..mcp_server.resolve import resolve_profile
from ..services import generation as generation_service, profiles
from ..utils import encode
from .transcription import ALLOWED_AUDIO_EXTS, UPLOAD_CHUNK_SIZE, transcribe_file, transcribe_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI compatible"])

# OpenAI's built-in voice names mean "the default voice" here: a per-client
# binding or the global default playback voice, whichever is configured.
STOCK_VOICES = frozenset(
    {"alloy", "ash", "ballad", "coral", "echo", "fable", "marin", "cedar", "nova", "onyx", "sage", "shimmer", "verse"}
)
# OpenAI model names that mean "the profile's default engine".
ALIAS_MODELS = frozenset({"tts-1", "tts-1-hd", "gpt-4o-mini-tts"})
WHISPER_ALIASES = {"whisper-1": None, "gpt-4o-transcribe": None, "gpt-4o-mini-transcribe": None}

SAMPLE_MAX_BYTES = 50 * 1024 * 1024
MAX_SAMPLES_PER_VOICE = 10


def fail(
    status: int, message: str, *, code: str | None = None, param: str | None = None, headers=None
) -> HTTPException:
    """An ``HTTPException`` whose detail is already the OpenAI envelope."""
    return HTTPException(
        status_code=status, detail=openai_error(status, message, code=code, param=param), headers=headers
    )


def _resolve_voice(voice: str, client_id: str | None, db: Session, principal: Principal, engine: str | None):
    """A visible profile row, a ``PresetVoice``, or a 404 in the envelope."""
    stock = voice.lower() in STOCK_VOICES
    if not stock:
        profile = resolve_profile(voice, client_id, db, principal=principal)
        if profile is not None:
            return profile
    preset = profiles.find_preset_voice(voice, engine)
    if preset is not None:
        return preset
    if stock:
        profile = resolve_profile(None, client_id, db, principal=principal)
        if profile is not None:
            return profile
        message = f"'{voice}' maps to the default voice, and none is configured; GET /v1/voices lists the voices."
    else:
        message = f"No voice named '{voice}'; GET /v1/voices lists the voices."
    raise fail(404, message, code="voice_not_found", param="voice")


def _resolve_model(model: str) -> tuple[str | None, str | None]:
    """``(engine, model_size)`` for an OpenAI alias, a registry model name or an engine name."""
    from ..backends import TTS_ENGINES, engine_has_model_sizes, get_model_config  # lazy: heavy import

    if not model or model in ALIAS_MODELS:
        return None, None
    cfg = get_model_config(model)
    if cfg is not None and cfg.engine in TTS_ENGINES:
        return cfg.engine, (cfg.model_size if engine_has_model_sizes(cfg.engine) else None)
    if model in TTS_ENGINES:
        return model, None
    raise fail(
        404,
        f"Unknown model '{model}'; GET /v1/models lists the available models.",
        code="model_not_found",
        param="model",
    )


@router.post("/audio/speech")
async def create_speech(data: models.OpenAISpeechRequest, request: Request, db: Session = Depends(get_db)):
    """Synthesize ``input`` in ``voice`` and stream it as ``response_format``.

    ``voice`` is a profile name or id, a built-in preset voice id, or one of
    OpenAI's stock names (the configured default voice); ``model`` is
    ``tts-1`` for the voice's default engine, or a model or engine name from
    ``GET /v1/models``.  ``wav`` and ``pcm`` stream chunk by chunk; the
    compressed formats stream when ffmpeg is installed and are otherwise
    sent once the clip is complete.  ``speed`` (0.25-4.0) is native on
    Kokoro and a pitch-preserving time stretch on every other engine.
    """
    principal = get_principal()
    formats = encode.available_formats()
    if data.response_format not in formats:
        raise fail(
            400,
            f"response_format '{data.response_format}' needs ffmpeg on the server; available: {', '.join(formats)}",
            code="unsupported_format",
            param="response_format",
        )

    engine, model_size = _resolve_model(data.model)
    voice = _resolve_voice(data.voice, request.headers.get("X-Voicebox-Client-Id"), db, principal, engine)
    preset = voice if isinstance(voice, profiles.PresetVoice) else None
    try:
        stream_request = models.StreamGenerationRequest(
            profile_id=voice.id,
            text=data.input,
            language=data.language or getattr(voice, "language", None) or "en",
            engine=engine,
            model_size=model_size,
            instruct=data.instructions,
            format="pcm",
            speed=data.speed,
        )
    except ValidationError as e:
        first = e.errors()[0] if e.errors() else {}
        raise fail(400, str(first.get("msg", "invalid request")), code="invalid_value") from e

    try:
        opened = await generation_service.open_stream(stream_request, db, principal, voice=preset)
    except generation_service.GenerationRefused as e:
        raise fail(e.status_code, e.detail, headers=e.headers) from e

    fmt = data.response_format
    body = encode.encode_stream(generation_service.stream_frames(opened), opened.sample_rate, fmt)
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Voicebox-Job-Id": opened.session.job_id,
        "X-Voicebox-Sample-Rate": str(opened.sample_rate),
        "X-Voicebox-Engine": opened.engine,
        "X-Voicebox-Voice": voice.id,
        "Content-Disposition": f'inline; filename="speech.{encode.EXTENSIONS[fmt]}"',
    }
    return StreamingResponse(body, media_type=encode.MEDIA_TYPES[fmt], headers=headers)


def _whisper_size(model: str) -> str | None:
    from ..backends import WHISPER_HF_REPOS  # lazy: heavy import

    if model in WHISPER_ALIASES:
        return None
    size = model.removeprefix("whisper-")
    if size in WHISPER_HF_REPOS:
        return size
    raise fail(
        404,
        f"Unknown model '{model}'; use whisper-1 or one of whisper-{', whisper-'.join(WHISPER_HF_REPOS)}.",
        code="model_not_found",
        param="model",
    )


@router.post("/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    response_format: str = Form("json"),
):
    """Transcribe an uploaded file with Whisper (``json`` or ``text``).

    ``whisper-1`` uses the server's current Whisper size; ``whisper-turbo``,
    ``whisper-large`` and the other registry sizes pick one explicitly.
    Timestamped formats (``srt``, ``vtt``, ``verbose_json``) are not available.
    """
    if response_format not in ("json", "text"):
        raise fail(
            400,
            f"response_format '{response_format}' is not supported; use json or text",
            code="unsupported_format",
            param="response_format",
        )
    text, _duration = await transcribe_upload(file, language, _whisper_size(model))
    if response_format == "text":
        return PlainTextResponse(text)
    return {"text": text}


@router.get("/models")
async def list_models():
    """TTS and STT models with their download and load state, plus the output formats this server can produce."""
    from ..backends import (  # lazy: heavy import
        check_model_loaded,
        engine_has_model_sizes,
        get_stt_model_configs,
        get_tts_backend_for_engine,
        get_tts_model_configs,
    )
    from ..services import transcribe

    data = []
    for cfg in get_tts_model_configs():
        backend = get_tts_backend_for_engine(cfg.engine)
        downloaded = (
            backend._is_model_cached(cfg.model_size)
            if engine_has_model_sizes(cfg.engine) or cfg.engine == "tada"
            else backend._is_model_cached()
        )
        data.append(_model_entry(cfg, "tts", downloaded, check_model_loaded(cfg)))
    whisper = transcribe.get_whisper_model()
    for cfg in get_stt_model_configs():
        data.append(_model_entry(cfg, "stt", whisper._is_model_cached(cfg.model_size), check_model_loaded(cfg)))
    return {"object": "list", "data": data, "formats": encode.available_formats()}


def _model_entry(cfg, kind: str, downloaded: bool, loaded: bool) -> dict:
    return {
        "id": cfg.model_name,
        "object": "model",
        "created": 0,
        "owned_by": "voicebox",
        "kind": kind,
        "engine": cfg.engine,
        "display_name": cfg.display_name,
        "languages": list(cfg.languages),
        "supports_instruct": cfg.supports_instruct,
        "downloaded": bool(downloaded),
        "loaded": bool(loaded),
    }


def _profile_entry(row) -> dict:
    owner = getattr(row, "owner_key_id", None)
    return {
        "id": row.id,
        "object": "voice",
        "kind": "profile",
        "name": row.name,
        "description": row.description,
        "language": row.language,
        "voice_type": getattr(row, "voice_type", None) or "cloned",
        "engine": getattr(row, "default_engine", None) or getattr(row, "preset_engine", None),
        "owner": owner,
        "shared": owner is None,
    }


def _preset_entry(preset: profiles.PresetVoice) -> dict:
    return {
        "id": preset.voice_id,
        "object": "voice",
        "kind": "preset",
        "name": preset.name,
        "description": preset.description,
        "language": preset.language,
        "voice_type": "preset",
        "engine": preset.engine,
        "gender": preset.gender,
        "owner": None,
        "shared": True,
    }


@router.get("/voices")
async def list_voices(db: Session = Depends(get_db)):
    """Everything ``voice`` accepts: the caller's visible profiles (``kind: profile``) and the built-in presets."""
    rows = await profiles.list_profiles(db, principal=get_principal())
    data = [_profile_entry(row) for row in sorted(rows, key=lambda r: r.name.lower())]
    data.extend(_preset_entry(preset) for preset in profiles.list_preset_voices())
    return {"object": "list", "data": data}


async def _save_sample(upload: UploadFile, index: int) -> str:
    ext = Path(upload.filename or "").suffix.lower()
    suffix = ext if ext in ALLOWED_AUDIO_EXTS else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        total = 0
        while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
            total += len(chunk)
            if total > SAMPLE_MAX_BYTES:
                Path(tmp.name).unlink(missing_ok=True)
                raise fail(
                    413,
                    f"file {index + 1} is larger than {SAMPLE_MAX_BYTES // (1024 * 1024)} MB",
                    code="file_too_large",
                    param="file",
                )
            tmp.write(chunk)
        return tmp.name


@router.post("/voices", status_code=201)
async def create_voice(
    name: str = Form(..., min_length=1, max_length=100),
    file: list[UploadFile] = File(..., description="1-10 reference recordings, 2-30 s each"),
    text: list[str] = Form([], description="transcript per file, in order; missing ones are transcribed with Whisper"),
    language: str = Form("en"),
    engine: str | None = Form(
        None, description="default cloning engine (qwen, luxtts, chatterbox, chatterbox_turbo, tada)"
    ),
    description: str | None = Form(None, max_length=500),
    db: Session = Depends(get_db),
):
    """Create a cloned voice owned by the calling key from one or more reference recordings.

    The voice is private to the key that created it (admins see everything);
    ``max_voices`` in the key's limits caps how many a key may hold.
    """
    principal = get_principal()
    if not file or len(file) > MAX_SAMPLES_PER_VOICE:
        raise fail(400, f"send between 1 and {MAX_SAMPLES_PER_VOICE} files", code="invalid_value", param="file")
    if engine and engine not in profiles.CLONING_ENGINES:
        raise fail(
            400,
            f"engine '{engine}' cannot clone voices; use one of {', '.join(sorted(profiles.CLONING_ENGINES))}",
            code="invalid_value",
            param="engine",
        )
    limit = principal.limits.max_voices
    if limit is not None and profiles.count_owned(db, principal.key_id) >= limit:
        raise fail(
            403,
            f"this key already holds {limit} voice(s); delete one or ask an admin to raise max_voices",
            code="voice_limit_reached",
        )
    try:
        create = models.VoiceProfileCreate(name=name, description=description, language=language, default_engine=engine)
    except ValidationError as e:
        first = e.errors()[0] if e.errors() else {}
        param = ".".join(str(part) for part in first.get("loc", ())) or None
        raise fail(400, str(first.get("msg", "invalid value")), code="invalid_value", param=param) from e

    transcripts = [t.strip() for t in text]
    paths: list[str] = []
    try:
        for index, upload in enumerate(file):
            paths.append(await _save_sample(upload, index))
        while len(transcripts) < len(paths):
            transcripts.append("")
        for index, path in enumerate(paths):
            if not transcripts[index]:
                transcripts[index], _duration = await transcribe_file(path, language, None)

        try:
            profile = await profiles.create_profile(create, db, owner_key_id=principal.key_id)
        except ValueError as e:
            status = 409 if "already exists" in str(e) else 400
            raise fail(status, str(e), code="voice_exists" if status == 409 else "invalid_value", param="name") from e
        try:
            for path, transcript in zip(paths, transcripts, strict=True):
                await profiles.add_profile_sample(profile.id, path, transcript, db)
        except ValueError as e:
            await profiles.delete_profile(profile.id, db)
            raise fail(400, str(e), code="invalid_value", param="file") from e
        except Exception:
            await profiles.delete_profile(profile.id, db)
            raise
    finally:
        for path in paths:
            Path(path).unlink(missing_ok=True)

    row = db.query(DBVoiceProfile).filter_by(id=profile.id).first()
    return _profile_entry(row)


@router.delete("/voices/{voice}")
async def delete_voice(voice: str, db: Session = Depends(get_db)):
    """Delete a voice this key created (admins may delete any profile); built-in presets cannot be deleted."""
    principal = get_principal()
    row = profiles.get_profile_orm_by_name_or_id(voice, db, principal)
    if row is None:
        raise fail(404, f"No voice named '{voice}' that this key may delete.", code="voice_not_found", param="voice")
    if not principal.is_admin and row.owner_key_id != principal.key_id:
        raise fail(
            403,
            "Only the key that created a voice can delete it; shared voices are managed by an admin.",
            code="voice_not_owned",
            param="voice",
        )
    await profiles.delete_profile(row.id, db)
    return {"id": row.id, "object": "voice", "deleted": True}


async def _http_exception(request: Request, exc: StarletteHTTPException):
    if not is_openai_path(request.url.path):
        return await http_exception_handler(request, exc)
    detail = exc.detail
    body = detail if isinstance(detail, dict) and "error" in detail else openai_error(exc.status_code, str(detail))
    return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))


async def _validation_error(request: Request, exc: RequestValidationError):
    if not is_openai_path(request.url.path):
        return await request_validation_exception_handler(request, exc)
    errors = exc.errors()
    first = errors[0] if errors else {}
    param = ".".join(str(part) for part in first.get("loc", ()) if part != "body") or None
    message = f"{param}: {first.get('msg', 'invalid value')}" if param else str(first.get("msg", "invalid request"))
    return JSONResponse(openai_error(400, message, code="invalid_value", param=param), status_code=400)


def install_openai_compat(app: FastAPI) -> None:
    """Register the ``/v1``-scoped exception handlers (the router is included by ``register_routers``)."""
    app.add_exception_handler(StarletteHTTPException, _http_exception)
    app.add_exception_handler(RequestValidationError, _validation_error)
