"""OpenAI-compatible audio API under ``/v1``.

``POST /v1/audio/speech`` streams synthesis in the format the client asks for,
``POST /v1/audio/transcriptions`` runs Whisper, ``GET /v1/models`` and
``GET /v1/voices`` list what this server can do.  Any OpenAI SDK works with
``base_url="https://<host>/v1"`` and a client API key.  Errors under ``/v1``
use OpenAI's ``{"error": {...}}`` envelope (see ``backend/api_errors.py``);
the handlers installed by ``install_openai_compat`` re-shape FastAPI's own
``HTTPException`` and validation responses for these paths only.
"""

from __future__ import annotations

import logging

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
from ..database import VoiceProfile as DBVoiceProfile, get_db
from ..mcp_server.resolve import resolve_profile
from ..services import generation as generation_service
from ..utils import encode
from .transcription import transcribe_upload

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


def fail(
    status: int, message: str, *, code: str | None = None, param: str | None = None, headers=None
) -> HTTPException:
    """An ``HTTPException`` whose detail is already the OpenAI envelope."""
    return HTTPException(
        status_code=status, detail=openai_error(status, message, code=code, param=param), headers=headers
    )


def _resolve_voice(voice: str, client_id: str | None, db: Session):
    explicit = None if voice.lower() in STOCK_VOICES else voice
    profile = resolve_profile(explicit, client_id, db)
    if profile is not None:
        return profile
    if explicit is None:
        message = (
            f"'{voice}' maps to the default voice, and none is configured; GET /v1/voices lists the voice profiles."
        )
    else:
        message = f"No voice profile named '{voice}'; GET /v1/voices lists the voice profiles."
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

    ``voice`` is a Voicebox profile name or id (OpenAI's stock names select the
    configured default voice); ``model`` is ``tts-1`` for the profile's default
    engine, or a model or engine name from ``GET /v1/models``.  ``wav`` and
    ``pcm`` stream chunk by chunk; the compressed formats stream when ffmpeg is
    installed and are otherwise sent once the clip is complete.
    """
    principal = get_principal()
    if data.speed != 1.0:
        raise fail(400, "speed other than 1.0 is not supported", code="unsupported_value", param="speed")
    formats = encode.available_formats()
    if data.response_format not in formats:
        raise fail(
            400,
            f"response_format '{data.response_format}' needs ffmpeg on the server; available: {', '.join(formats)}",
            code="unsupported_format",
            param="response_format",
        )

    profile = _resolve_voice(data.voice, request.headers.get("X-Voicebox-Client-Id"), db)
    engine, model_size = _resolve_model(data.model)
    try:
        stream_request = models.StreamGenerationRequest(
            profile_id=profile.id,
            text=data.input,
            language=data.language or getattr(profile, "language", None) or "en",
            engine=engine,
            model_size=model_size,
            instruct=data.instructions,
            format="pcm",
        )
    except ValidationError as e:
        first = e.errors()[0] if e.errors() else {}
        raise fail(400, str(first.get("msg", "invalid request")), code="invalid_value") from e

    try:
        opened = await generation_service.open_stream(stream_request, db, principal)
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


@router.get("/voices")
async def list_voices(db: Session = Depends(get_db)):
    """Voice profiles by name, for the ``voice`` field of ``/v1/audio/speech``."""
    rows = db.query(DBVoiceProfile).order_by(DBVoiceProfile.name).all()
    data = [
        {
            "id": row.id,
            "object": "voice",
            "name": row.name,
            "description": row.description,
            "language": row.language,
            "voice_type": getattr(row, "voice_type", None) or "cloned",
            "engine": getattr(row, "default_engine", None) or getattr(row, "preset_engine", None),
        }
        for row in rows
    ]
    return {"object": "list", "data": data}


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
