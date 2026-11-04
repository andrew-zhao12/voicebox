"""Voicebox TTS settings shared by the HTTP and incremental transports."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from pipecat.services.settings import TTSSettings
from pipecat.utils.types import NOT_GIVEN, NotGiven


@dataclass
class VoiceboxTTSSettings(TTSSettings):
    """Sparse Pipecat settings updates; omitted fields retain their current values."""

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    instructions: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pronunciation_replacements: dict[str, str] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    first_chunk_chars: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_chunk_chars: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_flush_chars: int | NotGiven = field(default_factory=lambda: NOT_GIVEN)


def pronunciation_file(filename: str | None) -> dict[str, str] | None:
    """Read a small literal pronunciation dictionary, failing before the call starts."""
    if not filename:
        return None
    path = Path(filename).expanduser()
    if path.stat().st_size > 65536:
        raise ValueError("VOICEBOX_PRONUNCIATIONS_FILE must be at most 64 KiB")
    result = json.loads(path.read_text())
    validate_pronunciations(result)
    return result


def validate_pronunciations(value):
    """Keep malformed or unbounded pronunciation maps out of speech requests."""
    if value is None:
        return
    if not isinstance(value, dict) or len(value) > 100:
        raise ValueError("Pronunciations must be a JSON object with at most 100 entries")
    total = 0
    for source, target in value.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Pronunciation keys and values must be strings")
        if not source.strip() or not target.strip() or len(source) > 128 or len(target) > 256:
            raise ValueError("Pronunciation terms must be nonempty and at most 128/256 characters")
        total += len(source) + len(target)
    if total > 8192:
        raise ValueError("Pronunciations must total at most 8192 characters")


def complete_settings(model, voice, settings=None) -> VoiceboxTTSSettings:
    """Build the complete settings store required by Pipecat 1.15."""
    result = VoiceboxTTSSettings(
        model=model,
        voice=voice,
        language=None,
        speed=None,
        instructions=None,
        pronunciation_replacements=None,
        first_chunk_chars=None,
        max_chunk_chars=None,
        text_flush_chars=120,
    )
    if settings:
        result.apply_update(settings)
    validate_settings(result)
    return result


def validate_settings(settings):
    """Validate complete settings before sending any audio or text."""
    settings.validate_complete()
    if (
        not settings.model
        or not settings.model.strip()
        or not settings.voice
        or not settings.voice.strip()
    ):
        raise ValueError("Voicebox voice and TTS model must not be empty")
    if settings.speed is not None and (
        isinstance(settings.speed, bool)
        or not math.isfinite(settings.speed)
        or not 0.25 <= settings.speed <= 4
    ):
        raise ValueError("VOICEBOX_TTS_SPEED must be between 0.25 and 4")
    if settings.instructions is not None and len(settings.instructions) > 500:
        raise ValueError("VOICEBOX_TTS_INSTRUCTIONS must be at most 500 characters")
    for value, name, low, high in (
        (settings.first_chunk_chars, "VOICEBOX_FIRST_CHUNK_CHARS", 20, 1000),
        (settings.max_chunk_chars, "VOICEBOX_MAX_CHUNK_CHARS", 100, 5000),
        (settings.text_flush_chars, "VOICEBOX_TEXT_FLUSH_CHARS", 20, 1000),
    ):
        if value is not None and (type(value) is not int or not low <= value <= high):
            raise ValueError(f"{name} must be an integer between {low} and {high}")
    validate_pronunciations(settings.pronunciation_replacements)


def speech_options(settings) -> dict:
    """Only send configured extensions, leaving Voicebox defaults intact."""
    return {
        name: value
        for name in (
            "language",
            "speed",
            "instructions",
            "pronunciation_replacements",
            "first_chunk_chars",
            "max_chunk_chars",
        )
        if (value := getattr(settings, name)) is not None
    }
