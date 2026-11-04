"""Bounded reusable speech settings and partial update schemas."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .utils.speech_text import validate_hints, validate_pronunciations


class SpeechSettingsUpdate(BaseModel):
    """Omitted fields retain their values; null resets one field to its empty default."""

    model_config = ConfigDict(extra="forbid")

    pronunciation_replacements: dict[str, str] | None = None
    stt_prompt: str | None = Field(None, max_length=4096)
    stt_hotwords: list[str] | None = None

    @field_validator("pronunciation_replacements")
    @classmethod
    def check_pronunciations(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        """Share request and saved-dictionary limits."""
        return validate_pronunciations(value)

    @field_validator("stt_hotwords")
    @classmethod
    def check_hotwords(cls, value: list[str] | None) -> list[str] | None:
        """Bound individual terms and their combined size."""
        validate_hints(None, value)
        return value


class SpeechSettingsResponse(SpeechSettingsUpdate):
    """A JSON settings document that can be exported and PUT back unchanged."""

    pronunciation_replacements: dict[str, str] = Field(default_factory=dict)
    stt_prompt: str = ""
    stt_hotwords: list[str] = Field(default_factory=list)
