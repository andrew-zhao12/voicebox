"""Bounded vocabulary hints and literal pronunciation replacements for speech APIs."""

import re

MAX_PROMPT_CHARS = 4096
MAX_HOTWORDS = 100
MAX_HOTWORD_CHARS = 100
MAX_REPLACEMENTS = 100
MAX_SPEECH_CHARS = 50000


def transcription_prompt(prompt: str | None, hotwords: list[str] | tuple[str, ...] | None) -> str | None:
    """Combine context and vocabulary for recognizers that accept a text prompt."""
    validate_hints(prompt, hotwords)
    vocabulary = ", ".join(hotwords or ())
    return "\n".join(part for part in (prompt, vocabulary) if part) or None


def validate_hints(prompt: str | None, hotwords: list[str] | tuple[str, ...] | None) -> None:
    """Reject oversized or malformed hints before inference or model downloads."""
    if prompt is not None and (not isinstance(prompt, str) or len(prompt) > MAX_PROMPT_CHARS):
        raise ValueError(f"prompt must be text of at most {MAX_PROMPT_CHARS} characters")
    if hotwords is not None:
        if not isinstance(hotwords, (list, tuple)) or len(hotwords) > MAX_HOTWORDS:
            raise ValueError(f"hotwords must be a list of at most {MAX_HOTWORDS} terms")
        if any(not isinstance(word, str) or not word.strip() or len(word) > MAX_HOTWORD_CHARS for word in hotwords):
            raise ValueError(f"each hotword must contain 1-{MAX_HOTWORD_CHARS} characters")
        if sum(map(len, hotwords)) > MAX_PROMPT_CHARS:
            raise ValueError(f"hotwords must total at most {MAX_PROMPT_CHARS} characters")


def validate_pronunciations(replacements: dict[str, str] | None) -> dict[str, str] | None:
    """Validate a request dictionary; replacements are literal text, never regex or SSML."""
    if replacements is None:
        return None
    if not isinstance(replacements, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in replacements.items()
    ):
        raise ValueError("pronunciation replacements must map text to text")
    if len(replacements) > MAX_REPLACEMENTS:
        raise ValueError(f"at most {MAX_REPLACEMENTS} pronunciation replacements are allowed")
    for source, spoken in replacements.items():
        if not source.strip() or len(source) > 128 or not spoken.strip() or len(spoken) > 256:
            raise ValueError("pronunciation keys need 1-128 characters and values need 1-256 characters")
    if sum(len(key) + len(value) for key, value in replacements.items()) > 8192:
        raise ValueError("pronunciation replacements must total at most 8192 characters")
    return replacements


def apply_pronunciations(text: str, replacements: dict[str, str] | None) -> str:
    """Replace whole literal terms once, case-sensitively, with longest matches first."""
    validate_pronunciations(replacements)
    if replacements:
        pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(key) for key in sorted(replacements, key=len, reverse=True)) + r")(?!\w)"
        )
        text = pattern.sub(lambda match: replacements[match.group()], text)
    if len(text) > MAX_SPEECH_CHARS:
        raise ValueError(f"pronunciation-expanded text exceeds {MAX_SPEECH_CHARS} characters")
    return text
