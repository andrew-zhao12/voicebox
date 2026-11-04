"""Dependency-free model identifiers and language coverage for optional STT engines."""

WHISPER_LANGUAGES = [
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
]

FASTER_WHISPER_REPOS = {
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large": "Systran/faster-whisper-large-v3",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

NEMOTRON_REPOS = {
    "en-0.6b": "nvidia/nemotron-speech-streaming-en-0.6b",
    "multilingual-0.6b": "nvidia/nemotron-3.5-asr-streaming-0.6b",
}

# NVIDIA distinguishes these 32 usable locales from eight adaptation-only locales
# that need fine-tuning. Tokenizer prompt keys alone overstate model coverage.
NEMOTRON_MULTILINGUAL_LOCALES = [
    "en-US",
    "en-GB",
    "es-US",
    "es-ES",
    "fr-FR",
    "fr-CA",
    "it-IT",
    "pt-BR",
    "pt-PT",
    "nl-NL",
    "de-DE",
    "tr-TR",
    "ru-RU",
    "ar-AR",
    "hi-IN",
    "ja-JP",
    "ko-KR",
    "vi-VN",
    "uk-UA",
    "pl-PL",
    "sv-SE",
    "cs-CZ",
    "nb-NO",
    "da-DK",
    "bg-BG",
    "fi-FI",
    "hr-HR",
    "sk-SK",
    "zh-CN",
    "hu-HU",
    "ro-RO",
    "et-EE",
]

_NEMOTRON_LANGUAGE_ALIASES = {locale.split("-")[0]: locale for locale in NEMOTRON_MULTILINGUAL_LOCALES}
_NEMOTRON_LANGUAGE_ALIASES.update({"en": "en-US", "es": "es-ES", "fr": "fr-FR", "pt": "pt-BR", "no": "nb-NO"})


def nemotron_language(language: str | None, model_size: str) -> str | None:
    """Validate language hints without accepting tokenizer-only unsupported locales."""
    if model_size not in NEMOTRON_REPOS:
        raise ValueError(f"Unknown Nemotron model size: {model_size}")
    if model_size == "en-0.6b":
        if language not in (None, "auto", "en", "en-US", "en-GB"):
            raise ValueError("Nemotron English supports English only; use the multilingual model for other languages")
        return "en"
    if language in (None, "auto"):
        return None
    resolved = _NEMOTRON_LANGUAGE_ALIASES.get(language, language)
    if resolved not in NEMOTRON_MULTILINGUAL_LOCALES:
        raise ValueError(f"Unsupported Nemotron transcription language: {language}")
    return resolved
