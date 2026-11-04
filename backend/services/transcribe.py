"""STT selection shared by file transcription and live sessions."""

from ..backends import STTBackend, get_stt_backend

STT_ALIASES = frozenset({"whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"})


def get_whisper_model() -> STTBackend:
    """Return the platform's default Whisper backend (MLX or PyTorch)."""
    return get_stt_backend()


def resolve_stt(model: str | None):
    """Return ``(backend, model_size, config)``; keep legacy bare Whisper sizes valid."""
    from ..backends import WHISPER_HF_REPOS, get_stt_backend_for_engine, get_stt_model_configs

    configs = get_stt_model_configs()
    if not model or model in STT_ALIASES:
        backend = get_whisper_model()
        size = backend.model_size
        cfg = next(c for c in configs if c.engine == "whisper" and c.model_size == size)
        return backend, size, cfg
    name = f"whisper-{model}" if model in WHISPER_HF_REPOS else model
    cfg = next((c for c in configs if c.model_name == name), None)
    if cfg is None:
        raise ValueError(f"Unknown STT model '{model}'; GET /v1/models lists available model IDs")
    backend = get_whisper_model() if cfg.engine == "whisper" else get_stt_backend_for_engine(cfg.engine)
    return backend, cfg.model_size, cfg


def validate_options(cfg, *, prompt=None, hotwords=None, word_timestamps=False):
    """Reject unsupported inference options rather than silently ignoring them."""
    from ..utils.speech_text import validate_hints

    validate_hints(prompt, hotwords)
    for requested, capability, label in (
        (prompt, "supports_prompt", "prompt"),
        (hotwords, "supports_hotwords", "hotwords"),
        (word_timestamps, "supports_word_timestamps", "word timestamps"),
    ):
        if requested and not getattr(cfg, capability, cfg.engine == "whisper"):
            raise ValueError(f"{cfg.model_name} does not support {label}")


def unload_whisper_model():
    """Unload the default Whisper model to free memory."""
    get_whisper_model().unload_model()
