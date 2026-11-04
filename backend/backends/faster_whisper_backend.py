"""Optional CTranslate2 Whisper inference on CPU or CUDA, with word alignment."""

import asyncio
import importlib.util
import os
import threading
from dataclasses import asdict

from ..utils.subtitles import Transcript, segments_from_dicts
from .stt_capabilities import FASTER_WHISPER_REPOS


class FasterWhisperSTTBackend:
    """Run faster-whisper without importing its optional runtime at server startup."""

    def __init__(self) -> None:
        self.model = None
        self.model_size: str | None = None
        self._lock = threading.RLock()

    def _load(self, model_size: str) -> None:
        if model_size not in FASTER_WHISPER_REPOS:
            raise ValueError(f"Unknown faster-whisper model size: {model_size}")
        if self.model is not None and self.model_size == model_size:
            return
        try:
            import ctranslate2  # lazy: optional heavy import
            from faster_whisper import WhisperModel  # lazy: optional heavy import
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is unavailable. Install backend/requirements-faster-whisper.txt in the server environment."
            ) from exc
        device = os.environ.get("VOICEBOX_FASTER_WHISPER_DEVICE", "auto")
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        if device not in ("cpu", "cuda"):
            raise ValueError("VOICEBOX_FASTER_WHISPER_DEVICE must be auto, cpu, or cuda")
        compute_type = os.environ.get("VOICEBOX_FASTER_WHISPER_COMPUTE_TYPE") or (
            "float16" if device == "cuda" else "int8"
        )
        model = WhisperModel(
            FASTER_WHISPER_REPOS[model_size],
            device=device,
            compute_type=compute_type,
            cpu_threads=int(os.environ.get("VOICEBOX_REPLICA_THREADS", "4")),
            download_root=os.environ.get("HF_HUB_CACHE"),
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "").upper() in ("1", "ON", "YES", "TRUE"),
        )
        self.model = model
        self.model_size = model_size

    async def load_model(self, model_size: str = "base") -> None:
        """Load converted Whisper weights, retaining an existing model if loading fails."""

        def load() -> None:
            with self._lock:
                self._load(model_size)

        await asyncio.to_thread(load)

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        model_size: str | None = None,
        *,
        prompt: str | None = None,
        hotwords: list[str] | tuple[str, ...] | None = None,
        word_timestamps: bool = False,
    ) -> str:
        """Transcribe a recording with optional decoder vocabulary hints."""
        result = await self.transcribe_detailed(
            audio_path, language, model_size, prompt=prompt, hotwords=hotwords, word_timestamps=word_timestamps
        )
        return result.text

    async def transcribe_detailed(
        self,
        audio_path: str,
        language: str | None = None,
        model_size: str | None = None,
        *,
        prompt: str | None = None,
        hotwords: list[str] | tuple[str, ...] | None = None,
        word_timestamps: bool = False,
    ) -> Transcript:
        """Consume the lazy decoder inside its worker thread and preserve aligned words."""
        from ..utils.speech_text import validate_hints
        from ..utils.subtitles import words_from_dicts

        validate_hints(prompt, hotwords)

        def transcribe() -> Transcript:
            with self._lock:
                self._load(model_size or self.model_size or "base")
                results, info = self.model.transcribe(
                    audio_path,
                    language=language,
                    initial_prompt=prompt,
                    hotwords=", ".join(hotwords) if hotwords else None,
                    word_timestamps=word_timestamps,
                )
                results = list(results)
                segments = segments_from_dicts([asdict(segment) for segment in results], duration=info.duration)
                retained_ids = {segment.id for segment in segments}
                words = words_from_dicts(
                    (
                        asdict(word)
                        for segment in results
                        if segment.id in retained_ids
                        for word in (segment.words or ())
                        if word_timestamps
                    ),
                    duration=float(info.duration),
                )
                return Transcript(
                    text=" ".join(segment.text for segment in segments),
                    segments=segments,
                    language=info.language,
                    duration=float(info.duration),
                    words=words,
                )

        return await asyncio.to_thread(transcribe)

    def unload_model(self) -> None:
        """Release weights after any in-flight decoder has finished."""
        with self._lock:
            self.model = None
            self.model_size = None

    def is_loaded(self) -> bool:
        """Return whether converted weights are loaded."""
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        from .base import is_model_cached

        repo = FASTER_WHISPER_REPOS.get(model_size)
        return bool(repo and is_model_cached(repo, required_files=["config.json", "model.bin", "tokenizer.json"]))

    @staticmethod
    def runtime_available() -> bool:
        """Report whether the optional runtime is installed without importing it."""
        return importlib.util.find_spec("faster_whisper") is not None
