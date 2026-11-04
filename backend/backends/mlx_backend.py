"""
MLX backend implementation for TTS and STT using mlx-audio.
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing
from typing import Optional, List, Tuple
import asyncio
import logging
import threading
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# PATCH: Import and apply offline patch BEFORE any huggingface_hub usage
# This prevents mlx_audio from making network requests when models are cached
from ..utils.hf_offline_patch import patch_huggingface_hub_offline, ensure_original_qwen_config_cached

patch_huggingface_hub_offline()
ensure_original_qwen_config_cached()

from . import TTSBackend, STTBackend, LANGUAGE_CODE_TO_NAME, WHISPER_HF_REPOS
from .base import (
    is_model_cached,
    combine_voice_prompts as _combine_voice_prompts,
    iterate_in_thread,
    model_load_progress,
)
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt


class MLXTTSBackend:
    """MLX-based TTS backend using mlx-audio."""

    # Seconds of audio per streamed piece when mlx-audio supports ``stream=True``.
    STREAMING_INTERVAL_S = 1.0

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self._current_model_size = None

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        """
        Get the MLX model path.

        Args:
            model_size: Model size (1.7B or 0.6B)

        Returns:
            HuggingFace Hub model ID for MLX
        """
        mlx_model_map = {
            "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        }

        if model_size not in mlx_model_map:
            raise ValueError(f"Unknown model size: {model_size}")

        hf_model_id = mlx_model_map[model_size]
        logger.info("Will download MLX model from HuggingFace Hub: %s", hf_model_id)

        return hf_model_id

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(
            self._get_model_path(model_size),
            weight_extensions=(".safetensors", ".bin", ".npz"),
        )

    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the MLX TTS model.

        Args:
            model_size: Model size to load (1.7B or 0.6B)
        """
        if model_size is None:
            model_size = self.model_size

        # If already loaded with correct size, return
        if self.model is not None and self._current_model_size == model_size:
            return

        # Unload existing model if different size requested
        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()

        # Run blocking load in thread pool
        await asyncio.to_thread(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        model_path = self._get_model_path(model_size)
        model_name = f"qwen-tts-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from mlx_audio.tts import load

            logger.info("Loading MLX TTS model %s...", model_size)

            self.model = load(model_path)

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("MLX TTS model %s loaded successfully", model_size)

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None
            logger.info("MLX TTS model unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        MLX backend stores voice prompt as a dict with audio path and text.
        The actual voice prompt processing happens during generation.

        Args:
            audio_path: Path to reference audio file
            reference_text: Transcript of reference audio
            use_cache: Whether to use cached prompt if available

        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        await self.load_model_async(None)

        # Check cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cached_prompt = get_cached_voice_prompt(cache_key)
            if cached_prompt is not None:
                # Return cached prompt (should be dict format)
                if isinstance(cached_prompt, dict):
                    # Validate that the cached audio file still exists
                    cached_audio_path = cached_prompt.get("ref_audio") or cached_prompt.get("ref_audio_path")
                    if cached_audio_path and Path(cached_audio_path).exists():
                        return cached_prompt, True
                    else:
                        # Cached file no longer exists, invalidate cache
                        logger.warning("Cached audio file not found: %s, regenerating prompt", cached_audio_path)

        # MLX voice prompt format - store audio path and text
        # The model will process this during generation
        voice_prompt_items = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }

        # Cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cache_voice_prompt(cache_key, voice_prompt_items)

        return voice_prompt_items, False

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text using voice prompt.

        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary with ref_audio and ref_text
            language: Language code (en or zh) - may not be fully supported by MLX
            seed: Random seed for reproducibility
            instruct: Natural language instruction (may not be supported by MLX)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model_async(None)

        logger.info("Generating audio for text: %s", text)

        def _generate_sync():
            """Run synchronous generation in thread pool."""
            audio_chunks = []
            sample_rate = 24000
            for audio, sr in self._iter_generate_sync(text, voice_prompt, language, seed, stream=False):
                audio_chunks.append(audio)
                sample_rate = sr

            # Concatenate all chunks
            if audio_chunks:
                audio = np.concatenate(audio_chunks)
            else:
                # Fallback: empty audio
                audio = np.array([], dtype=np.float32)

            return audio, sample_rate

        # Run blocking inference in thread pool
        audio, sample_rate = await asyncio.to_thread(_generate_sync)

        return audio, sample_rate

    async def generate_stream(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Yield ``(audio, sample_rate)`` pieces while mlx-audio is still generating.

        Used by ``utils.chunked_tts.generate_chunked_stream`` for sub-sentence
        latency.  With an mlx-audio that accepts ``stream=True`` the pieces
        are about ``STREAMING_INTERVAL_S`` seconds long; otherwise one piece
        per text segment.
        """
        await self.load_model_async(None)

        logger.info("Streaming audio for text: %s", text)
        stop = threading.Event()
        pieces = iterate_in_thread(
            lambda: self._iter_generate_sync(text, voice_prompt, language, seed, stream=True),
            stop=stop,
        )
        async with aclosing(pieces):
            async for piece in pieces:
                yield piece

    def _iter_generate_sync(
        self,
        text: str,
        voice_prompt: dict,
        language: str,
        seed: int | None,
        *,
        stream: bool,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Yield ``(audio, sample_rate)`` per mlx-audio result, synchronously.

        mlx-audio's ``generate()`` is a generator that yields one result per
        text segment, or, with ``stream=True`` on versions that support it,
        sub-segment pieces as they are produced.  Meant to run in a worker
        thread.
        """
        import inspect

        lang = LANGUAGE_CODE_TO_NAME.get(language, "auto")

        # Set seed if provided (MLX uses numpy random)
        if seed is not None:
            import mlx.core as mx

            np.random.seed(seed)
            mx.random.seed(seed)

        # Extract voice prompt info
        ref_audio = voice_prompt.get("ref_audio") or voice_prompt.get("ref_audio_path")
        ref_text = voice_prompt.get("ref_text", "")

        # Validate that the audio file exists
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("Audio file not found: %s", ref_audio)
            logger.warning("This may be due to a cached voice prompt referencing a deleted temp file.")
            logger.warning("Regenerating without voice prompt.")
            ref_audio = None

        params = inspect.signature(self.model.generate).parameters
        kwargs: dict = {"lang_code": lang}
        if stream and "stream" in params:
            kwargs["stream"] = True
            if "streaming_interval" in params:
                kwargs["streaming_interval"] = self.STREAMING_INTERVAL_S
        if ref_audio and "ref_audio" in params:
            kwargs["ref_audio"] = ref_audio
            kwargs["ref_text"] = ref_text

        # Inference runs with the process's default HF_HUB_OFFLINE
        # state. Forcing offline here (previously used to avoid lazy
        # mlx_audio lookups hanging when the network drops mid-inference,
        # issue #462) regressed online users because libraries make
        # legitimate metadata calls during generation.
        produced = 0
        try:
            for result in self.model.generate(text, **kwargs):
                produced += 1
                yield np.asarray(result.audio, dtype=np.float32), int(result.sample_rate)
        except Exception as e:
            if produced or "ref_audio" not in kwargs:
                raise
            # Voice cloning failed before any audio came out: fall back to the
            # plain voice.  Once audio has been emitted we raise instead, so
            # partial cloned audio is never followed by a full re-run.
            logger.warning("Voice cloning failed, generating without voice prompt: %s", e)
            kwargs.pop("ref_audio")
            kwargs.pop("ref_text")
            for result in self.model.generate(text, **kwargs):
                yield np.asarray(result.audio, dtype=np.float32), int(result.sample_rate)


def _install_alignment_heads(model, repo: str) -> bool:
    """Load the model's trained alignment heads without downloading on a client's request."""
    import json

    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo, "generation_config.json")
    if not isinstance(path, str):
        logger.warning("MLX Whisper word timestamps unavailable: %s has no cached generation_config.json", repo)
        return False
    try:
        heads = json.loads(Path(path).read_text()).get("alignment_heads")
        if not isinstance(heads, list) or not heads or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
            or not 0 <= pair[0] < model.dims.n_text_layer
            or not 0 <= pair[1] < model.dims.n_text_head
            for pair in heads
        ):
            raise ValueError("missing or invalid alignment_heads")
        model.set_alignment_heads(np.asarray(heads, dtype=np.int64))
        # mlx-audio 0.4.1 stores _alignment_heads but timing.find_alignment reads the public name.
        if not hasattr(model, "alignment_heads"):
            model.alignment_heads = model._alignment_heads
        return True
    except (OSError, ValueError, AttributeError) as exc:
        logger.warning("MLX Whisper word timestamps unavailable for %s: %s", repo, exc)
        return False


class MLXSTTBackend:
    """MLX-based STT backend using mlx-audio Whisper."""

    def __init__(self, model_size: str = "base"):
        self.model = None
        self.model_size = model_size
        self._word_alignment_available = False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        hf_repo = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
        return is_model_cached(hf_repo, weight_extensions=(".safetensors", ".bin", ".npz"))

    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the MLX Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self.model_size == model_size:
            return

        # Run blocking load in thread pool
        await asyncio.to_thread(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        progress_model_name = f"whisper-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(progress_model_name, is_cached):
            from mlx_audio.stt import load
            from mlx_audio.stt.models.whisper.whisper import HFTokenizerWrapper  # lazy: heavy import

            # mlx-audio's own lookup suppresses end-of-text on pre-v3 models; see _no_speech_token.
            HFTokenizerWrapper.no_speech = property(_no_speech_token)

            model_name = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
            logger.info("Loading MLX Whisper model %s...", model_size)

            self.model = load(model_name)
            self._word_alignment_available = _install_alignment_heads(self.model, model_name)

        self.model_size = model_size
        logger.info("MLX Whisper model %s loaded successfully", model_size)

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._word_alignment_available = False
            logger.info("MLX Whisper model unloaded")

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        *,
        prompt: str | None = None,
        hotwords: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_path: Path to audio file
            language: Optional language hint
            model_size: Optional model size override

        Returns:
            Transcribed text
        """
        await self.load_model_async(model_size)

        def _transcribe_sync():
            """Run synchronous transcription in thread pool."""
            return _result_text(self._generate_sync(audio_path, language, prompt=prompt, hotwords=hotwords))

        # Run blocking transcription in thread pool
        return await asyncio.to_thread(_transcribe_sync)

    def _generate_sync(
        self, audio_path: str, language: Optional[str], *, prompt: str | None = None,
        hotwords: list[str] | tuple[str, ...] | None = None, word_timestamps: bool = False,
    ):
        """MLX Whisper's ``generate`` on a file path (an ``STTOutput`` with text and segments)."""
        decode_options = {}
        if language:
            decode_options["language"] = language
        from ..utils.speech_text import transcription_prompt

        hint = transcription_prompt(prompt, hotwords)
        if hint:
            decode_options["initial_prompt"] = hint
        if word_timestamps:
            if not self._word_alignment_available:
                raise RuntimeError(
                    "Word timestamps are unavailable: this MLX Whisper model has no valid cached alignment_heads "
                    "in generation_config.json; use a model with alignment metadata or faster-whisper"
                )
            decode_options["word_timestamps"] = True

        # Inference runs with the process's default HF_HUB_OFFLINE
        # state — see the comment in MLXTTSBackend.generate for the
        # regression this revert fixes (issue #462).
        return self.model.generate(str(audio_path), **decode_options)

    async def transcribe_detailed(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        *,
        prompt: str | None = None,
        hotwords: list[str] | tuple[str, ...] | None = None,
        word_timestamps: bool = False,
    ):
        """Transcribe with segment timestamps (``utils.subtitles.Transcript``)."""
        from ..utils.audio import load_audio
        from ..utils.subtitles import Transcript, segments_from_dicts, text_of, words_from_dicts

        await self.load_model_async(model_size)

        def _transcribe_sync():
            audio, sr = load_audio(str(audio_path))
            duration = len(audio) / sr
            result = self._generate_sync(audio_path, language, prompt=prompt, hotwords=hotwords, word_timestamps=word_timestamps)
            raw_segments = getattr(result, "segments", None) or (result.get("segments") if isinstance(result, dict) else None)
            detected = getattr(result, "language", None) or (result.get("language") if isinstance(result, dict) else None)
            segments = segments_from_dicts(raw_segments or (), duration=duration)
            # Rebuilding the text from the kept segments drops the padding hallucinations too.
            text = text_of(segments) if segments else _result_text(result)
            kept = {segment.id for segment in segments}
            words = words_from_dicts(
                (word for i, segment in enumerate(raw_segments or ()) if segment.get("id", i) in kept for word in segment.get("words", ())),
                duration=duration,
            ) if word_timestamps else ()
            return Transcript(text=text, segments=segments, language=language or detected, duration=duration, words=words)

        return await asyncio.to_thread(_transcribe_sync)


def _no_speech_token(tokenizer) -> int | None:
    """Whisper's no-speech token id under either name; installed as mlx-audio's ``HFTokenizerWrapper.no_speech``.

    mlx-audio (0.4.1, and its main branch) looks up only ``<|nospeech|>``,
    which the tokenizers before large-v3 (tiny to medium, large-v2) call
    ``<|nocaptions|>``.  The lookup fell back to the unknown token, Whisper's
    ``<|endoftext|>``, and since decoding suppresses the no-speech token it
    suppressed end-of-text: every window decoded up to the token limit,
    appending invented text, and ``no_speech_prob`` measured end-of-text
    instead of silence.
    """
    hf_tokenizer = tokenizer.hf_tokenizer
    for name in ("<|nospeech|>", "<|nocaptions|>"):
        token_id = hf_tokenizer.convert_tokens_to_ids(name)
        if token_id is not None and token_id != hf_tokenizer.unk_token_id:
            return token_id
    return None


def _result_text(result) -> str:
    """The transcript text of whatever ``mlx_audio.stt`` returned (string, dict or ``STTOutput``)."""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    if hasattr(result, "text"):
        return str(result.text).strip()
    return str(result).strip()
