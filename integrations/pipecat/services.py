"""Pipecat speech services using Voicebox's OpenAI-compatible HTTP and WebSocket APIs."""

from collections.abc import AsyncGenerator
from copy import deepcopy
from urllib.parse import urlsplit, urlunsplit

from openai import AsyncOpenAI
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.stt import OpenAIRealtimeSTTService, OpenAISTTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from stt_resampler import VoiceboxSTTResampler
from tts_settings import VoiceboxTTSSettings, complete_settings, speech_options, validate_settings

SAMPLE_RATE = 24000


def api_url(server_url: str) -> str:
    """Normalize a server root or /v1 URL without embedding credentials or query strings."""
    parts = urlsplit(server_url.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("VOICEBOX_URL must be an http:// or https:// server URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("VOICEBOX_URL must not contain credentials, a query, or a fragment")
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def require_key(api_key: str) -> str:
    """Prevent the OpenAI SDK from sending OPENAI_API_KEY to Voicebox by fallback."""
    if not api_key.strip():
        raise ValueError("VOICEBOX_CLIENT_KEY is required; create a Voicebox client key")
    return api_key.strip()


class VoiceboxTTSService(TTSService):
    """Stream 24 kHz PCM with arbitrary preset and cloned Voicebox voice IDs.

    Pipecat 1.15's OpenAITTSService rejects voice IDs outside OpenAI's catalog.
    This adapter leaves voice validation to Voicebox and uses Pipecat's standard
    sentence aggregation, audio contexts, interruption handling, and metrics.
    """

    Settings = VoiceboxTTSSettings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        voice: str = "af_heart",
        model: str = "tts-1",
        settings: VoiceboxTTSSettings | None = None,
    ):
        """Create a service with explicit Voicebox credentials and its fixed PCM rate."""
        api_key = require_key(api_key)
        if not voice.strip() or not model.strip():
            raise ValueError("Voicebox voice and TTS model must not be empty")
        super().__init__(
            sample_rate=SAMPLE_RATE,
            push_start_frame=True,
            push_stop_frames=True,
            # Queued or cold synthesis can exceed Pipecat's three-second idle default.
            stop_frame_timeout_s=130,
            settings=complete_settings(model, voice, settings),
        )
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=api_url(base_url), timeout=120, max_retries=0
        )

    async def _update_settings(self, delta):
        candidate = deepcopy(self._settings)
        candidate.apply_update(delta)
        validate_settings(candidate)
        return await super()._update_settings(delta)

    def can_generate_metrics(self) -> bool:
        """Expose Pipecat's time-to-first-byte and usage measurements."""
        return True

    async def cleanup(self):
        """Close this service's HTTP pool when the pipeline ends."""
        try:
            await self._client.close()
        finally:
            await super().cleanup()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Yield PCM frames; cancellation closes HTTP and errors reach the pipeline."""
        received_audio = False
        try:
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._settings.model,
                voice=self._settings.voice,
                input=text,
                response_format="pcm",
                extra_body=speech_options(self._settings),
            ) as response:
                await self.start_tts_usage_metrics(text)
                # A 100 ms chunk keeps frames aligned without buffering whole sentences.
                async for chunk in response.iter_bytes(SAMPLE_RATE // 10 * 2):
                    if len(chunk) % 2:
                        raise ValueError("Voicebox returned an incomplete PCM16 sample")
                    if chunk:
                        received_audio = True
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, SAMPLE_RATE, 1, context_id=context_id)
                if not received_audio:
                    raise ValueError("Voicebox returned an empty speech response")
        except Exception as exc:
            # Keep the exception so Pipecat classifies permanent provider errors (e.g. 401).
            yield ErrorFrame(error=f"Voicebox speech failed: {exc}", exception=exc)
            if self.audio_context_available(context_id):
                await self.remove_audio_context(context_id)


class VoiceboxBatchSTTService(OpenAISTTService):
    """Use stock HTTP transcription, closing the SDK pool and marking completed utterances."""

    _requires_api_key = True

    async def run_stt(self, audio: bytes):
        """An HTTP transcription response is final for this utterance."""
        async for frame in super().run_stt(audio):
            if isinstance(frame, TranscriptionFrame):
                frame.finalized = True
            yield frame

    async def cleanup(self):
        """Close the per-session SDK client; upstream currently leaves this to GC."""
        try:
            await self._client.close()
        finally:
            await super().cleanup()


class VoiceboxRealtimeSTTService(OpenAIRealtimeSTTService):
    """Use stock GA transcription while identifying Voicebox's definitive final events."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._resampler = VoiceboxSTTResampler()

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        """Skip the provider-latency safety timer after an explicitly completed utterance."""
        if isinstance(frame, TranscriptionFrame) and isinstance(frame.result, dict):
            if frame.result.get("type") == "conversation.item.input_audio_transcription.completed":
                frame.finalized = True
        await super().push_frame(frame, direction)


def voicebox_stt(
    *,
    api_key: str,
    base_url: str,
    model: str = "whisper-1",
    language: str = "en",
    realtime: bool = True,
) -> VoiceboxBatchSTTService | VoiceboxRealtimeSTTService:
    """Configure Pipecat transcription; local VAD commits realtime utterances."""
    api_key = require_key(api_key)
    base_url = api_url(base_url)
    language_value = Language(language)
    if realtime:
        parts = urlsplit(base_url)
        ws_url = urlunsplit(
            (
                "wss" if parts.scheme == "https" else "ws",
                parts.netloc,
                parts.path + "/realtime",
                "",
                "",
            )
        )
        return VoiceboxRealtimeSTTService(
            api_key=api_key,
            base_url=ws_url,
            turn_detection=False,
            settings=OpenAIRealtimeSTTService.Settings(model=model, language=language_value),
        )
    return VoiceboxBatchSTTService(
        api_key=api_key,
        base_url=base_url,
        settings=OpenAISTTService.Settings(model=model, language=language_value),
    )
