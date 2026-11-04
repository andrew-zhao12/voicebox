"""Run a browser voice agent with Voicebox speech: python bot.py -t webrtc."""

import os
import re
from dataclasses import dataclass, field

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from services import (
    SAMPLE_RATE,
    VoiceboxSpeechTimeoutUserTurnStopStrategy,
    VoiceboxTTSService,
    api_url,
    require_key,
    voicebox_stt,
)
from streaming_tts import VoiceboxStreamingTTSService
from tts_settings import VoiceboxTTSSettings, complete_settings, pronunciation_file

GREETING = "Hello! I'm a voice assistant. How can I help you today?"
INSTRUCTIONS = (
    "You are a helpful voice assistant. Answer in one to three short sentences of plain text. "
    "Your responses will be spoken aloud, so avoid markdown, lists, emoji, and symbols."
)


@dataclass(frozen=True)
class Settings:
    """Independent credentials and endpoints for speech and conversation."""

    voicebox_url: str
    voicebox_key: str = field(repr=False)
    voice: str = "af_heart"
    tts_model: str = "tts-1"
    tts_transport: str = "http"
    tts_settings: VoiceboxTTSSettings | None = None
    tts_timeout: float = 120
    stt_model: str = "whisper-1"
    language: str = "en"
    realtime: bool = True
    echo: bool = False
    llm_url: str | None = None
    llm_key: str = field(default="", repr=False)
    llm_model: str = "gpt-4.1-mini"

    @classmethod
    def from_env(cls, *, require_llm: bool = True) -> "Settings":
        """Validate settings before accepting calls; echo mode needs no external LLM."""
        key = require_key(os.getenv("VOICEBOX_CLIENT_KEY", ""))
        url = api_url(os.getenv("VOICEBOX_URL", "http://127.0.0.1:17493"))
        llm_url = os.getenv("LLM_BASE_URL") or None
        # A custom provider never implicitly receives the user's OpenAI key.
        llm_key = os.getenv("LLM_API_KEY") or (
            "local" if llm_url else os.getenv("OPENAI_API_KEY", "")
        )
        mode = os.getenv("VOICEBOX_LLM", "openai").strip().lower()
        if mode not in {"openai", "echo"}:
            raise ValueError("VOICEBOX_LLM must be openai or echo")
        if require_llm and mode != "echo" and not llm_key:
            raise ValueError("Set OPENAI_API_KEY, LLM_API_KEY, LLM_BASE_URL, or VOICEBOX_LLM=echo")
        stt_mode = os.getenv("VOICEBOX_STT_REALTIME", "1").strip().lower()
        if stt_mode not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
            raise ValueError("VOICEBOX_STT_REALTIME must be 1 or 0")
        tts_transport = os.getenv("VOICEBOX_TTS_TRANSPORT", "http").strip().lower()
        if tts_transport not in {"http", "websocket"}:
            raise ValueError("VOICEBOX_TTS_TRANSPORT must be http or websocket")
        if tts_transport == "http" and (
            os.getenv("VOICEBOX_TEXT_FLUSH_CHARS") or os.getenv("VOICEBOX_TTS_TIMEOUT")
        ):
            raise ValueError(
                "VOICEBOX_TEXT_FLUSH_CHARS and VOICEBOX_TTS_TIMEOUT require websocket TTS"
            )
        tts_settings = complete_settings(
            os.getenv("VOICEBOX_TTS_MODEL", "tts-1"),
            os.getenv("VOICEBOX_VOICE", "af_heart"),
            VoiceboxTTSSettings(
                speed=float(os.environ["VOICEBOX_TTS_SPEED"])
                if os.getenv("VOICEBOX_TTS_SPEED")
                else None,
                instructions=os.getenv("VOICEBOX_TTS_INSTRUCTIONS") or None,
                pronunciation_replacements=pronunciation_file(
                    os.getenv("VOICEBOX_PRONUNCIATIONS_FILE")
                ),
                first_chunk_chars=int(os.environ["VOICEBOX_FIRST_CHUNK_CHARS"])
                if os.getenv("VOICEBOX_FIRST_CHUNK_CHARS")
                else None,
                max_chunk_chars=int(os.environ["VOICEBOX_MAX_CHUNK_CHARS"])
                if os.getenv("VOICEBOX_MAX_CHUNK_CHARS")
                else None,
                text_flush_chars=int(os.getenv("VOICEBOX_TEXT_FLUSH_CHARS", "120")),
            ),
        )
        tts_timeout = float(os.getenv("VOICEBOX_TTS_TIMEOUT", "120"))
        if not 1 <= tts_timeout <= 1800:
            raise ValueError("VOICEBOX_TTS_TIMEOUT must be between 1 and 1800 seconds")
        return cls(
            voicebox_url=url,
            voicebox_key=key,
            voice=os.getenv("VOICEBOX_VOICE", "af_heart"),
            tts_model=os.getenv("VOICEBOX_TTS_MODEL", "tts-1"),
            tts_transport=tts_transport,
            tts_settings=tts_settings,
            tts_timeout=tts_timeout,
            stt_model=os.getenv("VOICEBOX_STT_MODEL", "whisper-1"),
            language=os.getenv("VOICEBOX_LANGUAGE", "en"),
            realtime=stt_mode in {"1", "true", "yes", "on"},
            echo=mode == "echo",
            llm_url=llm_url,
            llm_key=llm_key,
            llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        )


class EchoLLM(FrameProcessor):
    """Deterministic conversation substitute for testing speech without an LLM account."""

    def __init__(self, *, tail: str = ""):
        super().__init__()
        self.tail = tail

    async def process_frame(self, frame, direction: FrameDirection):
        """Reply to the latest plain-text user message through standard LLM frames."""
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            messages = frame.context.get_messages()
            text = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            if isinstance(text, list):
                text = " ".join(p.get("text", "") for p in text if p.get("type") == "text")
            users = sum(m.get("role") == "user" for m in messages)
            if self.tail and users % 2 == 1:
                text += " " + self.tail
            await self.push_frame(LLMFullResponseStartFrame())
            # Exercise incremental text in the local smoke just as an LLM streams tokens.
            for token in re.findall(r"\S+\s*", f"You said: {text}"):
                await self.push_frame(LLMTextFrame(token))
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class ChatService(OpenAILLMService):
    """Close the per-call chat client's HTTP pool when its pipeline ends."""

    async def cleanup(self):
        """Release connections owned by Pipecat's OpenAI-compatible chat service."""
        try:
            await self._client.close()
        finally:
            await super().cleanup()


def voicebox_tts(settings: Settings):
    """Select speech transport explicitly; HTTP remains the default."""
    options = dict(
        api_key=settings.voicebox_key,
        base_url=settings.voicebox_url,
        voice=settings.voice,
        model=settings.tts_model,
        settings=settings.tts_settings,
    )
    if settings.tts_transport == "websocket":
        return VoiceboxStreamingTTSService(**options, timeout=settings.tts_timeout)
    return VoiceboxTTSService(**options)


def build_pipeline(transport: BaseTransport, settings: Settings) -> Pipeline:
    """Build local VAD -> Voicebox STT -> conversation -> Voicebox TTS."""
    stt = voicebox_stt(
        api_key=settings.voicebox_key,
        base_url=settings.voicebox_url,
        model=settings.stt_model,
        language=settings.language,
        realtime=settings.realtime,
    )
    tts = voicebox_tts(settings)
    llm = (
        EchoLLM(tail=os.getenv("VOICEBOX_ECHO_TAIL", ""))
        if settings.echo
        else ChatService(
            api_key=settings.llm_key,
            base_url=settings.llm_url,
            settings=OpenAILLMService.Settings(
                model=settings.llm_model, system_instruction=INSTRUCTIONS
            ),
        )
    )
    user, assistant = LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # Explicit silence-based turns avoid Pipecat's optional smart-turn model download.
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    VoiceboxSpeechTimeoutUserTurnStopStrategy(stt)
                    if settings.realtime
                    else SpeechTimeoutUserTurnStopStrategy()
                ]
            ),
        ),
    )
    return Pipeline([transport.input(), stt, user, llm, tts, transport.output(), assistant])


async def bot(runner_args: RunnerArguments):
    """Pipecat runner entry point for a local SmallWebRTC browser session."""
    settings = Settings.from_env()
    transport = await create_transport(
        runner_args,
        {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)},
    )
    worker = PipelineWorker(
        build_pipeline(transport, settings),
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=SAMPLE_RATE,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        await worker.queue_frames([TTSSpeakFrame(GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        await runner.cancel()

    await runner.run()


if __name__ == "__main__":
    import sys

    from pipecat.runner.run import main

    # --help remains usable before keys are configured.
    if "--help" not in sys.argv and "-h" not in sys.argv:
        try:
            Settings.from_env()
        except ValueError as exc:
            sys.exit(str(exc))
    main()
