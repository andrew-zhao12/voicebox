"""Per-utterance native decoder sessions with cancellation-safe inference ownership."""

import asyncio
import logging
from dataclasses import dataclass

from . import transcribe
from .inference_slots import whisper_slot

logger = logging.getLogger(__name__)


@dataclass
class _State:
    stream: object = None
    consumed: int = 0
    busy: bool = False
    closed: bool = False


class NativeTranscriber:
    """Send each sample once; retain decoder state across partials and close after in-flight work."""

    def __init__(self):
        self.states: dict[str, _State] = {}

    @staticmethod
    def supports(model: str | None) -> bool:
        from ..backends import get_stt_model_configs

        return any(cfg.model_name == model and cfg.supports_streaming for cfg in get_stt_model_configs())

    async def transcribe(self, work):
        state = self.states.setdefault(work.item_id, _State())
        return await whisper_slot.run(self._step, state, work)

    async def _step(self, state, work):
        if state.closed:
            raise RuntimeError("Native decoder session is closed")
        state.busy = True
        try:
            if state.stream is None:
                backend, size, cfg = transcribe.resolve_stt(work.model)
                transcribe.validate_options(
                    cfg, prompt=work.prompt, hotwords=work.hotwords, word_timestamps=work.word_timestamps
                )
                state.stream = await backend.open_stream(
                    work.language,
                    size,
                    prompt=work.prompt,
                    hotwords=work.hotwords,
                    word_timestamps=work.word_timestamps,
                )
            audio = work.audio[state.consumed :]
            text = await state.stream.accept_audio(audio) if audio.size else ""
            state.consumed = max(state.consumed, work.audio.size)
            if work.kind == "final":
                result = await state.stream.finish()
                state.closed = True
                return result
            return text
        except (Exception, asyncio.CancelledError):
            state.closed = True
            raise
        finally:
            state.busy = False
            if state.closed:
                await self._close(state)

    async def prune(self, keep: set[str]):
        for item_id in list(self.states):
            if item_id not in keep:
                state = self.states.pop(item_id)
                state.closed = True
                if not state.busy:
                    await self._close(state)

    @staticmethod
    async def _close(state):
        stream, state.stream = state.stream, None
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                # A failed decoder cleanup must not strand socket/session admission or hide inference errors.
                logger.exception("Native STT decoder cleanup failed")
