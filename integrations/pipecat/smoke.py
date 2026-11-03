"""Check real Voicebox TTS and Pipecat STT without a browser or paid LLM: python smoke.py."""

import argparse
import asyncio
import io
import json
import re
import time
import wave

from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from bot import Settings
from services import SAMPLE_RATE, VoiceboxTTSService, voicebox_stt

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."


class TranscriptCollector(FrameProcessor):
    """Resolve one utterance or surface a pipeline error to the smoke driver."""

    def __init__(self):
        super().__init__()
        self.result = asyncio.get_running_loop().create_future()

    async def process_frame(self, frame, direction: FrameDirection):
        """Capture a final transcript while preserving pipeline control frames."""
        await super().process_frame(frame, direction)
        if not self.result.done():
            if isinstance(frame, TranscriptionFrame):
                self.result.set_result(frame.text)
            elif isinstance(frame, ErrorFrame):
                self.result.set_exception(RuntimeError(frame.error))
        await self.push_frame(frame, direction)


async def transcribe_realtime(stt, pcm: bytes, timeout: float) -> str:
    """Exercise the stock service's GA handshake, audio upload, commit, and final event."""
    collector = TranscriptCollector()
    worker = PipelineWorker(
        Pipeline([stt, collector]),
        params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE),
        idle_timeout_secs=timeout,
        enable_rtvi=False,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        async with asyncio.timeout(timeout):
            # This smoke is tied to the pinned SDK; audio before session.updated is dropped.
            while not stt._session_ready:
                if running.done():
                    await running
                    raise RuntimeError("Pipecat stopped before the transcription session was ready")
                await asyncio.sleep(0.02)
            # Leave a short quiet tail, as the live microphone would.
            audio = pcm + b"\0" * (SAMPLE_RATE // 2)
            for offset in range(0, len(audio), 4800):
                await worker.queue_frames(
                    [InputAudioRawFrame(audio[offset : offset + 4800], SAMPLE_RATE, 1)]
                )
            await worker.queue_frames([VADUserStoppedSpeakingFrame(stop_secs=0.25)])
            return await collector.result
    finally:
        await runner.cancel()
        await running


def wav_bytes(pcm: bytes) -> bytes:
    """Wrap the fixed-rate mono PCM returned by Voicebox in a batch-upload WAV."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buffer.getvalue()


def word_error_rate(expected: str, actual: str) -> float:
    """Compare spoken words, ignoring punctuation and capitalization."""
    reference = re.findall(r"\w+", expected.lower())
    result = re.findall(r"\w+", actual.lower())
    row = list(range(len(result) + 1))
    for i, token in enumerate(reference, 1):
        previous, row = row, [i]
        for j, received in enumerate(result, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (token != received)))
    return row[-1] / max(1, len(reference))


async def check(*, mode: str, text: str, timeout: float, max_wer: float):
    """Synthesize with the real adapter, transcribe with stock services, and assert content."""
    settings = Settings.from_env(require_llm=False)
    tts = VoiceboxTTSService(
        api_key=settings.voicebox_key,
        base_url=settings.voicebox_url,
        voice=settings.voice,
        model=settings.tts_model,
    )
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout):
            chunks = []
            async for frame in tts.run_tts(text, "smoke"):
                if isinstance(frame, ErrorFrame):
                    raise RuntimeError(frame.error)
                chunks.append(frame.audio)
    finally:
        await tts.cleanup()
    pcm = b"".join(chunks)
    if not pcm:
        raise RuntimeError("Voicebox produced no speech")
    print(
        json.dumps(
            {
                "tts_seconds": round(time.monotonic() - started, 3),
                "audio_seconds": len(pcm) / SAMPLE_RATE / 2,
            }
        )
    )
    for realtime in [False, True] if mode == "both" else [mode == "realtime"]:
        stt = voicebox_stt(
            api_key=settings.voicebox_key,
            base_url=settings.voicebox_url,
            model=settings.stt_model,
            language=settings.language,
            realtime=realtime,
        )
        started = time.monotonic()
        if realtime:
            transcript = await transcribe_realtime(stt, pcm, timeout)
        else:
            try:
                async with asyncio.timeout(timeout):
                    frames = [frame async for frame in stt.run_stt(wav_bytes(pcm))]
                errors = [frame.error for frame in frames if isinstance(frame, ErrorFrame)]
                if errors:
                    raise RuntimeError("; ".join(errors))
                transcript = " ".join(
                    frame.text for frame in frames if isinstance(frame, TranscriptionFrame)
                )
            finally:
                await stt.cleanup()
        wer = word_error_rate(text, transcript)
        print(
            json.dumps(
                {
                    "stt": "realtime" if realtime else "batch",
                    "transcript": transcript,
                    "wer": wer,
                    "seconds": round(time.monotonic() - started, 3),
                }
            )
        )
        if wer > max_wer:
            raise RuntimeError(f"Transcript WER {wer:.2f} exceeds {max_wer:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stt", choices=["batch", "realtime", "both"], default="both")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-wer", type=float, default=0.25)
    args = parser.parse_args()
    asyncio.run(check(mode=args.stt, text=args.text, timeout=args.timeout, max_wer=args.max_wer))
