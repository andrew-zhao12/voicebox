"""Encode streamed PCM into the formats OpenAI clients ask for.

``wav``, ``pcm`` and ``ulaw_8000`` (G.711 mu-law at 8 kHz, for telephony) are
produced in Python and stream chunk by chunk.  The
compressed formats stream through an ffmpeg pipe when ffmpeg is installed
(the Docker image ships it).  Without ffmpeg, ``mp3``, ``flac`` and ``opus``
are encoded with libsndfile once the whole clip exists (soundfile bundles
those encoders), so they still work on a desktop or dev machine, just not
incrementally; ``aac`` needs ffmpeg.

Only numpy, soundfile and asyncio are used, so the module stays importable
without the ML stack.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import shutil
from collections.abc import AsyncIterator

import numpy as np
import soundfile as sf

from .wav_stream import float_to_pcm16_bytes, streaming_wav_header

logger = logging.getLogger(__name__)

FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm", "ulaw_8000")
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "ulaw_8000": "audio/basic",
}
EXTENSIONS = {
    "mp3": "mp3",
    "opus": "ogg",
    "aac": "aac",
    "flac": "flac",
    "wav": "wav",
    "pcm": "pcm",
    "ulaw_8000": "ulaw",
}
# Encoded in this module, so always available and always streamed.
_PYTHON_FORMATS = ("wav", "pcm", "ulaw_8000")
TELEPHONY_RATE = 8000
# G.711 mu-law segment ends for 14-bit magnitudes (ITU-T G.711, as in audioop).
_ULAW_SEGMENT_ENDS = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])

# Container and codec per format.  ``-write_xing 0`` keeps the mp3 muxer from
# trying to seek back into a pipe; adts and ogg are stream containers anyway.
_FFMPEG_ARGS = {
    "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-b:a", "128k", "-write_xing", "0"],
    "opus": ["-f", "ogg", "-codec:a", "libopus", "-b:a", "64k"],
    "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "128k"],
    "flac": ["-f", "flac", "-codec:a", "flac"],
}
# soundfile (libsndfile) fallbacks: format and subtype.
_SOUNDFILE = {"mp3": ("MP3", "MPEG_LAYER_III"), "flac": ("FLAC", "PCM_16"), "opus": ("OGG", "OPUS")}
_READ_CHUNK = 16384


class UnsupportedFormatError(ValueError):
    pass


class EncodingError(RuntimeError):
    pass


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _soundfile_supports(fmt: str) -> bool:
    if fmt not in _SOUNDFILE:
        return False
    major, subtype = _SOUNDFILE[fmt]
    try:
        return major in sf.available_formats() and subtype in sf.available_subtypes(major)
    except Exception:
        return False


def available_formats() -> list[str]:
    """Formats this server can produce, in ``FORMATS`` order."""
    has_ffmpeg = ffmpeg_path() is not None
    return [fmt for fmt in FORMATS if fmt in _PYTHON_FORMATS or has_ffmpeg or _soundfile_supports(fmt)]


def streams_incrementally(fmt: str) -> bool:
    """Whether bytes leave before synthesis has finished."""
    return fmt in _PYTHON_FORMATS or ffmpeg_path() is not None


def output_sample_rate(fmt: str, sample_rate: int) -> int:
    """The rate of the encoded audio: the engine's, except for the telephony format."""
    return TELEPHONY_RATE if fmt == "ulaw_8000" else sample_rate


def _lowpass_kernel(cutoff: float, sample_rate: int, taps: int) -> np.ndarray:
    """Windowed-sinc FIR low-pass (Hann), unity DC gain."""
    n = np.arange(taps) - (taps - 1) / 2
    fc = cutoff / sample_rate
    kernel = 2 * fc * np.sinc(2 * fc * n)
    kernel *= np.hanning(taps)
    return (kernel / kernel.sum()).astype(np.float32)


def linear_to_ulaw(pcm16: np.ndarray) -> bytes:
    """G.711 mu-law bytes for signed 16-bit samples, identical to ``audioop.lin2ulaw``."""
    value = pcm16.astype(np.int32) >> 2
    negative = value < 0
    magnitude = np.minimum(np.where(negative, -value, value), 8159) + 33
    segment = np.searchsorted(_ULAW_SEGMENT_ENDS, magnitude)
    code = (np.minimum(segment, 7) << 4) | ((magnitude >> (np.minimum(segment, 7) + 1)) & 0x0F)
    code = np.where(segment >= 8, 0x7F, code)
    return (code ^ np.where(negative, 0x7F, 0xFF)).astype(np.uint8).tobytes()


class TelephonyEncoder:
    """Stream float PCM at any engine rate into mu-law at 8 kHz.

    A low-pass filter removes everything above the telephone band before the
    samples are dropped; decimating without it folds sibilants back into the
    band at full level.  The filter history and the decimation position carry
    across chunks, so chunked output matches a single pass, and the filter's
    delay is skipped at the start and flushed at the end so the output keeps
    the input's timing and length.
    """

    def __init__(self, sample_rate: int) -> None:
        if sample_rate < TELEPHONY_RATE:
            raise UnsupportedFormatError(f"ulaw_8000 from {sample_rate} Hz")
        self.step = sample_rate / TELEPHONY_RATE
        # The transition band narrows as the rate rises, so the kernel grows with it.
        taps = max(63, int(63 * self.step / 3) | 1)
        self.kernel = _lowpass_kernel(0.45 * TELEPHONY_RATE, sample_rate, taps)
        self.history = np.zeros(taps - 1, dtype=np.float32)
        self.delay = (taps - 1) // 2
        self.position = float(self.delay)
        self._flushed = False

    def feed(self, frame: np.ndarray) -> bytes:
        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return b""
        signal = np.concatenate([self.history, samples])
        filtered = np.convolve(signal, self.kernel, mode="valid")
        self.history = signal[-(self.kernel.size - 1) :]
        positions = np.arange(self.position, filtered.size, self.step)
        if self.step.is_integer():
            picked = filtered[positions.astype(np.int64)]
        else:
            picked = np.interp(positions, np.arange(filtered.size), filtered)
        self.position += self.step * positions.size - filtered.size
        return linear_to_ulaw(np.frombuffer(float_to_pcm16_bytes(picked), dtype="<i2"))

    def flush(self) -> bytes:
        """Release the last samples, which are still inside the filter."""
        if self._flushed:
            return b""
        self._flushed = True
        return self.feed(np.zeros(self.delay, dtype=np.float32))


def encode_bytes(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Encode a whole clip with libsndfile (``mp3``, ``flac``, ``opus``, ``wav``)."""
    if fmt == "wav":
        return streaming_wav_header(sample_rate, data_length=len(audio) * 2) + float_to_pcm16_bytes(audio)
    if fmt == "pcm":
        return float_to_pcm16_bytes(audio)
    if fmt == "ulaw_8000":
        encoder = TelephonyEncoder(sample_rate)
        return encoder.feed(audio) + encoder.flush()
    if not _soundfile_supports(fmt):
        raise UnsupportedFormatError(fmt)
    major, subtype = _SOUNDFILE[fmt]
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), sample_rate, format=major, subtype=subtype)
    return buffer.getvalue()


async def encode_stream(frames: AsyncIterator[np.ndarray], sample_rate: int, fmt: str) -> AsyncIterator[bytes]:
    """Turn float PCM frames into encoded bytes, streaming whenever the format allows."""
    if fmt not in FORMATS:
        raise UnsupportedFormatError(fmt)
    if fmt == "wav":
        yield streaming_wav_header(sample_rate)
        async for frame in frames:
            yield float_to_pcm16_bytes(frame)
        return
    if fmt == "pcm":
        async for frame in frames:
            yield float_to_pcm16_bytes(frame)
        return
    if fmt == "ulaw_8000":
        encoder = TelephonyEncoder(sample_rate)
        async for frame in frames:
            if chunk := encoder.feed(frame):
                yield chunk
        if tail := encoder.flush():
            yield tail
        return
    ffmpeg = ffmpeg_path()
    if ffmpeg is not None:
        async for chunk in _ffmpeg_stream(ffmpeg, frames, sample_rate, fmt):
            yield chunk
        return
    if not _soundfile_supports(fmt):
        raise UnsupportedFormatError(fmt)
    parts = [np.asarray(frame, dtype=np.float32) async for frame in frames]
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    yield encode_bytes(audio, sample_rate, fmt)


async def _ffmpeg_stream(
    ffmpeg: str, frames: AsyncIterator[np.ndarray], sample_rate: int, fmt: str
) -> AsyncIterator[bytes]:
    """Pipe PCM16 through ffmpeg and yield the container bytes as they come out."""
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        *_FFMPEG_ARGS[fmt],
        "pipe:1",
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_chunks: list[bytes] = []

    async def feed() -> None:
        try:
            async for frame in frames:
                process.stdin.write(float_to_pcm16_bytes(frame))
                await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with contextlib.suppress(Exception):
                process.stdin.close()

    async def drain_stderr() -> None:
        while True:
            chunk = await process.stderr.read(_READ_CHUNK)
            if not chunk:
                return
            if sum(len(c) for c in stderr_chunks) < 64 * 1024:
                stderr_chunks.append(chunk)

    feeder = asyncio.create_task(feed())
    stderr_task = asyncio.create_task(drain_stderr())
    try:
        while True:
            chunk = await process.stdout.read(_READ_CHUNK)
            if not chunk:
                break
            yield chunk
        await feeder
        await stderr_task
        code = await process.wait()
        if code != 0:
            message = b"".join(stderr_chunks).decode(errors="replace").strip()
            raise EncodingError(f"ffmpeg exited with {code}: {message or 'no output'}")
    finally:
        for task in (feeder, stderr_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(feeder, stderr_task, return_exceptions=True)
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
