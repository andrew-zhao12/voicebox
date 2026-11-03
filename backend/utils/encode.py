"""Encode streamed PCM into the formats OpenAI clients ask for.

``wav``, ``pcm`` and ``ulaw_8000`` (G.711 mu-law at 8 kHz, for telephony) are
produced in Python and stream chunk by chunk.  ``pcm`` is always 24 kHz, the
rate OpenAI documents for it, because raw samples carry no rate a client could
read; ``wav`` and the compressed formats keep the engine's rate.  The
compressed formats stream through an ffmpeg pipe when ffmpeg is installed
(the Docker image ships it).  Without ffmpeg, ``mp3``, ``flac`` and ``opus``
are encoded with libsndfile once the whole clip exists (soundfile bundles
those encoders), so they still work on a desktop or dev machine, just not
incrementally; ``aac`` needs ffmpeg.  The G.711 decoders turn telephony
audio that clients send back into 16-bit samples.

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

from .resample import StreamResampler, resample
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
# OpenAI's ``pcm`` is 24 kHz; clients such as LiveKit's plugin assume that rate and never check it.
PCM_RATE = 24000
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
    """The rate of the encoded audio: the engine's, except for raw PCM and the telephony format."""
    if fmt == "ulaw_8000":
        return TELEPHONY_RATE
    if fmt == "pcm":
        return PCM_RATE
    return sample_rate


def linear_to_ulaw(pcm16: np.ndarray) -> bytes:
    """G.711 mu-law bytes for signed 16-bit samples, identical to ``audioop.lin2ulaw``."""
    value = pcm16.astype(np.int32) >> 2
    negative = value < 0
    magnitude = np.minimum(np.where(negative, -value, value), 8159) + 33
    segment = np.searchsorted(_ULAW_SEGMENT_ENDS, magnitude)
    code = (np.minimum(segment, 7) << 4) | ((magnitude >> (np.minimum(segment, 7) + 1)) & 0x0F)
    code = np.where(segment >= 8, 0x7F, code)
    return (code ^ np.where(negative, 0x7F, 0xFF)).astype(np.uint8).tobytes()


def _ulaw_table() -> np.ndarray:
    """The 16-bit sample for each of the 256 mu-law codes (ITU-T G.711, as in audioop)."""
    code = ~np.arange(256) & 0xFF
    magnitude = (((code & 0x0F) << 3) + 0x84) << ((code >> 4) & 0x07)
    return np.where(code & 0x80, 0x84 - magnitude, magnitude - 0x84).astype(np.int16)


def _alaw_table() -> np.ndarray:
    """The 16-bit sample for each of the 256 A-law codes (ITU-T G.711, as in audioop)."""
    code = np.arange(256) ^ 0x55
    segment = (code >> 4) & 0x07
    mantissa = (code & 0x0F) << 4
    magnitude = np.where(segment == 0, mantissa + 0x08, (mantissa + 0x108) << np.maximum(segment - 1, 0))
    return np.where(code & 0x80, magnitude, -magnitude).astype(np.int16)


_ULAW_TO_LINEAR = _ulaw_table()
_ALAW_TO_LINEAR = _alaw_table()


def ulaw_to_linear(data: bytes) -> np.ndarray:
    """Signed 16-bit samples for G.711 mu-law bytes, identical to ``audioop.ulaw2lin(data, 2)``."""
    return _ULAW_TO_LINEAR[np.frombuffer(data, dtype=np.uint8)]


def alaw_to_linear(data: bytes) -> np.ndarray:
    """Signed 16-bit samples for G.711 A-law bytes, identical to ``audioop.alaw2lin(data, 2)``."""
    return _ALAW_TO_LINEAR[np.frombuffer(data, dtype=np.uint8)]


def _float_to_ulaw(samples: np.ndarray) -> bytes:
    return linear_to_ulaw(np.frombuffer(float_to_pcm16_bytes(samples), dtype="<i2"))


class TelephonyEncoder:
    """Stream float PCM at any engine rate into mu-law at 8 kHz.

    The band-limited resampler removes everything above the telephone band
    before the rate drops; dropping samples without it folds sibilants back
    into the band at full level.  Its state carries across chunks, so chunked
    output matches a single pass, and it compensates its own delay, so the
    output keeps the input's timing and length.
    """

    def __init__(self, sample_rate: int) -> None:
        if sample_rate < TELEPHONY_RATE:
            raise UnsupportedFormatError(f"ulaw_8000 from {sample_rate} Hz")
        self._resampler = StreamResampler(sample_rate, TELEPHONY_RATE)

    def feed(self, frame: np.ndarray) -> bytes:
        return _float_to_ulaw(self._resampler.feed(frame))

    def flush(self) -> bytes:
        """Release the last samples, which are still inside the filter; later calls return nothing."""
        return _float_to_ulaw(self._resampler.flush())


def encode_bytes(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Encode a whole clip (``mp3``, ``flac`` and ``opus`` with libsndfile; ``pcm`` at 24 kHz, as streamed)."""
    if fmt == "wav":
        return streaming_wav_header(sample_rate, data_length=len(audio) * 2) + float_to_pcm16_bytes(audio)
    if fmt == "pcm":
        if sample_rate != PCM_RATE:
            audio = resample(audio, sample_rate, PCM_RATE)
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


async def _resampled(frames: AsyncIterator[np.ndarray], src_rate: int, dst_rate: int) -> AsyncIterator[np.ndarray]:
    """*frames* at *dst_rate*, through one resampler so chunk edges leave no trace, flushed at the end."""
    resampler = StreamResampler(src_rate, dst_rate)
    async for frame in frames:
        if (out := resampler.feed(frame)).size:
            yield out
    if (tail := resampler.flush()).size:
        yield tail


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
        if sample_rate != PCM_RATE:
            frames = _resampled(frames, sample_rate, PCM_RATE)
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
