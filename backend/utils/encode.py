"""Encode streamed PCM into the formats OpenAI clients ask for.

``wav`` and ``pcm`` are produced in Python and stream chunk by chunk.  The
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

FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
EXTENSIONS = {"mp3": "mp3", "opus": "ogg", "aac": "aac", "flac": "flac", "wav": "wav", "pcm": "pcm"}

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
    return [fmt for fmt in FORMATS if fmt in ("wav", "pcm") or has_ffmpeg or _soundfile_supports(fmt)]


def streams_incrementally(fmt: str) -> bool:
    """Whether bytes leave before synthesis has finished."""
    return fmt in ("wav", "pcm") or ffmpeg_path() is not None


def encode_bytes(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Encode a whole clip with libsndfile (``mp3``, ``flac``, ``opus``, ``wav``)."""
    if fmt == "wav":
        return streaming_wav_header(sample_rate, data_length=len(audio) * 2) + float_to_pcm16_bytes(audio)
    if fmt == "pcm":
        return float_to_pcm16_bytes(audio)
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
