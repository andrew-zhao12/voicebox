"""Streamed encoding: wav/pcm in Python, compressed formats through ffmpeg or libsndfile (torch-free)."""

from __future__ import annotations

import io
import shutil

import numpy as np
import pytest
import soundfile as sf

from backend.utils import encode
from backend.utils.wav_stream import float_to_pcm16_bytes, streaming_wav_header

SR = 24000


def tone(seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


async def frames(*chunks):
    for chunk in chunks:
        yield chunk


async def collect(gen) -> bytes:
    return b"".join([chunk async for chunk in gen])


def test_available_formats_always_include_wav_and_pcm():
    formats = encode.available_formats()
    assert "wav" in formats
    assert "pcm" in formats
    assert formats == [fmt for fmt in encode.FORMATS if fmt in formats]


async def test_wav_and_pcm_stream_chunk_by_chunk():
    a, b = tone(0.1), tone(0.2)
    chunks = [chunk async for chunk in encode.encode_stream(frames(a, b), SR, "wav")]
    assert chunks == [streaming_wav_header(SR), float_to_pcm16_bytes(a), float_to_pcm16_bytes(b)]
    chunks = [chunk async for chunk in encode.encode_stream(frames(a, b), SR, "pcm")]
    assert chunks == [float_to_pcm16_bytes(a), float_to_pcm16_bytes(b)]


def test_encode_bytes_wav_carries_the_real_length():
    audio = tone(0.1)
    data = encode.encode_bytes(audio, SR, "wav")
    decoded, rate = sf.read(io.BytesIO(data), dtype="float32")
    assert rate == SR
    assert len(decoded) == len(audio)


@pytest.mark.parametrize("fmt", ["mp3", "flac", "opus"])
async def test_libsndfile_fallback_encodes_the_whole_clip(monkeypatch, fmt):
    monkeypatch.setattr(encode, "ffmpeg_path", lambda: None)
    if fmt not in encode.available_formats():
        pytest.skip(f"libsndfile here cannot write {fmt}")
    assert not encode.streams_incrementally(fmt)
    chunks = [chunk async for chunk in encode.encode_stream(frames(tone(0.3), tone(0.2)), SR, fmt)]
    assert len(chunks) == 1
    decoded, rate = sf.read(io.BytesIO(chunks[0]), dtype="float32")
    assert rate in (SR, 48000)
    assert 0.4 < len(decoded) / rate < 0.7  # mp3 pads the ends


async def test_unsupported_formats_raise(monkeypatch):
    monkeypatch.setattr(encode, "ffmpeg_path", lambda: None)
    with pytest.raises(encode.UnsupportedFormatError):
        await collect(encode.encode_stream(frames(tone(0.1)), SR, "aac"))
    with pytest.raises(encode.UnsupportedFormatError):
        await collect(encode.encode_stream(frames(tone(0.1)), SR, "bogus"))
    assert "aac" not in encode.available_formats()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("fmt", ["mp3", "opus", "aac", "flac"])
async def test_ffmpeg_pipe_produces_a_decodable_stream(fmt):
    assert encode.streams_incrementally(fmt)
    data = await collect(encode.encode_stream(frames(tone(0.3), tone(0.3)), SR, fmt))
    assert len(data) > 100
    if fmt == "aac":
        assert data[0] == 0xFF  # ADTS sync word
        assert data[1] & 0xF0 == 0xF0
        return
    decoded, rate = sf.read(io.BytesIO(data), dtype="float32")
    assert rate in (SR, 48000)
    assert 0.5 < len(decoded) / rate < 0.8


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_ffmpeg_is_stopped_when_the_consumer_leaves():
    seen = []

    async def endless():
        while True:
            seen.append(1)
            yield tone(0.05)

    stream = encode.encode_stream(endless(), SR, "mp3")
    first = await stream.__anext__()
    assert first
    await stream.aclose()
    produced = len(seen)
    assert produced < 10_000  # the feeder stopped with the consumer
