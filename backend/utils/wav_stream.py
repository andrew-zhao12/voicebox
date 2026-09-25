"""Helpers for streaming PCM audio over HTTP.

A WAV header normally carries the total data length, which a live stream
cannot know.  Writing ``0xFFFFFFFF`` in both size fields is the widely used
"unknown length" convention: ffmpeg, sox, browsers (fetch + decode) and most
players simply read such a stream to EOF.

Only ``struct`` and numpy are used so the module stays importable without the
ML stack.
"""

import struct

import numpy as np

WAV_UNKNOWN_LENGTH = 0xFFFFFFFF
WAV_HEADER_BYTES = 44


def streaming_wav_header(
    sample_rate: int,
    *,
    channels: int = 1,
    bits_per_sample: int = 16,
    data_length: int = WAV_UNKNOWN_LENGTH,
) -> bytes:
    """Build a 44-byte RIFF/WAVE PCM header.

    With the default ``data_length`` both size fields are ``0xFFFFFFFF``,
    marking an unknown-length stream.  Pass a real byte count to build a
    conventional header.
    """
    if sample_rate <= 0 or channels <= 0 or bits_per_sample <= 0 or bits_per_sample % 8:
        raise ValueError("Invalid WAV parameters")
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    riff_size = WAV_UNKNOWN_LENGTH if data_length == WAV_UNKNOWN_LENGTH else 36 + data_length
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", riff_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample),
            b"data",
            struct.pack("<I", data_length),
        ]
    )


def float_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float audio in ``[-1, 1]`` to little-endian signed 16-bit PCM.

    Accepts a 1-D mono array or a ``(channels, samples)`` array, which is
    interleaved the way WAV expects.  The conversion reproduces libsndfile's
    PCM_16 writer sample for sample (round to a 32-bit integer, then take the
    upper 16 bits), so a streamed clip is byte-identical to the same audio
    saved with ``soundfile`` by the non-streaming path.
    """
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.T.reshape(-1)
    elif samples.ndim != 1:
        raise ValueError("Expected a 1-D or (channels, samples) array")
    scaled = np.rint(np.clip(samples, -1.0, 1.0) * 2147483648.0)
    as_int32 = np.clip(scaled, -2147483648, 2147483647).astype(np.int64)
    return (as_int32 >> 16).astype("<i2").tobytes()
