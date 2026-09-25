"""Tests for the streaming WAV header and PCM16 helpers."""

import io
import struct

import numpy as np
import pytest
import soundfile as sf

from backend.utils.wav_stream import (
    WAV_HEADER_BYTES,
    WAV_UNKNOWN_LENGTH,
    float_to_pcm16_bytes,
    streaming_wav_header,
)


def test_header_layout_marks_unknown_length():
    header = streaming_wav_header(24000)

    assert len(header) == WAV_HEADER_BYTES
    assert header[0:4] == b"RIFF"
    assert struct.unpack("<I", header[4:8])[0] == WAV_UNKNOWN_LENGTH
    assert header[8:12] == b"WAVE"
    assert header[12:16] == b"fmt "
    fmt = struct.unpack("<IHHIIHH", header[16:36])
    assert fmt == (16, 1, 1, 24000, 48000, 2, 16)
    assert header[36:40] == b"data"
    assert struct.unpack("<I", header[40:44])[0] == WAV_UNKNOWN_LENGTH


def test_header_with_known_length_is_a_regular_wav():
    header = streaming_wav_header(48000, channels=2, data_length=1000)

    assert struct.unpack("<I", header[4:8])[0] == 36 + 1000
    assert struct.unpack("<I", header[40:44])[0] == 1000
    _, _, channels, rate, byte_rate, block_align, _ = struct.unpack("<IHHIIHH", header[16:36])
    assert (channels, rate, byte_rate, block_align) == (2, 48000, 192000, 4)


@pytest.mark.parametrize(
    "bad",
    [
        {"sample_rate": 0},
        {"sample_rate": 24000, "channels": 0},
        {"sample_rate": 24000, "bits_per_sample": 12},
    ],
)
def test_header_rejects_invalid_parameters(bad):
    with pytest.raises(ValueError, match="Invalid WAV parameters"):
        streaming_wav_header(**bad)


def test_pcm16_scales_rounds_and_clips_like_libsndfile():
    values = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.3, -0.3, -9e-11, -1e-5], dtype=np.float32)

    pcm = float_to_pcm16_bytes(values)

    # Large negatives floor (-0.3 -> -9831) while values below half a 32-bit
    # step round to zero (-9e-11 -> 0): the rule libsndfile applies.
    assert struct.unpack("<11h", pcm) == (0, 16384, -16384, 32767, -32768, 32767, -32768, 9830, -9831, 0, -1)


def test_pcm16_matches_libsndfile_byte_for_byte():
    rng = np.random.default_rng(3)
    audio = np.concatenate(
        [
            (rng.standard_normal(20000) * 0.4).clip(-1, 1),
            rng.standard_normal(5000) * 1e-9,  # near-zero tail such as a fade-out
        ]
    ).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format="WAV")
    written = buffer.getvalue()
    data_offset = written.index(b"data") + 8

    assert float_to_pcm16_bytes(audio) == written[data_offset:]


def test_pcm16_interleaves_channels():
    stereo = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)  # (channels, samples)

    values = struct.unpack("<4h", float_to_pcm16_bytes(stereo))

    left = struct.unpack("<2h", float_to_pcm16_bytes(stereo[0]))
    right = struct.unpack("<2h", float_to_pcm16_bytes(stereo[1]))
    assert list(values) == [left[0], right[0], left[1], right[1]]


def test_pcm16_rejects_3d_input():
    with pytest.raises(ValueError, match="Expected a 1-D"):
        float_to_pcm16_bytes(np.zeros((1, 1, 4), dtype=np.float32))


def test_header_plus_pcm_round_trips_through_soundfile():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(2400) * 0.3).clip(-1, 1).astype(np.float32)
    pcm = float_to_pcm16_bytes(audio)

    decoded, rate = sf.read(io.BytesIO(streaming_wav_header(24000, data_length=len(pcm)) + pcm), dtype="float32")

    assert rate == 24000
    assert decoded.shape == audio.shape
    assert np.max(np.abs(decoded - audio)) <= 1.0 / 32768 + 1e-6
