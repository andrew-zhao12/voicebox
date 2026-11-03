"""Stateful band-limited resampling: chunking, length, timing and quality (torch-free)."""

import itertools

import numpy as np
import pytest

from backend.utils.resample import StreamResampler, resample

# Engine output to OpenAI's 24 kHz pcm, client audio to Whisper's 16 kHz, and telephony.
QUALITY_PAIRS = [
    (48000, 24000),
    (24000, 16000),
    (44100, 16000),
    (22050, 16000),
    (48000, 16000),
    (8000, 16000),
    (16000, 24000),
    (24000, 48000),
    (24000, 8000),
]
DOWNSAMPLING = [(src, dst) for src, dst in QUALITY_PAIRS if dst < src]
# Many phases, a large ratio, and two pairs past the exact phase table (interpolated phases).
ODD_PAIRS = [(44100, 48000), (48000, 8000), (44099, 16000), (16000, 44099)]


def tone(freq: float, rate: int, length: int, amplitude: float = 0.5) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * np.arange(length) / rate)


def stream(resampler: StreamResampler, audio: np.ndarray, chunk: int) -> np.ndarray:
    parts = [resampler.feed(audio[i : i + chunk]) for i in range(0, len(audio), chunk)]
    return np.concatenate([*parts, resampler.flush()])


def interior(signal: np.ndarray, rate: int) -> np.ndarray:
    """Drop 20 ms at both ends, where the filter window reaches past the clip."""
    edge = rate // 50
    return signal[edge:-edge]


def snr_db(signal: np.ndarray, ideal: np.ndarray) -> float:
    return 10 * np.log10(np.sum(ideal**2) / np.sum((signal - ideal) ** 2))


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def test_equal_rates_pass_audio_through_untouched():
    audio = np.random.default_rng(0).uniform(-1, 1, 1000).astype(np.float32)
    resampler = StreamResampler(24000, 24000)

    assert np.array_equal(resampler.feed(audio[:10]), audio[:10])
    assert np.array_equal(resampler.feed(audio[10:]), audio[10:])
    assert resampler.flush().size == 0
    assert np.array_equal(resample(audio, 16000, 16000), audio)
    assert StreamResampler(16000, 16000).feed(np.zeros(3)).dtype == np.float32


@pytest.mark.parametrize(("src", "dst"), [*QUALITY_PAIRS, *ODD_PAIRS])
def test_chunked_output_matches_a_single_pass(src, dst):
    rng = np.random.default_rng(src * 7 + dst)
    audio = rng.uniform(-0.9, 0.9, src // 2).astype(np.float32)
    resampler = StreamResampler(src, dst)
    parts, start = [], 0
    while start < len(audio):
        size = int(rng.choice([0, 1, 1, 2, 3, 17, 160, 441, 1200, 5000]))
        parts.append(resampler.feed(audio[start : start + size]))
        start += size
    chunked = np.concatenate([*parts, resampler.flush(), resampler.flush()])

    assert chunked.dtype == np.float32
    np.testing.assert_allclose(chunked, resample(audio, src, dst), rtol=0, atol=1e-6)


@pytest.mark.parametrize(("src", "dst"), [*QUALITY_PAIRS, *ODD_PAIRS])
def test_flush_leaves_the_scaled_length_rounded_up(src, dst):
    for length in (0, 1, 2, 3, 101, 4801):
        out = resample(np.zeros(length, dtype=np.float32), src, dst)
        assert len(out) == (length * dst + src - 1) // src, length


@pytest.mark.parametrize("src", [22050, 44100])
def test_a_minute_in_10_ms_chunks_neither_drifts_nor_changes_length(src):
    dst, freq = 16000, 1234.5
    audio = tone(freq, src, 60 * src).astype(np.float32)
    bounds = [round(i * src / 100) for i in range(6001)]
    resampler = StreamResampler(src, dst)
    pieces = [resampler.feed(audio[start:end]) for start, end in itertools.pairwise(bounds)]
    out = np.concatenate([*pieces, resampler.flush()])

    assert abs(len(out) - round(len(audio) * dst / src)) <= 1
    # A position that drifted would leave the last second out of phase.
    last_second = slice(len(out) - dst, len(out) - dst // 50)
    assert snr_db(out[last_second], tone(freq, dst, len(out))[last_second]) >= 60


@pytest.mark.parametrize(("src", "dst"), [*QUALITY_PAIRS, *ODD_PAIRS])
def test_an_impulse_comes_out_when_it_went_in(src, dst):
    length = src // 4
    for at in (0, 1, 333, length // 2, length - 1):
        audio = np.zeros(length, dtype=np.float32)
        audio[at] = 1.0
        out = stream(StreamResampler(src, dst), audio, src // 100)
        assert abs(int(np.argmax(np.abs(out))) - at * dst / src) <= 1, at


@pytest.mark.parametrize(("src", "dst"), [*QUALITY_PAIRS, *ODD_PAIRS])
def test_in_band_sines_come_through_clean(src, dst):
    top = 0.4 * min(src, dst)
    for freq in (100, 440, top / 4, top / 2, 3 * top / 4, top):
        audio = tone(freq, src, src // 2).astype(np.float32)
        out = stream(StreamResampler(src, dst), audio, src // 50)
        ideal = tone(freq, dst, len(out))
        assert snr_db(interior(out, dst), interior(ideal, dst)) >= 60, freq


@pytest.mark.parametrize(("src", "dst"), DOWNSAMPLING)
def test_tones_above_the_output_band_are_removed(src, dst):
    """Unfiltered, they would fold back into the band at full level (10 kHz lands on 6 kHz at 16 kHz)."""
    for freq in [f for f in (0.55 * dst, 0.625 * dst, 0.9 * dst) if f < src / 2]:
        audio = tone(freq, src, src // 2).astype(np.float32)
        out = stream(StreamResampler(src, dst), audio, src // 50)
        assert 20 * np.log10(rms(interior(out, dst)) / rms(audio)) <= -50, freq


def test_flush_releases_the_lookahead_once_and_reset_starts_over():
    audio = tone(1000, 48000, 4800).astype(np.float32)
    resampler = StreamResampler(48000, 16000)
    head = resampler.feed(audio)
    tail = resampler.flush()

    assert len(head) + len(tail) == 1600
    assert 0 < len(tail) <= 16000 * 0.002  # under 2 ms held back
    assert resampler.flush().size == 0
    with pytest.raises(RuntimeError, match="after flush"):
        resampler.feed(audio)
    resampler.reset()
    again = np.concatenate([resampler.feed(audio), resampler.flush()])
    assert np.array_equal(again, np.concatenate([head, tail]))


@pytest.mark.parametrize(("src", "dst"), [(0, 16000), (16000, -1)])
def test_rates_must_be_positive(src, dst):
    with pytest.raises(ValueError, match="positive"):
        StreamResampler(src, dst)
