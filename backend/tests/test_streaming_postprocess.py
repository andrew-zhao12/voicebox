"""Tests for the chunk-wise normalizer, runaway cut detection and stateful chunk effects."""

import numpy as np
import pytest

from backend.utils.audio import StreamingNormalizer, find_tts_runaway_cut, has_tts_runaway, normalize_audio

SAMPLE_RATE = 1000


def test_normalizer_single_chunk_matches_normalize_audio():
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(4000) * 0.05).astype(np.float32)

    assert np.allclose(StreamingNormalizer().process(audio), normalize_audio(audio), atol=1e-6)


def test_normalizer_locks_gain_from_first_chunk_with_signal():
    normalizer = StreamingNormalizer()
    silent = np.zeros(100, dtype=np.float32)

    assert np.array_equal(normalizer.process(silent), silent)
    assert normalizer.gain is None

    loud = np.full(100, 0.2, dtype=np.float32)
    normalizer.process(loud)
    gain = normalizer.gain
    assert gain is not None

    quiet = np.full(100, 0.02, dtype=np.float32)
    assert np.allclose(normalizer.process(quiet), np.clip(quiet * gain, -0.85, 0.85))
    assert normalizer.gain == gain


def test_normalizer_caps_gain_and_clips_peaks():
    normalizer = StreamingNormalizer()

    out = normalizer.process(np.full(100, 0.002, dtype=np.float32))  # uncapped gain would be 50x

    assert np.isclose(normalizer.gain, 10.0)
    assert np.allclose(out, 0.02, atol=1e-6)
    assert np.all(np.abs(normalizer.process(np.full(100, 0.5, dtype=np.float32))) <= 0.85)


def test_runaway_cut_points_at_start_of_the_silence():
    speech = np.full(2 * SAMPLE_RATE, 0.2, dtype=np.float32)
    gap = np.zeros(2500, dtype=np.float32)
    noise = np.full(SAMPLE_RATE, 0.8, dtype=np.float32)
    audio = np.concatenate([speech, gap, noise])

    assert find_tts_runaway_cut(audio, SAMPLE_RATE) == len(speech)
    assert has_tts_runaway(audio, SAMPLE_RATE) is True


def test_runaway_cut_is_none_for_stable_audio():
    speech = np.full(SAMPLE_RATE, 0.2, dtype=np.float32)
    audio = np.concatenate([speech, np.zeros(1200, dtype=np.float32), speech])

    assert find_tts_runaway_cut(audio, SAMPLE_RATE) is None
    assert has_tts_runaway(audio, SAMPLE_RATE) is False


@pytest.mark.parametrize("effect_type", ["highpass", "reverb", "compressor", "delay"])
def test_streaming_effects_carry_state_across_chunks(effect_type):
    from backend.utils.effects import StreamingEffects, apply_effects

    rng = np.random.default_rng(2)
    audio = (rng.standard_normal(24000) * 0.2).astype(np.float32)
    chain = [{"type": effect_type}]

    whole = apply_effects(audio, 24000, chain)
    streaming = StreamingEffects(chain)
    chunked = np.concatenate([streaming.process(part, 24000) for part in np.array_split(audio, 5)])

    assert chunked.shape == whole.shape
    assert np.allclose(chunked, whole, atol=1e-4)


def test_streaming_effects_passes_through_empty_chains_and_chunks():
    from backend.utils.effects import StreamingEffects

    chunk = np.full(10, 0.1, dtype=np.float32)
    assert StreamingEffects([]).process(chunk, 24000) is chunk
    empty = np.array([], dtype=np.float32)
    assert StreamingEffects([{"type": "reverb"}]).process(empty, 24000) is empty
