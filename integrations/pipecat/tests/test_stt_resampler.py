"""Real SoXR regressions: no idle dither, quiet speech and streaming tails survive."""

import numpy as np
import pytest
import soxr

from services import voicebox_stt
from stt_resampler import VoiceboxSTTResampler


@pytest.mark.asyncio
async def test_realtime_service_keeps_idle_audio_exactly_silent():
    service = voicebox_stt(api_key="test-client", base_url="http://localhost:1")
    chunks = [await service._resampler.resample(bytes(640), 16000, 24000) for _ in range(100)]
    chunks.append(await service._resampler.flush())
    samples = np.frombuffer(b"".join(chunks), dtype=np.int16)
    assert len(samples) == 48000
    assert not np.any(samples)


@pytest.mark.asyncio
@pytest.mark.parametrize("packet_samples", [160, 320, 800])
async def test_quiet_signal_and_filter_tail_match_continuous_float_resampling(packet_samples):
    resampler = VoiceboxSTTResampler(clear_after_secs=None)
    source = np.concatenate(
        [
            np.zeros(1600, dtype=np.int16),
            np.ones(1600, dtype=np.int16),
            np.zeros(1600, dtype=np.int16),
        ]
    )
    parts = np.split(source, len(source) // packet_samples)
    chunks = [await resampler.resample(part.tobytes(), 16000, 24000) for part in parts]
    chunks.append(await resampler.flush())
    actual = np.frombuffer(b"".join(chunks), dtype=np.int16)
    reference = np.rint(
        soxr.resample(source.astype(np.float32), 16000, 24000, quality="VHQ")
    ).astype(np.int16)
    np.testing.assert_array_equal(actual, reference)
    assert np.count_nonzero(actual) >= 2000
    # The filter buffers speech: zero input packets must still drain that tail.
    assert any(any(chunk) for chunk in chunks[3200 // packet_samples : len(parts)])
    assert not any(await resampler.resample(bytes(6400), 16000, 24000))


@pytest.mark.asyncio
async def test_reset_discards_tail_and_same_rate_is_bit_exact():
    resampler = VoiceboxSTTResampler(clear_after_secs=None)
    await resampler.resample(np.full(100, 200, dtype=np.int16).tobytes(), 16000, 24000)
    await resampler.reset()
    assert not any(await resampler.flush())
    pcm = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16).tobytes()
    assert await resampler.resample(pcm, 24000, 24000) == pcm
