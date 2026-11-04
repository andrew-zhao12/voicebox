"""Preserve digital silence while retaining Pipecat's streaming resampler behavior."""

import numpy as np
import soxr
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler


class _FloatPCMStream:
    """Use float SoXR filtering and explicit PCM rounding instead of integer dither."""

    def __init__(self, in_rate, out_rate, quality):
        self._stream = soxr.ResampleStream(
            in_rate, out_rate, num_channels=1, quality=quality, dtype="float32"
        )

    def resample_chunk(self, audio, last=False):
        samples = self._stream.resample_chunk(audio.astype(np.float32), last=last)
        return np.clip(np.rint(samples), -32768, 32767).astype(np.int16)

    def clear(self):
        self._stream.clear()


class VoiceboxSTTResampler(SOXRStreamAudioResampler):
    """Keep upstream buffering, filter tails and resets, without injecting idle noise.

    Pipecat 1.15's int16 SoXR stream dithers exact zeros to +/-1 samples. Whisper
    can hallucinate on that continuous signal in manually committed sessions.
    Float filtering preserves silence and even one-LSB input without a VAD gate.
    """

    def _initialize(self, in_rate, out_rate):
        super()._initialize(in_rate, out_rate)
        self._soxr_stream = _FloatPCMStream(in_rate, out_rate, self._quality)
