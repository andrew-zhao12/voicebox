"""Resample streamed mono audio from one rate to another, chunk by chunk.

Each output sample is a Kaiser-windowed sinc evaluated at its exact position
in the input.  That position is kept as an integer ratio, so a long stream
never drifts, and the filter is centred on it, so the output keeps the input's
timing.  The filter's lookahead, about 25 samples at the lower of the two
rates (1.6 ms for 48 kHz to 16 kHz, 3.3 ms for 8 kHz to 16 kHz), is held back
until more input arrives or ``flush()`` releases it.  The cutoff sits below
the Nyquist frequency of the lower rate, which removes what would alias when
downsampling and the images when upsampling.

Only numpy is used, so the module stays importable without the ML stack.
"""

import functools
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Fractions of the lower rate: the pass band ends at 0.4, the stop band starts
# at 0.5 (that rate's Nyquist frequency) and the cutoff sits halfway.
_CUTOFF = 0.45
_TRANSITION = 0.1
# Stop band attenuation; a window design has the same ripple in the pass band.
_ATTENUATION_DB = 80.0
# Rate pairs with more phases than this (unusual ones such as 44099 -> 16000)
# interpolate between a fixed set of tabulated phases instead.
_MAX_EXACT_PHASES = 1024
_INTERPOLATED_PHASES = 512
# Outputs computed per vectorised step, which bounds the (outputs x taps) temporaries.
_BLOCK = 4096


@functools.lru_cache(maxsize=32)
def _design(src_rate: int, dst_rate: int, phases: int) -> tuple[np.ndarray, int]:
    """Tabulate the filter at ``phases + 1`` offsets in [0, 1]; returns (table, half width in input samples)."""
    ratio = max(1.0, src_rate / dst_rate)
    # Kaiser's estimates for this attenuation and transition width.  The band
    # narrows (relative to the input rate) when downsampling, so the kernel widens.
    order = (_ATTENUATION_DB - 7.95) / (2.285 * 2 * math.pi * _TRANSITION / ratio)
    beta = 0.1102 * (_ATTENUATION_DB - 8.7)
    half = math.ceil(order / 2)
    cutoff = _CUTOFF / ratio  # cycles per input sample
    # Row q weights inputs i - half + 1 .. i + half for an output at input position i + q / phases.
    tau = np.arange(phases + 1)[:, None] / phases + (half - 1) - np.arange(2 * half)
    window = np.i0(beta * np.sqrt(np.clip(1 - (tau / half) ** 2, 0, None))) / np.i0(beta)
    table = 2 * cutoff * np.sinc(2 * cutoff * tau) * window
    table /= table.sum(axis=1, keepdims=True)  # unity gain at DC for every phase
    table = table.astype(np.float32)
    table.flags.writeable = False  # shared by every resampler for this rate pair
    return table, half


class StreamResampler:
    """Band-limited resampling of a mono float32 stream from ``src_rate`` to ``dst_rate``.

    ``feed()`` returns every output sample the input so far fully determines;
    ``flush()`` treats the input as followed by silence and returns the rest,
    once.  Every output sample is computed by the same arithmetic however the
    input was chunked, so chunked output equals a single pass.  After
    ``flush()`` there are ``ceil(n * dst_rate / src_rate)`` output samples for
    ``n`` input samples, and output sample ``k`` is the input at time
    ``k / dst_rate``.  Equal rates pass audio through untouched.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError(f"sample rates must be positive, got {src_rate} -> {dst_rate}")
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        common = math.gcd(src_rate, dst_rate)
        self._up = dst_rate // common
        self._down = src_rate // common
        self._exact = self._up <= _MAX_EXACT_PHASES
        self._table: np.ndarray | None = None
        self._half = 0
        if src_rate != dst_rate:
            phases = self._up if self._exact else _INTERPOLATED_PHASES
            self._table, self._half = _design(src_rate, dst_rate, phases)
        self.reset()

    def reset(self) -> None:
        """Drop all state: the next sample fed starts a new stream at time zero."""
        # Silence precedes the stream; output 0's window reaches half - 1 samples into it.
        self._buffer = np.zeros(max(self._half - 1, 0), dtype=np.float32)
        self._start = 1 - self._half  # input index of _buffer[0]
        self._produced = 0
        self._flushed = False

    def feed(self, samples: np.ndarray) -> np.ndarray:
        """Add input samples (flattened to mono float32); return the output they complete, possibly none.

        Raises:
            RuntimeError: After ``flush()``, until ``reset()``.
        """
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._flushed:
            raise RuntimeError("StreamResampler.feed() after flush(); call reset() to start a new stream")
        if self._table is None or audio.size == 0:
            return audio
        self._buffer = np.concatenate([self._buffer, audio])
        return self._drain()

    def flush(self) -> np.ndarray:
        """Return the output still waiting on the filter's lookahead; later calls return nothing."""
        if self._flushed or self._table is None:
            self._flushed = True
            return np.zeros(0, dtype=np.float32)
        self._flushed = True
        self._buffer = np.concatenate([self._buffer, np.zeros(self._half, dtype=np.float32)])
        return self._drain()

    def _drain(self) -> np.ndarray:
        # Output k sits at input position k * down / up and is ready once the
        # input reaches half samples past it.
        last = self._start + self._buffer.size - 1 - self._half
        total = max(0, ((last + 1) * self._up + self._down - 1) // self._down)
        if total == self._produced:
            return np.zeros(0, dtype=np.float32)
        blocks = [
            self._compute(np.arange(k, min(k + _BLOCK, total), dtype=np.int64))
            for k in range(self._produced, total, _BLOCK)
        ]
        self._produced = total
        keep = total * self._down // self._up - self._half + 1  # the next output's first input
        self._buffer = self._buffer[keep - self._start :]
        self._start = keep
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    def _compute(self, outputs: np.ndarray) -> np.ndarray:
        index, phase = np.divmod(outputs * self._down, self._up)
        windows = sliding_window_view(self._buffer, 2 * self._half)[index - (self._half - 1) - self._start]
        # einsum, unlike a BLAS matmul, sums every row in the same fixed order,
        # so a sample's value does not depend on how the input was chunked.
        if self._exact:
            return np.einsum("ij,ij->i", windows, self._table[phase])
        row, rest = np.divmod(phase * _INTERPOLATED_PHASES, self._up)
        low = np.einsum("ij,ij->i", windows, self._table[row])
        high = np.einsum("ij,ij->i", windows, self._table[row + 1])
        return low + (rest / self._up).astype(np.float32) * (high - low)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a whole clip: the samples ``StreamResampler`` would stream, as one new float32 array."""
    resampler = StreamResampler(src_rate, dst_rate)
    return np.concatenate([resampler.feed(audio), resampler.flush()])
