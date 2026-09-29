"""Live transcription core: audio in, utterances out, stable partial text along the way.

Pure numpy and importable without torch, so the state machine is unit
tested in CI.  A ``TranscriptionSession`` receives PCM16 chunks, resamples
them to Whisper's 16 kHz, finds utterances with an energy VAD (or waits for
manual commits), and hands out ``Work`` items: a *partial* every
``partial_interval_ms`` while speech goes on, and a *final* when an
utterance ends or reaches ``MAX_UTTERANCE_S``.  The caller (the WebSocket
route) runs Whisper on the work and feeds hypotheses back through
``partial_text``/``final_text``, which turn them into OpenAI-Realtime-style
events with append-only deltas (LocalAgreement: only the word prefix two
consecutive hypotheses agree on is emitted, so nothing needs retracting).
"""

from __future__ import annotations

import base64
import binascii
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

TARGET_RATE = 16000
FRAME_MS = 20
MAX_UTTERANCE_S = 30.0  # one Whisper window
# Trailing silence kept on an utterance the VAD closed; the rest of the
# silence_duration_ms is trimmed so Whisper does not hallucinate over it.
TAIL_KEEP_MS = 200
# Past this length an utterance is cut at the next short pause, so the hard
# cut at MAX_UTTERANCE_S (which lands mid-word and makes Whisper hallucinate
# at the edge) is the exception rather than the rule.
SOFT_CAP_S = 24.0
SUPPORTED_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
DELTA_EVENT = "conversation.item.input_audio_transcription.delta"
COMPLETED_EVENT = "conversation.item.input_audio_transcription.completed"
FAILED_EVENT = "conversation.item.input_audio_transcription.failed"


class ProtocolError(ValueError):
    """A client message the session cannot accept; ``code`` is the error code sent back."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SessionConfig:
    input_sample_rate: int = 24000
    model: str | None = None
    language: str | None = None
    vad: bool = True
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    partial_interval_ms: int = 1000

    @classmethod
    def from_update(cls, base: SessionConfig, payload: dict) -> SessionConfig:
        """Apply a ``transcription_session.update`` payload (OpenAI Realtime field names) to *base*."""
        session = payload.get("session", payload)
        if not isinstance(session, dict):
            raise ProtocolError("invalid_session", "session must be an object")
        values: dict = {}
        fmt = session.get("input_audio_format", "pcm16")
        if fmt != "pcm16":
            raise ProtocolError("unsupported_audio_format", "only input_audio_format 'pcm16' is supported")
        if "input_sample_rate" in session:
            rate = session["input_sample_rate"]
            if rate not in SUPPORTED_RATES:
                raise ProtocolError("unsupported_sample_rate", f"input_sample_rate must be one of {SUPPORTED_RATES}")
            values["input_sample_rate"] = int(rate)
        transcription = session.get("input_audio_transcription")
        if transcription is not None:
            if not isinstance(transcription, dict):
                raise ProtocolError("invalid_session", "input_audio_transcription must be an object")
            if "model" in transcription:
                values["model"] = str(transcription["model"]) if transcription["model"] else None
            if "language" in transcription:
                language = transcription["language"]
                values["language"] = str(language) if language else None
        if "turn_detection" in session:
            turn = session["turn_detection"]
            if turn is None:
                values["vad"] = False
            elif isinstance(turn, dict):
                if turn.get("type", "server_vad") != "server_vad":
                    raise ProtocolError("unsupported_turn_detection", "turn_detection.type must be server_vad or null")
                values["vad"] = True
                for key in ("threshold", "prefix_padding_ms", "silence_duration_ms"):
                    if key in turn:
                        values[key] = turn[key]
            else:
                raise ProtocolError("invalid_session", "turn_detection must be an object or null")
        if "partial_interval_ms" in session:
            values["partial_interval_ms"] = session["partial_interval_ms"]
        config = replace(base, **values)
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ProtocolError("invalid_session", "threshold must be between 0 and 1")
        if not 0 <= int(self.prefix_padding_ms) <= 2000:
            raise ProtocolError("invalid_session", "prefix_padding_ms must be between 0 and 2000")
        if not 100 <= int(self.silence_duration_ms) <= 5000:
            raise ProtocolError("invalid_session", "silence_duration_ms must be between 100 and 5000")
        if not 300 <= int(self.partial_interval_ms) <= 10000:
            raise ProtocolError("invalid_session", "partial_interval_ms must be between 300 and 10000")

    def to_event(self) -> dict:
        return {
            "input_audio_format": "pcm16",
            "input_sample_rate": self.input_sample_rate,
            "input_audio_transcription": {"model": self.model or "whisper-1", "language": self.language},
            "turn_detection": (
                {
                    "type": "server_vad",
                    "threshold": self.threshold,
                    "prefix_padding_ms": self.prefix_padding_ms,
                    "silence_duration_ms": self.silence_duration_ms,
                }
                if self.vad
                else None
            ),
            "partial_interval_ms": self.partial_interval_ms,
        }


def decode_pcm16(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit mono to float32 in [-1, 1] (a trailing odd byte is dropped)."""
    usable = len(data) - (len(data) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0


def decode_base64_audio(text: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ProtocolError("invalid_audio", "audio must be base64 PCM16") from e


def _lowpass_kernel(cutoff: float, sample_rate: int, taps: int = 63) -> np.ndarray:
    """Windowed-sinc FIR low-pass (Hann), unity DC gain."""
    n = np.arange(taps) - (taps - 1) / 2
    fc = cutoff / sample_rate
    kernel = 2 * fc * np.sinc(2 * fc * n)
    kernel *= np.hanning(taps)
    return (kernel / kernel.sum()).astype(np.float32)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample float32 audio; a low-pass filter precedes downsampling, then linear interpolation."""
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    signal = audio.astype(np.float32, copy=False)
    if dst_rate < src_rate:
        signal = np.convolve(signal, _lowpass_kernel(0.45 * dst_rate, src_rate), mode="same")
    length = max(1, round(audio.size * dst_rate / src_rate))
    positions = np.arange(length) * (src_rate / dst_rate)
    return np.interp(positions, np.arange(audio.size), signal).astype(np.float32)


def _rms_db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
    return 20 * math.log10(max(rms, 1e-6))


class EnergyVAD:
    """Frame-energy voice activity with an adaptive noise floor.

    The floor drops to a quiet frame at once and rises at most ``RISE_DB_PER_S``
    while frames stay loud (never above ``MAX_FLOOR_DB``), so speech that
    starts immediately or runs for half a minute does not lift it out from
    under itself.  A frame is speech when it exceeds the floor by a margin
    that ``threshold`` (0..1) maps onto 6..24 dB, and never below
    ``MIN_SPEECH_DB``.
    """

    RISE_DB_PER_S = 0.25
    MAX_FLOOR_DB = -40.0
    MIN_SPEECH_DB = -45.0

    def __init__(self, threshold: float = 0.5, sample_rate: int = TARGET_RATE, frame_ms: int = FRAME_MS) -> None:
        self.frame_samples = sample_rate * frame_ms // 1000
        self.margin_db = 6.0 + 18.0 * min(1.0, max(0.0, threshold))
        self.floor_db = -50.0
        self._rise_per_frame = self.RISE_DB_PER_S * frame_ms / 1000

    def is_speech(self, frame: np.ndarray) -> bool:
        level = _rms_db(frame)
        self.floor_db = min(level, self.floor_db + self._rise_per_frame, self.MAX_FLOOR_DB)
        return level > max(self.floor_db + self.margin_db, self.MIN_SPEECH_DB)


@dataclass
class Work:
    """Audio for the transcriber: a running partial or the final pass of an utterance."""

    kind: Literal["partial", "final"]
    item_id: str
    audio: np.ndarray  # float32 at TARGET_RATE
    language: str | None
    model: str | None


class LocalAgreement:
    """Append-only deltas: emit the words two consecutive hypotheses share beyond what was already sent."""

    def __init__(self) -> None:
        self.previous: list[str] = []
        self.emitted: list[str] = []

    def partial(self, hypothesis: str) -> str | None:
        words = hypothesis.split()
        stable = 0
        for a, b in zip(self.previous, words, strict=False):
            if a.casefold() == b.casefold():
                stable += 1
            else:
                break
        self.previous = words
        if stable <= len(self.emitted):
            return None
        if [w.casefold() for w in words[: len(self.emitted)]] != [w.casefold() for w in self.emitted]:
            return None  # the hypothesis revised what was already sent; wait for the final
        delta = words[len(self.emitted) : stable]
        self.emitted = words[:stable]
        return (" " if self.emitted[: -len(delta)] else "") + " ".join(delta)

    def final(self, transcript: str) -> str | None:
        """The delta that completes the emitted text, or ``None`` when the final replaces it."""
        words = transcript.split()
        if [w.casefold() for w in words[: len(self.emitted)]] != [w.casefold() for w in self.emitted]:
            return None
        rest = words[len(self.emitted) :]
        if not rest:
            return ""
        return (" " if self.emitted else "") + " ".join(rest)


@dataclass
class _Utterance:
    item_id: str
    audio: list[np.ndarray] = field(default_factory=list)
    samples: int = 0
    start_ms: int = 0
    since_partial: int = 0  # samples added since the last partial was cut
    quiet_frames: int = 0  # consecutive non-speech frames at the tail
    agreement: LocalAgreement = field(default_factory=LocalAgreement)


class TranscriptionSession:
    """State machine for one client connection; every method returns the events to send."""

    def __init__(self, config: SessionConfig, *, item_ids: Callable[[], str] | None = None) -> None:
        self.config = config
        self._new_id = item_ids or (lambda: f"item_{uuid.uuid4().hex[:12]}")
        self._pending = np.zeros(0, dtype=np.float32)  # resampled audio not yet framed
        self._vad = EnergyVAD(config.threshold)
        self._utterance: _Utterance | None = None
        self._prefix: list[np.ndarray] = []  # recent non-speech frames kept as padding
        self._silence_frames = 0
        self._speech_frames = 0
        self._total_ms = 0  # audio received, in ms at TARGET_RATE
        self.work: list[Work] = []
        self.utterances = 0
        # Utterances whose final transcript is pending, so a late partial cannot follow the final.
        self._finalized: set[str] = set()
        self._agreements: dict[str, LocalAgreement] = {}

    # -- configuration -----------------------------------------------------

    def update(self, config: SessionConfig) -> list[dict]:
        self.config = config
        self._vad = EnergyVAD(config.threshold)
        return [{"type": "transcription_session.updated", "session": config.to_event()}]

    # -- audio in ------------------------------------------------------------

    def append(self, pcm16: bytes) -> list[dict]:
        samples = decode_pcm16(pcm16)
        if samples.size == 0:
            return []
        audio = resample(samples, self.config.input_sample_rate, TARGET_RATE)
        self._pending = np.concatenate([self._pending, audio]) if self._pending.size else audio
        events: list[dict] = []
        frame = self._vad.frame_samples
        while self._pending.size >= frame:
            chunk, self._pending = self._pending[:frame], self._pending[frame:]
            events.extend(self._frame(chunk))
        return events

    def _frame(self, frame: np.ndarray) -> list[dict]:
        events: list[dict] = []
        cfg = self.config
        self._total_ms += FRAME_MS
        speech = self._vad.is_speech(frame)
        if not cfg.vad:
            # Manual turns: everything since the last commit belongs to the utterance
            # (the VAD result only finds a pause for the soft cap).
            if self._utterance is None:
                self._utterance = _Utterance(item_id=self._new_id(), start_ms=self._total_ms - FRAME_MS)
            events.extend(self._add_to_utterance(frame, speech))
            return events

        if self._utterance is None:
            self._prefix.append(frame)
            keep = max(1, cfg.prefix_padding_ms // FRAME_MS)
            del self._prefix[:-keep]
            if speech:
                self._speech_frames += 1
                min_frames = max(1, 150 // FRAME_MS)
                if self._speech_frames >= min_frames:
                    start_ms = max(0, self._total_ms - FRAME_MS * len(self._prefix))
                    self._utterance = _Utterance(item_id=self._new_id(), start_ms=start_ms)
                    self._silence_frames = 0
                    events.append(
                        {
                            "type": "input_audio_buffer.speech_started",
                            "item_id": self._utterance.item_id,
                            "audio_start_ms": start_ms,
                        }
                    )
                    for padded in self._prefix:  # includes this frame
                        events.extend(self._add_to_utterance(padded, True))
                    self._prefix = []
            else:
                self._speech_frames = 0
            return events

        events.extend(self._add_to_utterance(frame, speech))
        if speech:
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            if self._silence_frames * FRAME_MS >= cfg.silence_duration_ms and self._utterance is not None:
                trim_ms = max(0, cfg.silence_duration_ms - TAIL_KEEP_MS)
                events.append(
                    {
                        "type": "input_audio_buffer.speech_stopped",
                        "item_id": self._utterance.item_id,
                        "audio_end_ms": self._total_ms - trim_ms,
                    }
                )
                events.extend(self._finish_utterance(trim_samples=trim_ms * TARGET_RATE // 1000))
        return events

    def _add_to_utterance(self, frame: np.ndarray, speech: bool) -> list[dict]:
        utterance = self._utterance
        assert utterance is not None
        utterance.audio.append(frame)
        utterance.samples += frame.size
        utterance.since_partial += frame.size
        utterance.quiet_frames = 0 if speech else utterance.quiet_frames + 1
        events: list[dict] = []
        at_pause = utterance.quiet_frames * FRAME_MS >= 60 and utterance.samples >= SOFT_CAP_S * TARGET_RATE
        if at_pause or utterance.samples >= MAX_UTTERANCE_S * TARGET_RATE:
            # Whisper sees one 30 s window: close the utterance at a pause once it is long,
            # or at the window's end, and start the next one seamlessly.
            events.append(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": utterance.item_id,
                    "audio_end_ms": self._total_ms,
                }
            )
            events.extend(self._finish_utterance())
            if not self.config.vad:
                self._utterance = _Utterance(item_id=self._new_id(), start_ms=self._total_ms)
            else:
                self._silence_frames = 0
                self._speech_frames = 0
            return events
        interval = self.config.partial_interval_ms * TARGET_RATE // 1000
        if utterance.since_partial >= interval:
            utterance.since_partial = 0
            self._queue(
                Work(
                    "partial",
                    utterance.item_id,
                    np.concatenate(utterance.audio),
                    self.config.language,
                    self.config.model,
                )
            )
        return events

    def _finish_utterance(self, trim_samples: int = 0) -> list[dict]:
        utterance = self._utterance
        self._utterance = None
        if utterance is None or utterance.samples == 0:
            return []
        self.utterances += 1
        self._finalized.add(utterance.item_id)
        self.work = [w for w in self.work if not (w.kind == "partial" and w.item_id == utterance.item_id)]
        audio = np.concatenate(utterance.audio)
        if 0 < trim_samples < audio.size - TARGET_RATE // 10:
            audio = audio[:-trim_samples]
        self._queue(Work("final", utterance.item_id, audio, self.config.language, self.config.model))
        self._agreements[utterance.item_id] = utterance.agreement
        return [{"type": "input_audio_buffer.committed", "item_id": utterance.item_id}]

    def _queue(self, work: Work) -> None:
        if work.kind == "partial":
            # Only the newest partial per utterance matters.
            self.work = [w for w in self.work if not (w.kind == "partial" and w.item_id == work.item_id)]
        self.work.append(work)

    # -- client commands -----------------------------------------------------

    def commit(self) -> list[dict]:
        """Close the current utterance now (manual turns, or an early end with VAD on)."""
        if self._utterance is None or self._utterance.samples == 0:
            raise ProtocolError("input_audio_buffer_commit_empty", "there is no audio to commit")
        events = []
        if self.config.vad:
            events.append(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": self._utterance.item_id,
                    "audio_end_ms": self._total_ms,
                }
            )
        events.extend(self._finish_utterance())
        self._silence_frames = 0
        self._speech_frames = 0
        return events

    def clear(self) -> list[dict]:
        self._utterance = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._prefix = []
        self._silence_frames = 0
        self._speech_frames = 0
        self.work = [w for w in self.work if w.kind == "final"]
        return [{"type": "input_audio_buffer.cleared"}]

    def take_work(self) -> list[Work]:
        work, self.work = self.work, []
        return work

    # -- transcriber results -------------------------------------------------

    def partial_text(self, item_id: str, hypothesis: str) -> list[dict]:
        if item_id in self._finalized:
            return []
        utterance = self._utterance
        if utterance is None or utterance.item_id != item_id:
            return []
        delta = utterance.agreement.partial(hypothesis)
        if not delta:
            return []
        return [{"type": DELTA_EVENT, "item_id": item_id, "delta": delta}]

    def final_text(self, item_id: str, transcript: str) -> list[dict]:
        agreement = self._agreements.pop(item_id, LocalAgreement())
        self._finalized.discard(item_id)
        text = " ".join(transcript.split())
        events: list[dict] = []
        delta = agreement.final(text)
        if delta:
            events.append({"type": DELTA_EVENT, "item_id": item_id, "delta": delta})
        events.append({"type": COMPLETED_EVENT, "item_id": item_id, "transcript": text})
        return events

    def failed(self, item_id: str, code: str, message: str) -> list[dict]:
        self._agreements.pop(item_id, None)
        self._finalized.discard(item_id)
        return [
            {
                "type": FAILED_EVENT,
                "item_id": item_id,
                "error": {"type": "server_error", "code": code, "message": message},
            }
        ]

    @property
    def buffered_ms(self) -> int:
        """Audio held for the current utterance, for the caller's caps."""
        return (self._utterance.samples * 1000 // TARGET_RATE) if self._utterance else 0


def error_event(code: str, message: str, *, kind: str = "invalid_request_error") -> dict:
    return {"type": "error", "error": {"type": kind, "code": code, "message": message}}
