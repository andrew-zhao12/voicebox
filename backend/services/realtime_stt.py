"""Live transcription core: audio in, utterances out, stable partial text along the way.

Pure numpy and importable without torch, so the state machine is unit
tested in CI.  A ``TranscriptionSession`` receives PCM16 or G.711 chunks,
resamples them to Whisper's 16 kHz with one ``StreamResampler`` for the
whole stream (chunks are never resampled on their own), finds utterances
with an energy VAD (or waits for manual commits), and hands out ``Work``
items: a *partial* every ``partial_interval_ms`` while speech goes on, and a
*final* when an utterance ends or reaches ``MAX_UTTERANCE_S`` (at most
``MAX_PENDING_FINALS`` wait at once; past that an utterance fails at once
with ``whisper_busy``).  The caller
(the WebSocket route) runs Whisper on the work and feeds hypotheses back
through ``partial_text``/``final_text``, which turn them into
OpenAI-Realtime-style events with append-only deltas (LocalAgreement: only
the word prefix two consecutive hypotheses agree on is emitted, so nothing
needs retracting).

Two dialects differ only in the session events: ``voicebox`` answers with
``transcription_session.created``/``.updated`` and the flat beta session
object, ``openai`` with the GA ``session.created``/``.updated`` and a
``realtime.transcription_session`` object.  Both read ``session.update`` in
either shape, and every event carries an ``event_id``.
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

from ..utils.encode import alaw_to_linear, ulaw_to_linear
from ..utils.resample import StreamResampler

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
# Every session.created/.updated echoes these, so one huge value would turn
# each tiny update into megabytes of output.
MAX_LANGUAGE_CHARS = 32
MAX_PROMPT_CHARS = 4096
# Committed utterances of one session that may wait for their final pass; the
# next one fails at once (whisper_busy), so audio sent faster than Whisper
# transcribes it cannot pile up in memory.
MAX_PENDING_FINALS = 8
DEFAULT_RATE = 24000  # OpenAI's PCM rate
G711_RATE = 8000
# Beta ``input_audio_format`` names, and the GA format types they stand for.
INPUT_FORMATS = ("pcm16", "g711_ulaw", "g711_alaw")
GA_FORMATS = {"audio/pcm": "pcm16", "audio/pcmu": "g711_ulaw", "audio/pcma": "g711_alaw"}
DELTA_EVENT = "conversation.item.input_audio_transcription.delta"
COMPLETED_EVENT = "conversation.item.input_audio_transcription.completed"
FAILED_EVENT = "conversation.item.input_audio_transcription.failed"

Dialect = Literal["voicebox", "openai"]


class ProtocolError(ValueError):
    """A client message the session cannot accept; ``code`` is the error code sent back.

    ``event_id`` is the client event that caused it, when the client named one.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.event_id: str | None = None


@dataclass(frozen=True)
class SessionConfig:
    input_sample_rate: int = DEFAULT_RATE  # of PCM16 input; G.711 is always 8 kHz
    input_audio_format: str = "pcm16"
    model: str | None = None
    language: str | None = None
    prompt: str | None = None
    hotwords: tuple[str, ...] = ()
    word_timestamps: bool = False
    vad: bool = True
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    partial_interval_ms: int = 1000

    @property
    def source_rate(self) -> int:
        """Sample rate of the audio the client sends."""
        return self.input_sample_rate if self.input_audio_format == "pcm16" else G711_RATE

    @classmethod
    def from_update(cls, base: SessionConfig, payload: dict) -> SessionConfig:
        """Apply a ``session.update`` (or ``transcription_session.update``) payload to *base*.

        Both OpenAI shapes are read: the GA one (``session.type`` and
        ``session.audio.input.{format, transcription, turn_detection}``) and
        the flat beta one (``input_audio_format``, ``input_sample_rate``,
        ``input_audio_transcription``, ``turn_detection``).  Fields left out
        keep their value; settings Voicebox has no use for (noise reduction,
        ``include``, ``languages``, ``keywords``, ``delay``, ``idle_timeout_ms``)
        are ignored.
        """
        nested = "session" in payload
        session = payload["session"] if nested else payload
        if not isinstance(session, dict):
            raise ProtocolError("invalid_session", "session must be an object")
        # A top-level session's "type" is the event's own, so only a nested one marks the GA shape.
        if "audio" in session or (nested and "type" in session):
            values = _ga_values(session)
        else:
            values = _beta_values(base, session)
        if "partial_interval_ms" in session:  # a Voicebox extension, read in both shapes
            values["partial_interval_ms"] = session["partial_interval_ms"]
        return replace(base, **values)._checked()

    def _checked(self) -> SessionConfig:
        """This config with its numbers coerced; ``ProtocolError`` when one is not a number or out of range."""
        try:
            config = replace(
                self,
                threshold=float(self.threshold),
                prefix_padding_ms=int(self.prefix_padding_ms),
                silence_duration_ms=int(self.silence_duration_ms),
                partial_interval_ms=int(self.partial_interval_ms),
            )
        except (TypeError, ValueError, OverflowError) as e:
            raise ProtocolError("invalid_session", "turn_detection and partial_interval_ms take numbers") from e
        if not 0.0 <= config.threshold <= 1.0:
            raise ProtocolError("invalid_session", "threshold must be between 0 and 1")
        if not 0 <= config.prefix_padding_ms <= 2000:
            raise ProtocolError("invalid_session", "prefix_padding_ms must be between 0 and 2000")
        if not 100 <= config.silence_duration_ms <= 5000:
            raise ProtocolError("invalid_session", "silence_duration_ms must be between 100 and 5000")
        if not 300 <= config.partial_interval_ms <= 10000:
            raise ProtocolError("invalid_session", "partial_interval_ms must be between 300 and 10000")
        from ..utils.speech_text import validate_hints

        try:
            validate_hints(config.prompt, config.hotwords)
        except ValueError as e:
            raise ProtocolError("invalid_session", str(e)) from e
        return config

    def to_event(self) -> dict:
        """The flat session object of ``transcription_session.created``/``.updated``."""
        transcription = {"model": self.model or "whisper-1", "language": self.language}
        if self.prompt:
            transcription["prompt"] = self.prompt
        if self.hotwords:
            transcription["hotwords"] = list(self.hotwords)
        if self.word_timestamps:
            transcription["word_timestamps"] = True
        return {
            "input_audio_format": self.input_audio_format,
            "input_sample_rate": self.source_rate,
            "input_audio_transcription": transcription,
            "turn_detection": self._turn_detection(),
            "partial_interval_ms": self.partial_interval_ms,
        }

    def to_ga_session(self, session_id: str) -> dict:
        """The GA ``realtime.transcription_session`` object of ``session.created``/``.updated``."""
        if self.input_audio_format == "pcm16":
            fmt = {"type": "audio/pcm", "rate": self.input_sample_rate}
        else:
            fmt = {"type": "audio/pcmu" if self.input_audio_format == "g711_ulaw" else "audio/pcma"}
        return {
            "id": session_id,
            "object": "realtime.transcription_session",
            "type": "transcription",
            "audio": {
                "input": {
                    "format": fmt,
                    "transcription": {
                        "model": self.model or "whisper-1",
                        "language": self.language,
                        "prompt": self.prompt,
                        **({"hotwords": list(self.hotwords)} if self.hotwords else {}),
                        **({"word_timestamps": True} if self.word_timestamps else {}),
                    },
                    "turn_detection": self._turn_detection(),
                    "noise_reduction": None,
                }
            },
            "include": None,
        }

    def _turn_detection(self) -> dict | None:
        if not self.vad:
            return None
        return {
            "type": "server_vad",
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
        }


def _rate(value: object) -> int:
    for rate in SUPPORTED_RATES:
        if value == rate:
            return rate
    raise ProtocolError("unsupported_sample_rate", f"the sample rate must be one of {SUPPORTED_RATES}")


def _beta_values(base: SessionConfig, session: dict) -> dict:
    """Fields of the flat session shape (OpenAI's beta ``transcription_session.update``)."""
    values: dict = {}
    if "input_audio_format" in session:
        fmt = session["input_audio_format"]
        if fmt not in INPUT_FORMATS:
            raise ProtocolError("unsupported_audio_format", f"input_audio_format must be one of {INPUT_FORMATS}")
        values["input_audio_format"] = fmt
    if "input_sample_rate" in session:
        values["input_sample_rate"] = _rate(session["input_sample_rate"])
        is_g711 = values.get("input_audio_format", base.input_audio_format) != "pcm16"
        if is_g711 and values["input_sample_rate"] != G711_RATE:
            raise ProtocolError("unsupported_sample_rate", "G.711 audio is always 8000 Hz")
    if session.get("input_audio_transcription") is not None:
        values.update(_transcription_values(session["input_audio_transcription"], "input_audio_transcription"))
    if "turn_detection" in session:
        values.update(_turn_detection_values(session["turn_detection"]))
    return values


def _ga_values(session: dict) -> dict:
    """Fields of the GA session shape: ``{"type": "transcription", "audio": {"input": {...}}}``."""
    kind = session.get("type") or "transcription"
    if kind != "transcription":
        raise ProtocolError("unsupported_session_type", f"session type {kind!r} is not served, only 'transcription'")
    audio = session.get("audio")
    if audio is None:
        return {}
    if not isinstance(audio, dict):
        raise ProtocolError("invalid_session", "session.audio must be an object")
    source = audio.get("input")
    if source is None:
        return {}
    if not isinstance(source, dict):
        raise ProtocolError("invalid_session", "session.audio.input must be an object")
    values: dict = {}
    if source.get("format") is not None:
        values.update(_ga_format(source["format"]))
    if source.get("transcription") is not None:
        values.update(_transcription_values(source["transcription"], "session.audio.input.transcription"))
    if "turn_detection" in source:
        values.update(_turn_detection_values(source["turn_detection"]))
    return values


def _ga_format(fmt: object) -> dict:
    if not isinstance(fmt, dict):
        raise ProtocolError("invalid_session", "session.audio.input.format must be an object")
    kind = fmt.get("type") or "audio/pcm"
    name = GA_FORMATS.get(kind) if isinstance(kind, str) else None
    if name is None:
        raise ProtocolError("unsupported_audio_format", f"format.type must be one of {tuple(GA_FORMATS)}")
    if name != "pcm16":
        return {"input_audio_format": name}
    # OpenAI's PCM is 24 kHz only; Voicebox also takes the other SUPPORTED_RATES.
    rate = fmt.get("rate")
    return {"input_audio_format": name, "input_sample_rate": _rate(DEFAULT_RATE if rate is None else rate)}


def _transcription_values(transcription: object, where: str) -> dict:
    if not isinstance(transcription, dict):
        raise ProtocolError("invalid_session", f"{where} must be an object")
    values: dict = {}
    for key in ("model", "language", "prompt"):
        if key in transcription:
            value = transcription[key]
            values[key] = str(value) if value else None
    for key, cap in (("language", MAX_LANGUAGE_CHARS), ("prompt", MAX_PROMPT_CHARS)):
        if values.get(key) is not None and len(values[key]) > cap:
            raise ProtocolError("invalid_session", f"{where}.{key} is capped at {cap} characters")
    if "hotwords" in transcription:
        from ..utils.speech_text import validate_hints

        try:
            validate_hints(None, transcription["hotwords"])
        except ValueError as e:
            raise ProtocolError("invalid_session", str(e)) from e
        values["hotwords"] = tuple(transcription["hotwords"] or ())
    if "word_timestamps" in transcription:
        if not isinstance(transcription["word_timestamps"], bool):
            raise ProtocolError("invalid_session", "word_timestamps must be a boolean")
        values["word_timestamps"] = transcription["word_timestamps"]
    return values


def _turn_detection_values(turn: object) -> dict:
    if turn is None:
        return {"vad": False}
    if not isinstance(turn, dict):
        raise ProtocolError("invalid_session", "turn_detection must be an object or null")
    kind = turn.get("type") or "server_vad"
    if kind != "server_vad":
        raise ProtocolError(
            "unsupported_turn_detection",
            f"turn_detection type {kind!r} is not supported; use server_vad or null (semantic_vad is not available)",
        )
    values: dict = {"vad": True}
    for key in ("threshold", "prefix_padding_ms", "silence_duration_ms"):
        if turn.get(key) is not None:
            values[key] = turn[key]
    return values


def decode_pcm16(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit mono to float32 in [-1, 1] (a trailing odd byte is dropped)."""
    usable = len(data) - (len(data) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0


def decode_audio(data: bytes, fmt: str) -> np.ndarray:
    """Float32 samples in [-1, 1] from PCM16 or G.711 (``g711_ulaw``/``g711_alaw``) bytes."""
    if fmt == "pcm16":
        return decode_pcm16(data)
    decode = ulaw_to_linear if fmt == "g711_ulaw" else alaw_to_linear
    return decode(data).astype(np.float32) / 32768.0


def decode_base64_audio(text: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ProtocolError("invalid_audio", "audio must be base64-encoded") from e


def _rms_db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
    return 20 * math.log10(max(rms, 1e-6))


def _event(kind: str, **fields: object) -> dict:
    """A server event; every one carries its own ``event_id``."""
    return {"type": kind, "event_id": f"event_{uuid.uuid4().hex[:16]}", **fields}


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
    prompt: str | None = None
    hotwords: tuple[str, ...] = ()
    word_timestamps: bool = False


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

    def __init__(
        self,
        config: SessionConfig,
        *,
        dialect: Dialect = "voicebox",
        item_ids: Callable[[], str] | None = None,
    ) -> None:
        self.config = config
        self.dialect = dialect
        self.id = f"sess_{uuid.uuid4().hex[:16]}"
        self._new_id = item_ids or (lambda: f"item_{uuid.uuid4().hex[:12]}")
        self._resampler = StreamResampler(config.source_rate, TARGET_RATE)
        self._pending = np.zeros(0, dtype=np.float32)  # resampled audio not yet framed
        self._vad = EnergyVAD(config.threshold)
        self._utterance: _Utterance | None = None
        self._prefix: list[np.ndarray] = []  # recent non-speech frames kept as padding
        self._silence_frames = 0
        self._speech_frames = 0
        self._total_ms = 0  # audio received, in ms at TARGET_RATE
        self._last_committed: str | None = None
        self.work: list[Work] = []
        self.utterances = 0
        # Committed utterances whose final is pending (so a late partial cannot follow
        # the final): their LocalAgreement and the seconds of audio sent to Whisper.
        self._finals: dict[str, tuple[LocalAgreement, float]] = {}

    def created(self) -> list[dict]:
        return [self._session_event("created")]

    def update(self, config: SessionConfig) -> list[dict]:
        if (
            self._utterance is not None
            and self._utterance.samples
            and any(
                getattr(config, name) != getattr(self.config, name)
                for name in ("model", "language", "prompt", "hotwords", "word_timestamps")
            )
        ):
            raise ProtocolError(
                "utterance_active", "Commit or clear the current audio before changing transcription settings"
            )
        if config.source_rate != self.config.source_rate:
            # The old filter's lookahead still holds audio already received.
            self._pending = np.concatenate([self._pending, self._resampler.flush()])
            self._resampler = StreamResampler(config.source_rate, TARGET_RATE)
        self.config = config
        self._vad = EnergyVAD(config.threshold)
        return [self._session_event("updated")]

    def _session_event(self, action: str) -> dict:
        if self.dialect == "openai":
            return _event(f"session.{action}", session=self.config.to_ga_session(self.id))
        return _event(f"transcription_session.{action}", session=self.config.to_event())

    def append(self, data: bytes) -> list[dict]:
        """Feed audio in the session's input format: PCM16 or G.711 bytes."""
        samples = decode_audio(data, self.config.input_audio_format)
        if samples.size == 0:
            return []
        audio = self._resampler.feed(samples)
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
                        _event(
                            "input_audio_buffer.speech_started",
                            item_id=self._utterance.item_id,
                            audio_start_ms=start_ms,
                        )
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
                    _event(
                        "input_audio_buffer.speech_stopped",
                        item_id=self._utterance.item_id,
                        audio_end_ms=self._total_ms - trim_ms,
                    )
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
                _event("input_audio_buffer.speech_stopped", item_id=utterance.item_id, audio_end_ms=self._total_ms)
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
                    self.config.prompt,
                    self.config.hotwords,
                    self.config.word_timestamps,
                )
            )
        return events

    def _finish_utterance(self, trim_samples: int = 0) -> list[dict]:
        utterance = self._utterance
        self._utterance = None
        if utterance is None or utterance.samples == 0:
            return []
        self.utterances += 1
        self.work = [w for w in self.work if not (w.kind == "partial" and w.item_id == utterance.item_id)]
        previous, self._last_committed = self._last_committed, utterance.item_id
        committed = _event("input_audio_buffer.committed", item_id=utterance.item_id, previous_item_id=previous)
        if len(self._finals) >= MAX_PENDING_FINALS:
            message = f"{MAX_PENDING_FINALS} utterances of this session are already waiting for Whisper"
            return [committed, *self.failed(utterance.item_id, "whisper_busy", message)]
        audio = np.concatenate(utterance.audio)
        if 0 < trim_samples < audio.size - TARGET_RATE // 10:
            audio = audio[:-trim_samples]
        self._queue(
            Work(
                "final",
                utterance.item_id,
                audio,
                self.config.language,
                self.config.model,
                self.config.prompt,
                self.config.hotwords,
                self.config.word_timestamps,
            )
        )
        self._finals[utterance.item_id] = (utterance.agreement, audio.size / TARGET_RATE)
        return [committed]

    def _queue(self, work: Work) -> None:
        if work.kind == "partial":
            # Only the newest partial per utterance matters.
            self.work = [w for w in self.work if not (w.kind == "partial" and w.item_id == work.item_id)]
        self.work.append(work)

    def commit(self) -> list[dict]:
        """Close the current utterance now (manual turns, or an early end with VAD on).

        Audio still in the resampler's lookahead (a few ms) or short of a
        frame stays for the next utterance.
        """
        if self._utterance is None or self._utterance.samples == 0:
            raise ProtocolError("input_audio_buffer_commit_empty", "there is no audio to commit")
        events = []
        if self.config.vad:
            events.append(
                _event(
                    "input_audio_buffer.speech_stopped",
                    item_id=self._utterance.item_id,
                    audio_end_ms=self._total_ms,
                )
            )
        events.extend(self._finish_utterance())
        self._silence_frames = 0
        self._speech_frames = 0
        return events

    def clear(self) -> list[dict]:
        self._utterance = None
        self._resampler.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._prefix = []
        self._silence_frames = 0
        self._speech_frames = 0
        self.work = [w for w in self.work if w.kind == "final"]
        return [_event("input_audio_buffer.cleared")]

    def take_work(self) -> list[Work]:
        work, self.work = self.work, []
        return work

    def partial_text(self, item_id: str, hypothesis: str) -> list[dict]:
        if item_id in self._finals:
            return []
        utterance = self._utterance
        if utterance is None or utterance.item_id != item_id:
            return []
        delta = utterance.agreement.partial(hypothesis)
        if not delta:
            return []
        return [_event(DELTA_EVENT, item_id=item_id, content_index=0, delta=delta)]

    def final_text(self, item_id: str, transcript: str, *, words=None) -> list[dict]:
        agreement, seconds = self._finals.pop(item_id, (LocalAgreement(), 0.0))
        text = " ".join(transcript.split())
        events: list[dict] = []
        delta = agreement.final(text)
        if delta:
            events.append(_event(DELTA_EVENT, item_id=item_id, content_index=0, delta=delta))
        events.append(
            _event(
                COMPLETED_EVENT,
                item_id=item_id,
                content_index=0,
                transcript=text,
                usage={"type": "duration", "seconds": round(seconds, 3)},
                **({"words": words} if words is not None else {}),
            )
        )
        return events

    def failed(self, item_id: str, code: str, message: str) -> list[dict]:
        self._finals.pop(item_id, None)
        return [
            _event(
                FAILED_EVENT,
                item_id=item_id,
                content_index=0,
                error={"type": "server_error", "code": code, "message": message},
            )
        ]

    def abandon(self, code: str, message: str) -> list[dict]:
        """A ``failed`` event for every final still pending, when the session closes without them."""
        self.work = []
        return [event for item_id in list(self._finals) for event in self.failed(item_id, code, message)]

    @property
    def active_item_ids(self) -> set[str]:
        ids = set(self._finals)
        if self._utterance is not None:
            ids.add(self._utterance.item_id)
        return ids

    @property
    def pending_finals(self) -> int:
        """Committed utterances still waiting for their ``completed`` or ``failed`` event."""
        return len(self._finals)

    @property
    def buffered_ms(self) -> int:
        """Audio held for the current utterance, for the caller's caps."""
        return (self._utterance.samples * 1000 // TARGET_RATE) if self._utterance else 0


def error_event(
    code: str,
    message: str,
    *,
    kind: str = "invalid_request_error",
    client_event_id: str | None = None,
) -> dict:
    """An ``error`` event; *client_event_id* names the client event that caused it."""
    error = {"type": kind, "code": code, "message": message}
    if client_event_id is not None:
        error["event_id"] = client_event_id
    return _event("error", error=error)
