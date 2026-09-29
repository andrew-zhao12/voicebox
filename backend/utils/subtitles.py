"""Timestamped transcripts and their text renderings (SRT, WebVTT, OpenAI's ``verbose_json``).

Pure Python: the STT backends build a ``Transcript`` from what Whisper
returns (segment dictionaries on MLX, token offsets or long-form segments
with transformers) and the routes render it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TranscriptSegment:
    """One Whisper segment; the optional fields default to neutral values when a backend has no better data."""

    id: int
    start: float
    end: float
    text: str
    tokens: tuple[int, ...] = ()
    seek: int = 0
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: tuple[TranscriptSegment, ...] = field(default_factory=tuple)
    language: str | None = None
    duration: float = 0.0


def _clean(text: str) -> str:
    return " ".join(text.split())


def segments_from_dicts(items: Iterable[Mapping], *, duration: float | None = None) -> tuple[TranscriptSegment, ...]:
    """Segments from openai-whisper-shaped dictionaries (``start``, ``end``, ``text``, ...), as MLX Whisper returns.

    Whisper pads a clip to 30 s and tends to hallucinate over the padding,
    so segments without a letter or digit, segments that start after the
    clip ends, and segments Whisper itself rates as probable non-speech
    (``no_speech_prob`` above 0.6 with a mean log-probability below -1, its
    own decoding rule) are dropped, and the last end is clamped to the clip
    *duration* when it is known.
    """
    segments: list[TranscriptSegment] = []
    for item in items:
        text = _clean(str(item.get("text", "")))
        if not text or not any(ch.isalnum() for ch in text):
            continue
        if float(item.get("no_speech_prob") or 0.0) > 0.6 and float(item.get("avg_logprob") or 0.0) < -1.0:
            continue
        start = float(item.get("start") or 0.0)
        if duration is not None and start >= duration:
            continue
        end_raw = item.get("end")
        end = float(end_raw) if end_raw is not None else (duration if duration is not None else start)
        if duration is not None:
            end = min(end, duration)
        tokens = item.get("tokens") or ()
        segments.append(
            TranscriptSegment(
                id=len(segments) if "id" not in item else int(item["id"]),
                start=start,
                end=max(end, start),
                text=text,
                tokens=tuple(int(t) for t in tokens),
                seek=int(item.get("seek") or 0),
                temperature=float(item.get("temperature") or 0.0),
                avg_logprob=float(item.get("avg_logprob") or 0.0),
                compression_ratio=float(item.get("compression_ratio") or 0.0),
                no_speech_prob=float(item.get("no_speech_prob") or 0.0),
            )
        )
    return tuple(segments)


def text_of(segments: Iterable[TranscriptSegment]) -> str:
    """The transcript text as the segments spell it (what remains after filtering)."""
    return " ".join(segment.text for segment in segments)


def segments_from_offsets(
    offsets: Iterable[Mapping], *, duration: float | None = None
) -> tuple[TranscriptSegment, ...]:
    """Segments from ``WhisperTokenizer.decode(..., output_offsets=True)`` (``{"text", "timestamp": (start, end)}``)."""
    items = []
    for offset in offsets:
        start, end = offset.get("timestamp", (0.0, None))
        items.append({"text": offset.get("text", ""), "start": start, "end": end})
    return segments_from_dicts(items, duration=duration)


def format_timestamp(seconds: float, *, separator: str) -> str:
    """``HH:MM:SS,mmm`` (SRT uses a comma) or ``HH:MM:SS.mmm`` (WebVTT uses a dot)."""
    total_ms = round(max(0.0, seconds) * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def _cues(transcript: Transcript) -> Sequence[TranscriptSegment]:
    if transcript.segments:
        return transcript.segments
    if transcript.text.strip():
        return (TranscriptSegment(id=0, start=0.0, end=max(transcript.duration, 0.0), text=_clean(transcript.text)),)
    return ()


def to_srt(transcript: Transcript) -> str:
    blocks = []
    for number, segment in enumerate(_cues(transcript), start=1):
        start = format_timestamp(segment.start, separator=",")
        end = format_timestamp(segment.end, separator=",")
        blocks.append(f"{number}\n{start} --> {end}\n{segment.text}\n")
    return "\n".join(blocks)


def to_vtt(transcript: Transcript) -> str:
    lines = ["WEBVTT", ""]
    for segment in _cues(transcript):
        start = format_timestamp(segment.start, separator=".")
        end = format_timestamp(segment.end, separator=".")
        lines.extend([f"{start} --> {end}", segment.text, ""])
    return "\n".join(lines)


def to_verbose_json(transcript: Transcript, *, task: str = "transcribe") -> dict:
    """OpenAI's ``verbose_json`` shape with segment timestamps (no word timestamps)."""
    return {
        "task": task,
        "language": transcript.language or "unknown",
        "duration": round(float(transcript.duration), 3),
        "text": transcript.text,
        "segments": [
            {
                **asdict(segment),
                "tokens": list(segment.tokens),
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
            }
            for segment in transcript.segments
        ],
    }


def dumps_verbose_json(transcript: Transcript) -> str:
    return json.dumps(to_verbose_json(transcript), ensure_ascii=False)
