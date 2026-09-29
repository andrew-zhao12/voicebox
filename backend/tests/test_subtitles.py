"""Transcript segments and their SRT, WebVTT and verbose_json renderings (torch-free)."""

from backend.utils import subtitles
from backend.utils.subtitles import Transcript, TranscriptSegment


def test_segments_from_whisper_dicts_and_tokenizer_offsets():
    dicts = [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 1.2,
            "text": "  Hello   world. ",
            "tokens": [1, 2],
            "avg_logprob": -0.1,
        },
        {"start": 1.2, "end": None, "text": "Tail without end"},
        {"start": 3.0, "end": 3.5, "text": "   "},  # empty segments are dropped
    ]
    segments = subtitles.segments_from_dicts(dicts, duration=2.5)
    assert [s.text for s in segments] == ["Hello world.", "Tail without end"]
    assert segments[0].tokens == (1, 2)
    assert segments[0].avg_logprob == -0.1
    assert segments[1].end == 2.5  # a missing end takes the clip duration
    assert segments[1].id == 1

    offsets = [{"text": " Hello world.", "timestamp": (0.0, 1.2)}, {"text": " Second part.", "timestamp": (1.2, 2.8)}]
    from_offsets = subtitles.segments_from_offsets(offsets)
    assert [(s.start, s.end, s.text) for s in from_offsets] == [(0.0, 1.2, "Hello world."), (1.2, 2.8, "Second part.")]


def test_timestamp_formatting():
    assert subtitles.format_timestamp(0, separator=",") == "00:00:00,000"
    assert subtitles.format_timestamp(61.2346, separator=".") == "00:01:01.235"
    assert subtitles.format_timestamp(3725.5, separator=",") == "01:02:05,500"
    assert subtitles.format_timestamp(-1, separator=",") == "00:00:00,000"


def test_srt_vtt_and_verbose_json():
    transcript = Transcript(
        text="Hello world. Second part.",
        segments=(
            TranscriptSegment(id=0, start=0.0, end=1.2, text="Hello world.", tokens=(5, 6)),
            TranscriptSegment(id=1, start=1.2, end=2.8, text="Second part."),
        ),
        language="en",
        duration=2.8,
    )
    assert subtitles.to_srt(transcript) == (
        "1\n00:00:00,000 --> 00:00:01,200\nHello world.\n\n2\n00:00:01,200 --> 00:00:02,800\nSecond part.\n"
    )
    assert subtitles.to_vtt(transcript) == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.200\nHello world.\n\n00:00:01.200 --> 00:00:02.800\nSecond part.\n"
    )
    verbose = subtitles.to_verbose_json(transcript)
    assert verbose["task"] == "transcribe"
    assert verbose["language"] == "en"
    assert verbose["duration"] == 2.8
    assert verbose["text"] == "Hello world. Second part."
    assert verbose["segments"][0] == {
        "id": 0,
        "seek": 0,
        "start": 0.0,
        "end": 1.2,
        "text": "Hello world.",
        "tokens": [5, 6],
        "temperature": 0.0,
        "avg_logprob": 0.0,
        "compression_ratio": 0.0,
        "no_speech_prob": 0.0,
    }
    assert subtitles.dumps_verbose_json(transcript).startswith('{"task": "transcribe"')


def test_a_transcript_without_segments_becomes_one_cue():
    transcript = Transcript(text="Just text.", duration=4.0)
    assert subtitles.to_srt(transcript) == "1\n00:00:00,000 --> 00:00:04,000\nJust text.\n"
    assert subtitles.to_vtt(Transcript(text="")) == "WEBVTT\n"
    assert subtitles.to_verbose_json(transcript)["segments"] == []


def test_padding_hallucinations_are_dropped_and_ends_clamped():
    dicts = [
        {"start": 0.0, "end": 2.9, "text": " The quick brown fox."},
        {"start": 2.9, "end": 5.4, "text": " Then it runs home."},  # end past the 5.1 s clip
        {"start": 4.92, "end": 5.92, "text": "."},
        {"start": 5.92, "end": 6.92, "text": " ..."},
        {"start": 9.92, "end": 10.92, "text": " Real words after the clip ended"},
    ]
    segments = subtitles.segments_from_dicts(dicts, duration=5.1)
    assert [s.text for s in segments] == ["The quick brown fox.", "Then it runs home."]
    assert segments[-1].end == 5.1
    assert subtitles.text_of(segments) == "The quick brown fox. Then it runs home."
