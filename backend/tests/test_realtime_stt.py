"""Live transcription core (torch-free): resampling, VAD, deltas and the session state machine."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from backend.services import realtime_stt as rt
from backend.services.realtime_stt import LocalAgreement, ProtocolError, SessionConfig, TranscriptionSession


def tone(seconds: float, rate: int, freq: float = 440.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def silence(seconds: float, rate: int, level: float = 0.001) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * rate)) * level).astype(np.float32)


def test_session_config_from_openai_style_update():
    base = SessionConfig()
    cfg = SessionConfig.from_update(
        base,
        {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_sample_rate": 16000,
                "input_audio_transcription": {"model": "whisper-turbo", "language": "en"},
                "turn_detection": {"type": "server_vad", "threshold": 0.7, "silence_duration_ms": 800},
                "partial_interval_ms": 500,
            },
        },
    )
    assert (cfg.input_sample_rate, cfg.model, cfg.language) == (16000, "whisper-turbo", "en")
    assert (cfg.vad, cfg.threshold, cfg.silence_duration_ms, cfg.partial_interval_ms) == (True, 0.7, 800, 500)
    assert cfg.prefix_padding_ms == 300  # untouched
    manual = SessionConfig.from_update(cfg, {"session": {"turn_detection": None}})
    assert manual.vad is False
    assert manual.to_event()["turn_detection"] is None
    assert cfg.to_event()["input_audio_transcription"] == {"model": "whisper-turbo", "language": "en"}

    for bad, code in (
        ({"session": {"input_audio_format": "g711_ulaw"}}, "unsupported_audio_format"),
        ({"session": {"input_sample_rate": 11025}}, "unsupported_sample_rate"),
        ({"session": {"turn_detection": {"type": "semantic_vad"}}}, "unsupported_turn_detection"),
        ({"session": {"turn_detection": {"threshold": 3}}}, "invalid_session"),
        ({"session": {"partial_interval_ms": 50}}, "invalid_session"),
        ({"session": "nope"}, "invalid_session"),
    ):
        with pytest.raises(ProtocolError) as exc_info:
            SessionConfig.from_update(base, bad)
        assert exc_info.value.code == code


def test_pcm_decoding_and_resampling_keep_the_signal():
    audio = tone(1.0, 24000)
    decoded = rt.decode_pcm16(pcm16(audio) + b"\x01")  # odd trailing byte ignored
    assert decoded.shape == audio.shape
    assert np.abs(decoded - audio).max() < 1e-3

    down = rt.resample(decoded, 24000, 16000)
    assert abs(len(down) - 16000) <= 1
    crossings = np.count_nonzero(np.diff(np.signbit(down)))
    assert abs(crossings / 2 - 440) < 10  # the tone survives at the same frequency
    assert np.abs(down).max() < 0.35
    up = rt.resample(down, 16000, 48000)
    assert abs(len(up) - 48000) <= 2
    assert rt.resample(np.zeros(0, dtype=np.float32), 24000, 16000).size == 0
    assert rt.decode_base64_audio(base64.b64encode(b"ab").decode()) == b"ab"
    with pytest.raises(ProtocolError):
        rt.decode_base64_audio("not base64!!")


def test_vad_separates_speech_from_silence():
    vad = rt.EnergyVAD(threshold=0.5)
    frame = vad.frame_samples
    quiet = silence(1.0, 16000)
    loud = tone(1.0, 16000)
    quiet_hits = sum(vad.is_speech(quiet[i : i + frame]) for i in range(0, len(quiet) - frame, frame))
    loud_hits = sum(vad.is_speech(loud[i : i + frame]) for i in range(0, len(loud) - frame, frame))
    assert quiet_hits == 0
    assert loud_hits > 40


def test_local_agreement_emits_only_stable_prefixes():
    la = LocalAgreement()
    assert la.partial("hello") is None  # a single hypothesis proves nothing
    assert la.partial("hello there") == "hello"
    assert la.partial("hello there my") == " there"
    assert la.partial("hello there my friend") == " my"
    assert la.partial("hello there my friend") == " friend"
    assert la.partial("hello there my friend") is None  # nothing new
    assert la.final("Hello there my friend, how are you?") is None  # "friend," is not the word "friend"
    la2 = LocalAgreement()
    la2.partial("we will")
    la2.partial("we will go")
    assert la2.final("We will go now") == " go now"
    assert la2.final("Something else entirely") is None  # the final replaces what was sent
    assert LocalAgreement().final("Just final") == "Just final"


def _session(**overrides) -> TranscriptionSession:
    ids = iter(f"item_{n}" for n in range(1, 100))
    cfg = SessionConfig(input_sample_rate=16000, **overrides)
    return TranscriptionSession(cfg, item_ids=lambda: next(ids))


def _events_of(session: TranscriptionSession, audio: np.ndarray, chunk_ms: int = 100) -> list[dict]:
    events: list[dict] = []
    step = 16000 * chunk_ms // 1000
    for i in range(0, len(audio), step):
        events.extend(session.append(pcm16(audio[i : i + step])))
    return events


def test_server_vad_finds_an_utterance_and_schedules_partials_and_a_final():
    session = _session(partial_interval_ms=500, silence_duration_ms=500)
    events = _events_of(session, np.concatenate([silence(0.5, 16000), tone(2.0, 16000)]))
    assert [e["type"] for e in events] == ["input_audio_buffer.speech_started"]
    assert events[0]["item_id"] == "item_1"
    assert 200 <= events[0]["audio_start_ms"] <= 600  # prefix padding precedes the tone
    partials = session.take_work()
    assert partials
    assert {w.kind for w in partials} == {"partial"}
    assert all(w.item_id == "item_1" for w in partials)
    assert 0.4 <= len(partials[-1].audio) / 16000 <= 2.4

    events = _events_of(session, silence(1.0, 16000))
    assert [e["type"] for e in events] == ["input_audio_buffer.speech_stopped", "input_audio_buffer.committed"]
    work = session.take_work()
    assert [w.kind for w in work] == ["final"]  # pending partials of a finished utterance are dropped
    # The utterance plus the prefix padding and 200 ms of tail; the rest of the closing silence is trimmed.
    assert 2.2 <= len(work[0].audio) / 16000 <= 2.8
    assert events[0]["audio_end_ms"] < 3500
    assert session.utterances == 1
    assert session.take_work() == []


def test_partials_turn_into_deltas_and_the_final_completes():
    session = _session(partial_interval_ms=500)
    _events_of(session, np.concatenate([tone(1.5, 16000)]))
    assert session.partial_text("item_1", "the quick") == []
    assert session.partial_text("item_1", "the quick brown") == [
        {"type": rt.DELTA_EVENT, "item_id": "item_1", "delta": "the quick"}
    ]
    events = session.commit()
    assert [e["type"] for e in events] == ["input_audio_buffer.speech_stopped", "input_audio_buffer.committed"]
    assert session.partial_text("item_1", "late hypothesis") == []  # nothing after the final was scheduled
    completed = session.final_text("item_1", "The quick brown fox.")
    assert completed == [
        {"type": rt.DELTA_EVENT, "item_id": "item_1", "delta": " brown fox."},
        {"type": rt.COMPLETED_EVENT, "item_id": "item_1", "transcript": "The quick brown fox."},
    ]
    with pytest.raises(ProtocolError):
        session.commit()  # nothing buffered now


def test_manual_turns_and_the_thirty_second_cap():
    session = _session(vad=False, partial_interval_ms=10000)
    events = _events_of(session, tone(3.0, 16000), chunk_ms=500)
    assert events == []  # no VAD events in manual mode
    assert session.buffered_ms == 3000
    assert session.commit() == [{"type": "input_audio_buffer.committed", "item_id": "item_1"}]
    assert [w.kind for w in session.take_work()] == ["final"]

    events = _events_of(session, tone(31.0, 16000), chunk_ms=1000)
    assert [e["type"] for e in events] == ["input_audio_buffer.speech_stopped", "input_audio_buffer.committed"]
    assert events[1]["item_id"] == "item_2"
    work = session.take_work()
    assert work[-1].kind == "final"
    assert abs(len(work[-1].audio) / 16000 - 30.0) < 0.05  # no pause: the hard cut at the window's end
    assert session.buffered_ms == 1000  # the remainder opened item_3
    assert session.clear() == [{"type": "input_audio_buffer.cleared"}]
    assert session.buffered_ms == 0


def test_failed_and_error_events():
    session = _session()
    assert session.failed("item_9", "busy", "Whisper is busy")[0]["type"] == rt.FAILED_EVENT
    assert rt.error_event("idle_timeout", "no audio for 60 s")["error"]["code"] == "idle_timeout"


def test_long_utterances_are_cut_at_a_pause_after_the_soft_cap():
    session = _session(vad=False, partial_interval_ms=10000)
    audio = np.concatenate([tone(26.0, 16000), silence(0.5, 16000), tone(3.0, 16000)])
    events = _events_of(session, audio, chunk_ms=500)
    assert [e["type"] for e in events] == ["input_audio_buffer.speech_stopped", "input_audio_buffer.committed"]
    cut = events[0]["audio_end_ms"]
    assert 26000 <= cut <= 26600  # the pause after 26 s, not the 30 s window
    final = session.take_work()[-1]
    assert final.kind == "final"
    assert abs(len(final.audio) / 16000 - cut / 1000) < 0.05
    assert 2900 <= session.buffered_ms <= 3500  # the rest of the audio continues in the next item

    # With server VAD the same audio also ends the first item at that pause, then keeps listening.
    session = _session(vad=True, partial_interval_ms=10000, silence_duration_ms=2000)
    events = _events_of(session, audio, chunk_ms=500)
    types = [e["type"] for e in events]
    assert types[:3] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
    ]
    assert types.count("input_audio_buffer.speech_started") == 2
