"""Live transcription core (torch-free): session updates, decoding, resampling, VAD, deltas and both dialects."""

from __future__ import annotations

import base64
import contextlib
import json
from dataclasses import replace

import numpy as np
import pytest

from backend.services import realtime_stt as rt
from backend.services.realtime_stt import LocalAgreement, ProtocolError, SessionConfig, TranscriptionSession
from backend.utils.encode import alaw_to_linear, linear_to_ulaw
from backend.utils.resample import resample

# The first frame livekit-plugins-openai 1.8.8 sends on a realtime STT socket, as captured on the wire.
PLUGIN_UPDATE = {
    "type": "session.update",
    "session": {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"rate": 24000, "type": "audio/pcm"},
                "transcription": {"language": "en", "model": "whisper-base"},
                "turn_detection": {
                    "type": "server_vad",
                    "prefix_padding_ms": 600,
                    "silence_duration_ms": 350,
                    "threshold": 0.5,
                },
            }
        },
    },
}


def tone(seconds: float, rate: int, freq: float = 440.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def silence(seconds: float, rate: int, level: float = 0.001) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * rate)) * level).astype(np.float32)


def g711(audio: np.ndarray, fmt: str) -> bytes:
    samples = np.round(np.clip(audio, -1, 1) * 32767).astype(np.int16)
    if fmt == "g711_ulaw":
        return linear_to_ulaw(samples)
    # No A-law encoder ships with Voicebox: pick the code that decodes nearest to each sample.
    table = alaw_to_linear(bytes(range(256))).astype(np.int32)
    return np.abs(samples.astype(np.int32)[:, None] - table[None, :]).argmin(axis=1).astype(np.uint8).tobytes()


def crossings_hz(audio: np.ndarray, rate: int) -> float:
    return np.count_nonzero(np.diff(np.signbit(audio))) / 2 / (len(audio) / rate)


def strip(events: list[dict]) -> list[dict]:
    """Events without their random ``event_id``."""
    for event in events:
        assert event["event_id"].startswith("event_")
    return [{k: v for k, v in event.items() if k != "event_id"} for event in events]


def ga(session: dict) -> dict:
    return {"type": "session.update", "session": {"type": "transcription", **session}}


def test_session_config_from_the_flat_beta_update():
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
    # A top-level session's "type" is the event's own, not the GA session type.
    top_level = SessionConfig.from_update(base, {"type": "session.update", "input_sample_rate": 8000})
    assert top_level.input_sample_rate == 8000

    for bad, code in (
        ({"session": {"input_audio_format": "opus"}}, "unsupported_audio_format"),
        ({"session": {"input_sample_rate": 11025}}, "unsupported_sample_rate"),
        ({"session": {"input_audio_format": "g711_ulaw", "input_sample_rate": 16000}}, "unsupported_sample_rate"),
        ({"session": {"turn_detection": {"type": "semantic_vad"}}}, "unsupported_turn_detection"),
        ({"session": {"turn_detection": {"threshold": 3}}}, "invalid_session"),
        ({"session": {"turn_detection": {"threshold": "loud"}}}, "invalid_session"),
        ({"session": {"partial_interval_ms": 50}}, "invalid_session"),
        ({"session": "nope"}, "invalid_session"),
    ):
        with pytest.raises(ProtocolError) as exc_info:
            SessionConfig.from_update(base, bad)
        assert exc_info.value.code == code


def test_the_voicebox_session_object_is_unchanged():
    assert TranscriptionSession(SessionConfig()).created()[0]["session"] == {
        "input_audio_format": "pcm16",
        "input_sample_rate": 24000,
        "input_audio_transcription": {"model": "whisper-1", "language": None},
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
        },
        "partial_interval_ms": 1000,
    }


def test_numbers_are_coerced_and_nulls_keep_the_value():
    cfg = SessionConfig.from_update(
        SessionConfig(),
        {
            "session": {
                "turn_detection": {"threshold": "0.7", "silence_duration_ms": 800.0, "prefix_padding_ms": None},
                "partial_interval_ms": "500",
            }
        },
    )
    assert (cfg.threshold, cfg.silence_duration_ms, cfg.prefix_padding_ms) == (0.7, 800, 300)
    assert cfg.partial_interval_ms == 500
    assert isinstance(cfg.threshold, float)
    assert isinstance(cfg.silence_duration_ms, int)
    assert isinstance(cfg.partial_interval_ms, int)


def test_ga_session_update_from_the_livekit_plugin():
    cfg = SessionConfig.from_update(SessionConfig(input_sample_rate=16000), PLUGIN_UPDATE)
    assert (cfg.input_audio_format, cfg.input_sample_rate, cfg.source_rate) == ("pcm16", 24000, 24000)
    assert (cfg.model, cfg.language, cfg.prompt) == ("whisper-base", "en", None)
    assert (cfg.vad, cfg.threshold, cfg.prefix_padding_ms, cfg.silence_duration_ms) == (True, 0.5, 600, 350)
    assert cfg.partial_interval_ms == 1000


def test_ga_updates_are_partial_and_skip_what_whisper_cannot_use():
    base = SessionConfig.from_update(SessionConfig(), PLUGIN_UPDATE)
    cfg = SessionConfig.from_update(
        base,
        ga(
            {
                "audio": {
                    "input": {
                        "transcription": {
                            "model": "whisper-base",
                            "prompt": "Voicebox, LiveKit",
                            "languages": ["de", "fr"],
                            "keywords": ["Voicebox"],
                            "delay": "low",
                        },
                        "noise_reduction": {"type": "near_field"},
                        "turn_detection": {
                            "type": "server_vad",
                            "silence_duration_ms": 800,
                            "idle_timeout_ms": 5000,
                            "create_response": True,
                            "interrupt_response": False,
                        },
                    }
                },
                "include": ["item.input_audio_transcription.logprobs"],
                "partial_interval_ms": 600,  # the Voicebox extension is read in the GA shape too
            }
        ),
    )
    assert cfg == replace(base, prompt="Voicebox, LiveKit", silence_duration_ms=800, partial_interval_ms=600)
    assert SessionConfig.from_update(cfg, ga({})) == cfg
    assert SessionConfig.from_update(cfg, ga({"audio": {"input": {"transcription": None}}})) == cfg
    assert SessionConfig.from_update(cfg, ga({"audio": {"input": {"turn_detection": None}}})).vad is False
    # A GA type without a value is still a transcription session.
    assert SessionConfig.from_update(cfg, {"session": {"type": None, "audio": {}}}) == cfg


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ({"type": "audio/pcm", "rate": 24000}, ("pcm16", 24000)),
        ({"type": "audio/pcm"}, ("pcm16", 24000)),  # OpenAI's PCM is 24 kHz
        ({"type": None, "rate": None}, ("pcm16", 24000)),  # both fields are optional in the SDK
        ({"rate": 16000}, ("pcm16", 16000)),  # other rates are a Voicebox extension
        ({"type": "audio/pcmu"}, ("g711_ulaw", 8000)),
        ({"type": "audio/pcma"}, ("g711_alaw", 8000)),
    ],
)
def test_ga_audio_formats(fmt, expected):
    cfg = SessionConfig.from_update(SessionConfig(input_sample_rate=48000), ga({"audio": {"input": {"format": fmt}}}))
    assert (cfg.input_audio_format, cfg.source_rate) == expected


@pytest.mark.parametrize(
    ("session", "code"),
    [
        ({"type": "realtime", "model": "gpt-realtime"}, "unsupported_session_type"),
        (
            {
                "type": "transcription",
                "audio": {"input": {"turn_detection": {"type": "semantic_vad", "eagerness": "low"}}},
            },
            "unsupported_turn_detection",
        ),
        ({"type": "transcription", "audio": {"input": {"format": {"type": "audio/opus"}}}}, "unsupported_audio_format"),
        (
            {"type": "transcription", "audio": {"input": {"format": {"type": ["audio/pcm"]}}}},
            "unsupported_audio_format",
        ),
        ({"type": "transcription", "audio": {"input": {"format": {"rate": 11025}}}}, "unsupported_sample_rate"),
        ({"type": "transcription", "audio": "pcm"}, "invalid_session"),
        ({"type": "transcription", "audio": {"input": []}}, "invalid_session"),
        ({"type": "transcription", "audio": {"input": {"format": "audio/pcm"}}}, "invalid_session"),
        ({"type": "transcription", "audio": {"input": {"transcription": "whisper-1"}}}, "invalid_session"),
        ({"type": "transcription", "audio": {"input": {"turn_detection": "server_vad"}}}, "invalid_session"),
        ({"type": "transcription", "audio": {"input": {"turn_detection": {"threshold": "loud"}}}}, "invalid_session"),
        (
            {"type": "transcription", "audio": {"input": {"turn_detection": {"silence_duration_ms": 50}}}},
            "invalid_session",
        ),
    ],
)
def test_ga_rejections(session, code):
    with pytest.raises(ProtocolError) as exc_info:
        SessionConfig.from_update(SessionConfig(), {"type": "session.update", "session": session})
    assert exc_info.value.code == code


@pytest.mark.parametrize("shape", ["ga", "beta"])
def test_language_and_prompt_have_caps(shape):
    def update(**transcription) -> dict:
        if shape == "ga":
            return ga({"audio": {"input": {"transcription": transcription}}})
        return {"type": "transcription_session.update", "session": {"input_audio_transcription": transcription}}

    longest = {"language": "l" * rt.MAX_LANGUAGE_CHARS, "prompt": "p" * rt.MAX_PROMPT_CHARS}
    cfg = SessionConfig.from_update(SessionConfig(), update(**longest))
    assert (cfg.language, cfg.prompt) == (longest["language"], longest["prompt"])
    for field, cap in (("language", rt.MAX_LANGUAGE_CHARS), ("prompt", rt.MAX_PROMPT_CHARS)):
        with pytest.raises(ProtocolError) as exc_info:
            SessionConfig.from_update(cfg, update(**{field: "x" * (cap + 1)}))
        assert exc_info.value.code == "invalid_session"
        assert f"{field} is capped at {cap} characters" in str(exc_info.value)


@pytest.mark.parametrize("dialect", ["voicebox", "openai"])
def test_every_session_update_answer_stays_small(dialect):
    """``session.updated`` echoes the stored strings: one big prompt must not make every tiny update cost a megabyte."""
    session = TranscriptionSession(SessionConfig(), dialect=dialect)
    megabyte = "x" * (1 << 20)
    for transcription in ({"prompt": megabyte}, {"language": megabyte}):
        payload = ga({"audio": {"input": {"transcription": transcription}}})
        with contextlib.suppress(ProtocolError):  # as in the route, a refused update changes nothing
            session.update(SessionConfig.from_update(session.config, payload))
    [updated] = session.update(SessionConfig.from_update(session.config, {"type": "session.update", "session": {}}))
    assert len(json.dumps(updated)) < 2000


def test_beta_g711_formats_are_8_khz_and_persist():
    cfg = SessionConfig.from_update(
        SessionConfig(input_sample_rate=16000), {"session": {"input_audio_format": "g711_ulaw"}}
    )
    assert (cfg.input_audio_format, cfg.source_rate) == ("g711_ulaw", 8000)
    assert cfg.to_event()["input_sample_rate"] == 8000
    cfg = SessionConfig.from_update(cfg, {"session": {"input_audio_transcription": {"language": "de"}}})
    assert (cfg.input_audio_format, cfg.language) == ("g711_ulaw", "de")  # an update that names no format keeps it
    cfg = SessionConfig.from_update(cfg, {"session": {"input_audio_format": "g711_alaw", "input_sample_rate": 8000}})
    assert (cfg.input_audio_format, cfg.source_rate) == ("g711_alaw", 8000)
    pcm = SessionConfig.from_update(cfg, {"session": {"input_audio_format": "pcm16", "input_sample_rate": 24000}})
    assert (pcm.input_audio_format, pcm.source_rate) == ("pcm16", 24000)


def test_session_objects_of_both_dialects():
    cfg = SessionConfig.from_update(SessionConfig(), PLUGIN_UPDATE)
    turn = {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 600, "silence_duration_ms": 350}
    assert cfg.to_event() == {
        "input_audio_format": "pcm16",
        "input_sample_rate": 24000,
        "input_audio_transcription": {"model": "whisper-base", "language": "en"},
        "turn_detection": turn,
        "partial_interval_ms": 1000,
    }
    assert cfg.to_ga_session("sess_1") == {
        "id": "sess_1",
        "object": "realtime.transcription_session",
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "whisper-base", "language": "en", "prompt": None},
                "turn_detection": turn,
                "noise_reduction": None,
            }
        },
        "include": None,
    }
    prompted = replace(cfg, prompt="Voicebox", model=None, input_audio_format="g711_alaw", vad=False)
    assert prompted.to_event()["input_audio_transcription"] == {
        "model": "whisper-1",
        "language": "en",
        "prompt": "Voicebox",
    }
    audio = prompted.to_ga_session("sess_1")["audio"]["input"]
    assert (audio["format"], audio["turn_detection"]) == ({"type": "audio/pcma"}, None)


def test_dialects_name_the_session_events():
    voicebox = TranscriptionSession(SessionConfig())
    openai = TranscriptionSession(SessionConfig(model="whisper-base"), dialect="openai")
    [created] = voicebox.created()
    assert (created["type"], created["session"]) == ("transcription_session.created", SessionConfig().to_event())
    [updated] = voicebox.update(SessionConfig.from_update(voicebox.config, PLUGIN_UPDATE))
    assert updated["type"] == "transcription_session.updated"
    assert updated["session"]["input_audio_transcription"]["model"] == "whisper-base"

    [created_ga] = openai.created()
    assert created_ga["type"] == "session.created"
    assert created_ga["session"] == openai.config.to_ga_session(openai.id)
    assert openai.id.startswith("sess_")
    [updated_ga] = openai.update(SessionConfig.from_update(openai.config, PLUGIN_UPDATE))
    assert updated_ga["type"] == "session.updated"
    assert updated_ga["session"]["id"] == openai.id  # one session, one id
    assert updated_ga["session"]["audio"]["input"]["turn_detection"]["silence_duration_ms"] == 350
    ids = [event["event_id"] for event in (created, updated, created_ga, updated_ga)]
    assert len(set(ids)) == 4
    assert all(event_id.startswith("event_") for event_id in ids)


def test_pcm_decoding_keeps_the_signal():
    audio = tone(1.0, 24000)
    decoded = rt.decode_pcm16(pcm16(audio) + b"\x01")  # odd trailing byte ignored
    assert decoded.shape == audio.shape
    assert np.abs(decoded - audio).max() < 1e-3
    assert rt.decode_audio(pcm16(audio), "pcm16").tobytes() == rt.decode_pcm16(pcm16(audio)).tobytes()
    assert rt.decode_base64_audio(base64.b64encode(b"ab").decode()) == b"ab"
    with pytest.raises(ProtocolError):
        rt.decode_base64_audio("not base64!!")


@pytest.mark.parametrize("fmt", ["g711_ulaw", "g711_alaw"])
def test_g711_audio_is_decoded_and_resampled_to_16_khz(fmt):
    audio = tone(1.0, 8000)
    data = g711(audio, fmt)
    decoded = rt.decode_audio(data, fmt)
    assert decoded.shape == audio.shape
    assert np.abs(decoded - audio).max() < 0.01  # G.711 quantisation
    assert rt.decode_audio(b"", fmt).size == 0

    session = TranscriptionSession(SessionConfig(input_audio_format=fmt, vad=False, partial_interval_ms=10000))
    for i in range(0, len(data), 160):  # 20 ms telephony frames
        session.append(data[i : i + 160])
    session.commit()
    [work] = session.take_work()
    assert 16000 - 2 * session._vad.frame_samples <= len(work.audio) <= 16000
    assert abs(crossings_hz(work.audio, 16000) - 440) < 5
    assert 0.27 < np.abs(work.audio[1000:-1000]).max() < 0.33


@pytest.mark.parametrize(("rate", "chunk_ms"), [(24000, 50), (24000, 20), (48000, 10), (22050, 10), (8000, 20)])
def test_streamed_audio_is_resampled_as_one_stream(rate, chunk_ms):
    audio = tone(2.0, rate)
    session = TranscriptionSession(SessionConfig(input_sample_rate=rate, vad=False, partial_interval_ms=10000))
    step = rate * chunk_ms // 1000
    for i in range(0, len(audio), step):
        session.append(pcm16(audio[i : i + step]))
    session.commit()
    [work] = session.take_work()
    whole = resample(rt.decode_pcm16(pcm16(audio)), rate, rt.TARGET_RATE)
    # Everything but the filter's lookahead and a partial frame, sample for sample as if sent in one piece.
    assert len(whole) - 2 * session._vad.frame_samples <= len(work.audio) <= len(whole)
    assert np.abs(work.audio - whole[: len(work.audio)]).max() < 1e-6


def test_a_rate_change_keeps_the_audio_already_received():
    session = TranscriptionSession(SessionConfig(input_sample_rate=24000, vad=False, partial_interval_ms=10000))
    session.append(pcm16(tone(0.5, 24000)))
    session.update(SessionConfig.from_update(session.config, {"session": {"input_sample_rate": 48000}}))
    session.append(pcm16(tone(0.5, 48000)))
    received = sum(frame.size for frame in session._utterance.audio) + session._pending.size
    # Only the new filter's lookahead (about 25 samples) is still held back.
    assert 16000 - 30 <= received <= 16000


def test_clear_starts_the_stream_over():
    fresh = TranscriptionSession(SessionConfig(vad=False, partial_interval_ms=10000))
    cleared = TranscriptionSession(SessionConfig(vad=False, partial_interval_ms=10000))
    cleared.append(pcm16(tone(0.33, 24000, freq=1000)))
    assert [e["type"] for e in cleared.clear()] == ["input_audio_buffer.cleared"]
    audio = pcm16(tone(1.0, 24000))
    for session in (fresh, cleared):
        session.append(audio)
        session.commit()
    [a], [b] = fresh.take_work(), cleared.take_work()
    assert np.array_equal(a.audio, b.audio)


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
    assert strip(session.partial_text("item_1", "the quick brown")) == [
        {"type": rt.DELTA_EVENT, "item_id": "item_1", "content_index": 0, "delta": "the quick"}
    ]
    events = session.commit()
    assert [e["type"] for e in events] == ["input_audio_buffer.speech_stopped", "input_audio_buffer.committed"]
    assert session.partial_text("item_1", "late hypothesis") == []  # nothing after the final was scheduled
    completed = session.final_text("item_1", "The quick brown fox.")
    assert strip(completed) == [
        {"type": rt.DELTA_EVENT, "item_id": "item_1", "content_index": 0, "delta": " brown fox."},
        {
            "type": rt.COMPLETED_EVENT,
            "item_id": "item_1",
            "content_index": 0,
            "transcript": "The quick brown fox.",
            "usage": {"type": "duration", "seconds": 1.5},
        },
    ]
    with pytest.raises(ProtocolError):
        session.commit()  # nothing buffered now


def test_committed_items_chain_and_pending_finals_are_counted():
    session = _session(vad=False, partial_interval_ms=10000)
    _events_of(session, tone(1.0, 16000))
    [first] = strip(session.commit())
    assert first == {"type": "input_audio_buffer.committed", "item_id": "item_1", "previous_item_id": None}
    _events_of(session, tone(0.5, 16000))
    [second] = strip(session.commit())
    assert second == {"type": "input_audio_buffer.committed", "item_id": "item_2", "previous_item_id": "item_1"}
    assert session.pending_finals == 2
    assert session.final_text("item_1", "hello")[-1]["usage"] == {"type": "duration", "seconds": 1.0}
    [failed] = strip(session.failed("item_2", "whisper_busy", "Whisper is busy"))
    assert failed == {
        "type": rt.FAILED_EVENT,
        "item_id": "item_2",
        "content_index": 0,
        "error": {"type": "server_error", "code": "whisper_busy", "message": "Whisper is busy"},
    }
    assert session.pending_finals == 0


def test_abandon_fails_every_pending_final():
    session = _session(vad=False, partial_interval_ms=10000)
    for _ in range(2):
        _events_of(session, tone(1.0, 16000))
        session.commit()
    events = session.abandon("session_closed", "the session ended")
    assert [(e["type"], e["item_id"], e["error"]["code"]) for e in events] == [
        (rt.FAILED_EVENT, "item_1", "session_closed"),
        (rt.FAILED_EVENT, "item_2", "session_closed"),
    ]
    assert session.pending_finals == 0
    assert session.take_work() == []
    assert session.abandon("session_closed", "again") == []


def test_a_session_keeps_at_most_max_pending_finals():
    """Audio sent faster than Whisper transcribes it fails at once instead of piling up in memory."""
    session = _session(vad=False, partial_interval_ms=10000)
    for _ in range(rt.MAX_PENDING_FINALS):
        _events_of(session, tone(1.0, 16000))
        assert [e["type"] for e in session.commit()] == ["input_audio_buffer.committed"]
    assert [w.kind for w in session.take_work()] == ["final"] * rt.MAX_PENDING_FINALS

    _events_of(session, tone(1.0, 16000))
    committed, failed = strip(session.commit())
    newest = f"item_{rt.MAX_PENDING_FINALS + 1}"
    assert committed == {
        "type": "input_audio_buffer.committed",
        "item_id": newest,
        "previous_item_id": f"item_{rt.MAX_PENDING_FINALS}",
    }
    assert (failed["type"], failed["item_id"], failed["error"]["code"]) == (rt.FAILED_EVENT, newest, "whisper_busy")
    assert session.take_work() == []  # its audio is dropped, not queued
    assert session.pending_finals == rt.MAX_PENDING_FINALS

    session.final_text("item_1", "one")  # a final completes, so the next utterance is queued again
    _events_of(session, tone(1.0, 16000))
    assert [e["type"] for e in session.commit()] == ["input_audio_buffer.committed"]
    assert [w.kind for w in session.take_work()] == ["final"]
    assert session.pending_finals == rt.MAX_PENDING_FINALS


def test_manual_turns_and_the_thirty_second_cap():
    session = _session(vad=False, partial_interval_ms=10000)
    events = _events_of(session, tone(3.0, 16000), chunk_ms=500)
    assert events == []  # no VAD events in manual mode
    assert session.buffered_ms == 3000
    assert strip(session.commit()) == [
        {"type": "input_audio_buffer.committed", "item_id": "item_1", "previous_item_id": None}
    ]
    assert [w.kind for w in session.take_work()] == ["final"]

    events = _events_of(session, tone(31.0, 16000), chunk_ms=1000)
    assert [e["type"] for e in events] == ["input_audio_buffer.speech_stopped", "input_audio_buffer.committed"]
    assert events[1]["item_id"] == "item_2"
    work = session.take_work()
    assert work[-1].kind == "final"
    assert abs(len(work[-1].audio) / 16000 - 30.0) < 0.05  # no pause: the hard cut at the window's end
    assert session.buffered_ms == 1000  # the remainder opened item_3
    assert strip(session.clear()) == [{"type": "input_audio_buffer.cleared"}]
    assert session.buffered_ms == 0


@pytest.mark.parametrize("sample_rate", [16000, 24000, 48000])
def test_manual_silent_pcm_does_not_schedule_background_partials(sample_rate):
    session = TranscriptionSession(SessionConfig(input_sample_rate=sample_rate, vad=False))
    packet = bytes(sample_rate * 20 // 1000 * 2)
    for _ in range(500):
        assert session.append(packet) == []
        assert session.take_work() == []
    assert 9900 <= session.buffered_ms <= 10000


def test_manual_partials_accept_quiet_pcm_and_wait_for_new_signal():
    session = _session(vad=False, threshold=1.0)
    # One PCM16 quantization step is far below the server VAD speech threshold.
    quiet = tone(1.0, 16000, amplitude=2 / 32768)
    assert _events_of(session, quiet) == []
    [first] = session.take_work()
    assert first.kind == "partial"
    assert np.max(np.abs(first.audio)) == 1 / 32768

    assert _events_of(session, np.zeros(2 * 16000, dtype=np.float32)) == []
    assert session.take_work() == []
    assert _events_of(session, quiet) == []
    [second] = session.take_work()
    assert second.kind == "partial"
    assert second.item_id == first.item_id
    assert second.audio.size == 4 * 16000
    assert np.count_nonzero(second.audio) == 2 * np.count_nonzero(first.audio)

    session.commit()
    [final] = session.take_work()
    assert final.kind == "final"
    np.testing.assert_array_equal(final.audio, second.audio)


def test_manual_commit_still_transcribes_all_buffered_silent_pcm():
    session = _session(vad=False)
    assert session.append(bytes(16000 * 2)) == []
    assert session.take_work() == []
    assert strip(session.commit()) == [
        {"type": "input_audio_buffer.committed", "item_id": "item_1", "previous_item_id": None}
    ]
    [final] = session.take_work()
    assert final.kind == "final"
    assert final.audio.size == 16000
    assert not np.any(final.audio)
    assert session.pending_finals == 1
    assert session.final_text(final.item_id, "")[-1]["usage"] == {"type": "duration", "seconds": 1.0}
    assert session.pending_finals == 0


def test_automatic_caps_complete_silent_pcm_without_inference_and_keep_item_chain():
    session = _session(vad=False)
    events = []
    for _ in range(49):
        events.extend(session.append(bytes(16000 * 2)))
        assert session.take_work() == []
    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        rt.COMPLETED_EVENT,
    ] * 2
    for index in range(2):
        stopped, committed, completed = events[index * 3 : index * 3 + 3]
        item_id = f"item_{index + 1}"
        assert stopped["item_id"] == committed["item_id"] == completed["item_id"] == item_id
        assert committed["previous_item_id"] == ("item_1" if index else None)
        assert completed["transcript"] == ""
        assert completed["usage"] == {"type": "duration", "seconds": rt.SOFT_CAP_S}
    assert session.pending_finals == 0
    assert session.buffered_ms == 1000

    _events_of(session, tone(0.3, 16000))
    [committed] = session.commit()
    assert committed["item_id"] == "item_3"
    assert committed["previous_item_id"] == "item_2"
    [final] = session.take_work()
    assert final.kind == "final"
    assert final.audio.size == 20800
    assert np.any(final.audio)


def test_clear_resets_manual_signal_detection():
    session = _session(vad=False)
    _events_of(session, tone(0.3, 16000))
    session.clear()
    assert session.append(bytes(16000 * 2)) == []
    assert session.take_work() == []


def test_failed_and_error_events():
    session = _session()
    assert session.failed("item_9", "busy", "Whisper is busy")[0]["type"] == rt.FAILED_EVENT
    assert rt.error_event("idle_timeout", "no audio for 60 s")["error"]["code"] == "idle_timeout"
    [event] = strip([rt.error_event("invalid_session", "bad", client_event_id="evt_1")])
    assert event == {
        "type": "error",
        "error": {"type": "invalid_request_error", "code": "invalid_session", "message": "bad", "event_id": "evt_1"},
    }


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


def test_hints_are_snapshotted_for_work_and_require_an_utterance_boundary():
    cfg = SessionConfig.from_update(
        SessionConfig(),
        {
            "session": {
                "input_sample_rate": 16000,
                "turn_detection": None,
                "input_audio_transcription": {
                    "prompt": "Database discussion",
                    "hotwords": ["PostgreSQL"],
                    "word_timestamps": True,
                },
            }
        },
    )
    session = TranscriptionSession(cfg)
    audio = (np.ones(16000, dtype=np.float32) * 1000).astype("<i2").tobytes()
    session.append(audio)
    changed = SessionConfig.from_update(cfg, {"session": {"input_audio_transcription": {"prompt": "New context"}}})
    with pytest.raises(rt.ProtocolError, match="Commit or clear"):
        session.update(changed)
    session.commit()
    work = session.take_work()[-1]
    session.update(changed)
    assert work.prompt == "Database discussion"
    assert work.hotwords == ("PostgreSQL",)
    assert work.word_timestamps
    assert cfg.to_event()["input_audio_transcription"]["hotwords"] == ["PostgreSQL"]
