"""Bounded vocabulary hints, literal pronunciation dictionaries and real word alignments."""

import pytest

from backend.utils.speech_text import (
    apply_pronunciations,
    transcription_prompt,
    validate_hints,
    validate_pronunciations,
)
from backend.utils.subtitles import Transcript, to_verbose_json, words_from_dicts


def test_pronunciations_are_literal_whole_terms_longest_first_and_nonrecursive():
    text = "SQL Server uses SQL; NoSQL, sql and C++ remain distinct."
    assert apply_pronunciations(
        text, {"SQL": "sequel", "SQL Server": "S Q L server", "sequel": "wrong", "C++": "see plus plus"}
    ) == ("S Q L server uses sequel; NoSQL, sql and see plus plus remain distinct.")


@pytest.mark.parametrize("mapping", [{"": "empty"}, {"x": " "}, {"x": "y" * 257}, {1: "one"}, {"x": 2}])
def test_pronunciation_validation(mapping):
    with pytest.raises(ValueError, match=r"pronunciation|hotword|prompt"):
        validate_pronunciations(mapping)


def test_expanded_speech_cap_is_checked_after_replacement():
    with pytest.raises(ValueError, match="expanded"):
        apply_pronunciations("x " * 1000, {"x": "y" * 100})


def test_prompt_keeps_context_and_vocabulary():
    assert (
        transcription_prompt("Database discussion.", ["PostgreSQL", "Voicebox"])
        == "Database discussion.\nPostgreSQL, Voicebox"
    )
    assert transcription_prompt(None, None) is None


@pytest.mark.parametrize(
    ("prompt", "hotwords"), [("x" * 4097, None), (None, "not a list"), (None, [""]), (None, [1]), (None, ["x"] * 101)]
)
def test_hints_reject_bad_input(prompt, hotwords):
    with pytest.raises(ValueError, match=r"pronunciation|hotword|prompt"):
        validate_hints(prompt, hotwords)


def test_word_alignments_skip_missing_times_and_clip_to_real_audio():
    words = words_from_dicts(
        [
            {"word": "hello", "start": 0.1, "end": 0.8, "probability": 0.9},
            {"word": "world", "start": 0.8, "end": 2.0},
            {"word": "unknown", "start": None, "end": 1.0},
            {"word": "invalid", "start": float("nan"), "end": 1.0},
            {"word": "padding", "start": 3.0, "end": 4.0},
        ],
        duration=1.5,
    )
    assert [(w.word, w.start, w.end) for w in words] == [("hello", 0.1, 0.8), ("world", 0.8, 1.5)]
    result = to_verbose_json(Transcript("hello world", words=words), include_words=True)
    assert result["words"][-1] == {"word": "world", "start": 0.8, "end": 1.5}
    assert to_verbose_json(Transcript("silence"), include_words=True)["words"] == []
