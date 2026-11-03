"""
MLX Whisper end to end on Apple Silicon: clean transcripts at the sample rates clients send.

mlx-audio resolves Whisper's no-speech token to end-of-text on every model
before large-v3, so decoding suppressed end-of-text and each transcript ran on
with invented text.  ``MLXSTTBackend`` installs ``_no_speech_token`` to fix
that; these tests guard it.

The speech comes from macOS ``say`` and whisper-base must already be cached
(nothing is downloaded); the model tests skip when either is missing.

Usage:
    python -m pytest backend/tests/test_mlx_whisper.py -v
"""

import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from backend.backends.mlx_backend import _no_speech_token

# whisper-base returns exactly this; without the fix it appended dots or invented words.
SENTENCE = "The quick brown fox jumps over the lazy dog."

on_apple_silicon = pytest.mark.skipif(
    not (sys.platform == "darwin" and platform.machine() == "arm64"),
    reason="MLX Whisper only runs on Apple Silicon macOS",
)


class FakeHFTokenizer:
    """The two members ``_no_speech_token`` uses on a transformers ``WhisperTokenizer``."""

    unk_token_id = 50257  # Whisper's unknown token is <|endoftext|>

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)


@pytest.mark.parametrize(
    ("vocab", "expected"),
    [
        ({"<|nocaptions|>": 50362}, 50362),  # tiny to medium, large-v2
        ({"<|nospeech|>": 50363}, 50363),  # large-v3, large-v3-turbo
        ({}, None),
    ],
)
def test_no_speech_token_never_resolves_to_end_of_text(vocab, expected):
    wrapper = SimpleNamespace(hf_tokenizer=FakeHFTokenizer(vocab))
    assert _no_speech_token(wrapper) == expected


@pytest.fixture(scope="module")
def whisper():
    pytest.importorskip("mlx_audio")
    from backend.backends.mlx_backend import MLXSTTBackend

    stt = MLXSTTBackend("base")
    if not stt._is_model_cached("base"):
        pytest.skip("openai/whisper-base is not cached, and this test does not download it")
    yield stt
    stt.unload_model()


@pytest.fixture(scope="module")
def speech(tmp_path_factory):
    """A function writing ``SENTENCE``, spoken by macOS ``say``, as a PCM_16 WAV at the given rate."""
    say = shutil.which("say")
    if say is None:
        pytest.skip("macOS say is not available")
    directory = tmp_path_factory.mktemp("mlx_whisper")

    def make(rate: int) -> str:
        path = directory / f"speech_{rate}.wav"
        command = [say, "-v", "Samantha", "-o", str(path), f"--data-format=LEI16@{rate}", SENTENCE]
        result = subprocess.run(command, capture_output=True, timeout=60)
        if result.returncode != 0 or not path.exists():
            pytest.skip(f"say could not render speech: {result.stderr.decode(errors='replace').strip()}")
        return str(path)

    return make


@on_apple_silicon
@pytest.mark.parametrize("rate", [16000, 24000, 48000])
async def test_transcribe_returns_only_the_spoken_sentence(whisper, speech, rate):
    text = await whisper.transcribe(speech(rate), "en", "base")

    assert text == SENTENCE


@on_apple_silicon
@pytest.mark.parametrize("rate", [22050, 44100])
async def test_transcribe_detailed_keeps_segments_language_and_duration(whisper, speech, rate):
    path = speech(rate)
    transcript = await whisper.transcribe_detailed(path, None, "base")

    assert transcript.text == SENTENCE
    assert transcript.language == "en"
    assert transcript.duration == pytest.approx(sf.info(path).duration, abs=0.01)
    assert transcript.segments
    assert all(0 <= s.start < s.end <= transcript.duration for s in transcript.segments)
    assert " ".join(s.text for s in transcript.segments) == transcript.text


@on_apple_silicon
async def test_silence_transcribes_to_nothing(whisper, tmp_path):
    path = str(tmp_path / "silence.wav")
    sf.write(path, np.zeros(3 * 24000, dtype=np.float32), 24000, subtype="PCM_16")

    assert await whisper.transcribe(path, "en", "base") == ""
    assert (await whisper.transcribe_detailed(path, "en", "base")).segments == ()


@on_apple_silicon
async def test_whisper_base_tokenizer_can_end_a_transcript(whisper):
    await whisper.load_model_async("base")
    tokenizer = whisper.model.get_tokenizer(language="en")

    assert tokenizer.no_speech == tokenizer.hf_tokenizer.convert_tokens_to_ids("<|nocaptions|>")
    assert tokenizer.no_speech != tokenizer.eot
