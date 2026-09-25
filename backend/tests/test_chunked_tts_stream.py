"""Tests for sentence-level streaming in ``backend.utils.chunked_tts``."""

import asyncio

import numpy as np
import pytest

from backend.utils.chunked_tts import (
    FIRST_CHUNK_MIN_CHARS,
    CrossfadeJoiner,
    concatenate_audio_chunks,
    generate_chunked,
    generate_chunked_stream,
    iter_generated_chunks,
    split_text_into_chunks,
)

SAMPLE_RATE = 1000
LONG_TAIL = (
    " This is a much longer second sentence that keeps going for a while so the text is long."
    " And a third one that pushes the whole text past the first-chunk cap."
)
THIRD_SENTENCE = "And a third one that pushes the whole text past the first-chunk cap."


class FakeBackend:
    """Deterministic backend: one sample per character, value derived from the seed."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, int | None]] = []

    async def generate(self, text, voice_prompt, language="en", seed=None, instruct=None):
        self.calls.append((text, seed))
        value = 0.1 if seed is None else (seed % 7 + 1) / 10
        return np.full(len(text), value, dtype=np.float32), self.sample_rate


async def collect(stream):
    pieces = []
    sample_rate = None
    async for audio, sr in stream:
        pieces.append(audio)
        sample_rate = sr
    return pieces, sample_rate


def test_first_chunk_cuts_at_the_first_sentence():
    text = "Sure, I can help with that." + LONG_TAIL

    chunks = split_text_into_chunks(text, 800, first_chunk_chars=120)

    assert chunks == ["Sure, I can help with that.", LONG_TAIL.strip()]


def test_first_chunk_respects_the_minimum_length():
    text = "Hi." + LONG_TAIL

    chunks = split_text_into_chunks(text, 800, first_chunk_chars=120)

    assert chunks[0].startswith("Hi. This is a much longer")
    assert chunks[0].endswith("long.")
    assert len(chunks[0]) >= FIRST_CHUNK_MIN_CHARS
    assert chunks == [chunks[0], THIRD_SENTENCE]


def test_first_chunk_falls_back_to_a_clause_boundary():
    text = " ".join(["word"] * 10) + ", " + " ".join(["word"] * 50)

    chunks = split_text_into_chunks(text, 800, first_chunk_chars=120)

    assert chunks[0] == " ".join(["word"] * 10) + ","
    assert chunks == [chunks[0], " ".join(["word"] * 50)]


def test_first_chunk_without_any_boundary_uses_normal_splitting():
    text = " ".join(["word"] * 60)

    assert split_text_into_chunks(text, 800, first_chunk_chars=120) == [text]


def test_first_chunk_needs_whitespace_after_punctuation_at_the_cap():
    text = "a" * 28 + ".b" + " more text follows here and continues on."

    assert split_text_into_chunks(text, 800, first_chunk_chars=30) == [text]


def test_first_chunk_keeps_tags_intact():
    text = "[laugh] Well hello there my friend." + LONG_TAIL

    chunks = split_text_into_chunks(text, 800, first_chunk_chars=120)

    assert chunks[0] == "[laugh] Well hello there my friend."


def test_first_chunk_none_keeps_legacy_splitting():
    text = "Sure, I can help with that." + LONG_TAIL

    assert split_text_into_chunks(text, 800) == [text]
    assert split_text_into_chunks(text, 800, first_chunk_chars=None) == [text]
    assert split_text_into_chunks(text, 40) == split_text_into_chunks(text, 40, first_chunk_chars=None)


async def test_iter_generated_chunks_yields_in_order_with_offset_seeds():
    backend = FakeBackend()
    text = "First sentence here. Second sentence here. Third sentence here."

    pieces, sample_rate = await collect(
        iter_generated_chunks(backend, text, {}, seed=5, max_chunk_chars=30),
    )

    assert [call[0] for call in backend.calls] == [
        "First sentence here.",
        "Second sentence here.",
        "Third sentence here.",
    ]
    assert [call[1] for call in backend.calls] == [5, 6, 7]
    assert sample_rate == SAMPLE_RATE
    assert [len(p) for p in pieces] == [20, 21, 20]
    assert pieces[0][0] == pytest.approx((5 % 7 + 1) / 10)


async def test_iter_generated_chunks_single_chunk_uses_original_text_and_seed():
    backend = FakeBackend()

    pieces, _ = await collect(iter_generated_chunks(backend, "  short text  ", {}, seed=3))

    assert backend.calls == [("  short text  ", 3)]
    assert len(pieces) == 1


async def test_generate_chunked_equals_concatenated_iteration():
    text = "First sentence here. Second sentence here. Third sentence here."

    audio, sr = await generate_chunked(FakeBackend(), text, {}, seed=1, max_chunk_chars=30, crossfade_ms=10)
    pieces, _ = await collect(iter_generated_chunks(FakeBackend(), text, {}, seed=1, max_chunk_chars=30))

    assert sr == SAMPLE_RATE
    assert np.array_equal(audio, concatenate_audio_chunks(pieces, SAMPLE_RATE, crossfade_ms=10))


@pytest.mark.parametrize("crossfade_ms", [0, 50, 500])
@pytest.mark.parametrize("sample_rate", [1000, 24000])
def test_joiner_is_bit_identical_to_concatenate_audio_chunks(crossfade_ms, sample_rate):
    rng = np.random.default_rng(crossfade_ms + sample_rate)
    lengths = [0, 3, 40, 50, 51, 1000, 2500, 1199, 1200, 1201, 7]
    for trial in range(20):
        count = int(rng.integers(1, 7))
        chunks = [(rng.standard_normal(int(rng.choice(lengths))) * 0.5).astype(np.float32) for _ in range(count)]

        joiner = CrossfadeJoiner(crossfade_ms)
        streamed = [joiner.push(chunk, sample_rate) for chunk in chunks]
        streamed.append(joiner.flush())
        expected = concatenate_audio_chunks(chunks, sample_rate, crossfade_ms=crossfade_ms)

        assert np.array_equal(np.concatenate(streamed), expected), f"trial {trial}: {[len(c) for c in chunks]}"


def test_joiner_appends_continuation_pieces_without_blending():
    joiner = CrossfadeJoiner(50)
    a = np.full(100, 0.5, dtype=np.float32)
    b = np.full(100, 0.25, dtype=np.float32)
    c = np.full(100, 1.0, dtype=np.float32)

    out = [joiner.push(a, SAMPLE_RATE), joiner.push(b, SAMPLE_RATE, new_chunk=False), joiner.push(c, SAMPLE_RATE)]
    out.append(joiner.flush())
    joined = np.concatenate(out)

    expected = concatenate_audio_chunks([np.concatenate([a, b]), c], SAMPLE_RATE, crossfade_ms=50)
    assert np.array_equal(joined, expected)
    assert len(joined) == 250


async def test_stream_matches_generate_chunked_for_the_same_chunking():
    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."

    audio, sr = await generate_chunked(FakeBackend(), text, {}, seed=4, max_chunk_chars=45, crossfade_ms=20)
    pieces, stream_sr = await collect(
        generate_chunked_stream(
            FakeBackend(),
            text,
            {},
            seed=4,
            max_chunk_chars=45,
            crossfade_ms=20,
            first_chunk_chars=None,
        )
    )

    assert stream_sr == sr
    assert np.array_equal(np.concatenate(pieces), audio)


async def test_stream_yields_first_chunk_before_the_second_is_generated():
    gate = asyncio.Event()

    class GatedBackend(FakeBackend):
        async def generate(self, text, voice_prompt, language="en", seed=None, instruct=None):
            if len(self.calls) == 1:
                await gate.wait()
            return await super().generate(text, voice_prompt, language, seed, instruct)

    backend = GatedBackend()
    stream = generate_chunked_stream(
        backend,
        "First sentence here. Second sentence here.",
        {},
        max_chunk_chars=25,
        crossfade_ms=0,
        first_chunk_chars=None,
    )

    first_audio, _ = await asyncio.wait_for(stream.__anext__(), timeout=1)
    assert len(first_audio) == len("First sentence here.")
    assert not gate.is_set()

    gate.set()
    rest, _ = await collect(stream)
    assert [len(p) for p in rest] == [len("Second sentence here.")]


async def test_stream_keeps_the_runaway_retry_path():
    backend = FakeBackend()
    text = "x" * 241

    pieces, _ = await collect(
        generate_chunked_stream(
            backend,
            text,
            {},
            crossfade_ms=0,
            first_chunk_chars=None,
            runaway_detector=lambda audio, sr: len(audio) > 200,
        )
    )
    expected, _ = await generate_chunked(
        FakeBackend(),
        text,
        {},
        crossfade_ms=0,
        runaway_detector=lambda audio, sr: len(audio) > 200,
    )

    assert [len(call[0]) for call in backend.calls] == [241, 120, 120, 1]
    assert np.array_equal(np.concatenate(pieces), expected)


async def test_cancellation_waits_for_the_inflight_engine_call():
    class SlowBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = False

        async def generate(self, text, voice_prompt, language="en", seed=None, instruct=None):
            self.started.set()
            await self.release.wait()
            self.finished = True
            return await super().generate(text, voice_prompt, language, seed, instruct)

    backend = SlowBackend()

    async def consume():
        async for _ in generate_chunked_stream(backend, "hello world", {}, first_chunk_chars=None):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(backend.started.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()  # still draining the engine call

    backend.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.finished


class FakeStreamBackend(FakeBackend):
    """Backend that also streams each chunk in three pieces."""

    def __init__(self):
        super().__init__()
        self.stream_calls: list[str] = []

    async def generate_stream(self, text, voice_prompt, language="en", seed=None, instruct=None):
        self.stream_calls.append(text)
        audio, sr = await super().generate(text, voice_prompt, language, seed, instruct)
        for piece in np.array_split(audio, 3):
            yield piece, sr


async def test_backend_stream_pieces_are_forwarded_and_crossfaded_per_chunk():
    backend = FakeStreamBackend()
    text = "First sentence here. Second sentence here."

    pieces, _ = await collect(
        generate_chunked_stream(backend, text, {}, max_chunk_chars=25, crossfade_ms=10, first_chunk_chars=None)
    )
    expected, _ = await generate_chunked(FakeBackend(), text, {}, max_chunk_chars=25, crossfade_ms=10)

    assert backend.stream_calls == ["First sentence here.", "Second sentence here."]
    assert np.array_equal(np.concatenate(pieces), expected)


async def test_backend_stream_is_skipped_when_trimming_is_required():
    backend = FakeStreamBackend()

    await collect(
        generate_chunked_stream(
            backend,
            "First sentence here. Second sentence here.",
            {},
            max_chunk_chars=25,
            first_chunk_chars=None,
            trim_fn=lambda audio, sr: audio,
        )
    )

    assert backend.stream_calls == []
    assert len(backend.calls) == 2


async def test_backend_stream_runaway_cuts_and_continues():
    backend = FakeStreamBackend()
    text = "First sentence here. Second sentence here."

    def cut_after_ten(audio, sr):
        return 10 if len(audio) > 10 else None

    pieces, _ = await collect(
        generate_chunked_stream(
            backend,
            text,
            {},
            max_chunk_chars=25,
            crossfade_ms=0,
            first_chunk_chars=None,
            runaway_detector=lambda audio, sr: True,
            runaway_cut_fn=cut_after_ten,
        )
    )

    assert backend.stream_calls == ["First sentence here.", "Second sentence here."]
    assert [len(p) for p in pieces] == [7, 3, 7, 3]
