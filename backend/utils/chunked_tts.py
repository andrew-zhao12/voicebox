"""
Chunked TTS generation utilities.

Splits long text into sentence-boundary chunks, generates audio per-chunk
via any TTSBackend, and concatenates with crossfade.  All logic is
engine-agnostic — it wraps the standard ``TTSBackend.generate()`` interface.

Short text (≤ max_chunk_chars) uses the single-shot fast path with zero
overhead.

Two entry points share one per-chunk pipeline (runaway retry, trimming):

* ``generate_chunked`` returns a single crossfaded array once every chunk
  has been synthesized.
* ``generate_chunked_stream`` yields playable audio as each chunk finishes.
  Concatenating everything it yields gives, sample for sample, what
  ``generate_chunked`` returns for the same chunks.
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Callable, Iterator

import numpy as np

logger = logging.getLogger("voicebox.chunked-tts")

# Default chunk size in characters.  Can be overridden per-request via
# the ``max_chunk_chars`` field on GenerationRequest.
DEFAULT_MAX_CHUNK_CHARS = 800
# Streaming only: cap for the first chunk so the first audio arrives after
# one sentence instead of after ~800 characters of synthesis.
DEFAULT_FIRST_CHUNK_CHARS = 120
# Never split off a first chunk shorter than this; very short inputs give
# poor prosody on most engines.
FIRST_CHUNK_MIN_CHARS = 20
MAX_RUNAWAY_RETRIES = 2
MIN_RUNAWAY_RETRY_CHARS = 100

# Common abbreviations that should NOT be treated as sentence endings.
# Lowercase for case-insensitive matching.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "ave",
        "blvd",
        "inc",
        "ltd",
        "corp",
        "dept",
        "est",
        "approx",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "u.s",
        "u.s.a",
        "u.k",
    }
)

# Paralinguistic tags used by Chatterbox Turbo.  The splitter must never
# cut inside one of these.
_PARA_TAG_RE = re.compile(r"\[[^\]]*\]")
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")
_CJK_SENTENCE_END_RE = re.compile(
    r"[\N{IDEOGRAPHIC FULL STOP}\N{FULLWIDTH EXCLAMATION MARK}\N{FULLWIDTH QUESTION MARK}]"
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[;:,—](?:\s|$)")

ChunkStream = AsyncIterator[tuple[np.ndarray, int]]


def split_text_into_chunks(
    text: str,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    first_chunk_chars: int | None = None,
) -> list[str]:
    """Split *text* at natural boundaries into chunks of at most *max_chars*.

    Priority: sentence-end (``.!?`` not preceded by an abbreviation and not
    inside brackets) → clause boundary (``;:,—``) → whitespace → hard cut.

    Paralinguistic tags like ``[laugh]`` are treated as atomic and will not
    be split across chunks.

    ``first_chunk_chars`` (used by streaming) caps the first chunk so it ends
    at the first sentence boundary inside that window; the remainder is split
    with *max_chars* as usual.  ``None`` keeps the historical behaviour.
    """
    text = text.strip()
    if not text:
        return []

    head: list[str] = []
    if first_chunk_chars is not None and len(text) > first_chunk_chars:
        first, rest = _split_first_chunk(text, first_chunk_chars)
        if first is not None:
            head = [first]
            text = rest

    if len(text) <= max_chars:
        return head + ([text] if text else [])

    chunks: list[str] = []
    remaining = text

    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        chunk, remaining = _split_once(remaining, max_chars)
        if chunk:
            chunks.append(chunk)

    return head + chunks


def _split_once(remaining: str, max_chars: int) -> tuple[str, str]:
    """Cut one chunk of at most *max_chars* off the front of *remaining*."""
    segment = remaining[:max_chars]

    # Try to split at the last real sentence ending
    split_pos = _find_last_sentence_end(segment)
    if split_pos == -1:
        split_pos = _find_last_clause_boundary(segment)
    if split_pos == -1:
        split_pos = segment.rfind(" ")
    if split_pos == -1:
        # Absolute fallback: hard cut but avoid splitting inside a tag
        split_pos = _safe_hard_cut(segment, max_chars)

    return remaining[: split_pos + 1].strip(), remaining[split_pos + 1 :]


def _split_first_chunk(text: str, cap: int) -> tuple[str | None, str]:
    """Cut a short first chunk ending at the first sentence boundary in ``text[:cap]``.

    The boundary must leave at least ``FIRST_CHUNK_MIN_CHARS`` characters in
    the chunk.  Falls back to the last clause boundary in the window, and
    returns ``(None, text)`` when there is no natural boundary at all, in
    which case the caller uses the normal splitting.
    """
    # Look one character past the cap so punctuation exactly at the cap only
    # counts when it is really followed by whitespace or the end of the text.
    window = text[: cap + 1]
    min_pos = FIRST_CHUNK_MIN_CHARS - 1

    sentence_ends = [pos for pos in _iter_sentence_ends(window) if min_pos <= pos < cap]
    if sentence_ends:
        pos = sentence_ends[0]
    else:
        clause_ends = [pos for pos in _iter_clause_boundaries(window) if min_pos <= pos < cap]
        if not clause_ends:
            return None, text
        pos = clause_ends[-1]

    return text[: pos + 1].strip(), text[pos + 1 :].lstrip()


def _iter_sentence_ends(text: str) -> Iterator[int]:
    """Yield the indices of real sentence-ending punctuation in *text*, ascending.

    Skips periods that follow common abbreviations (``Dr.``, ``Mr.``, etc.),
    decimal points, and punctuation inside bracket tags (``[laugh]``).  CJK
    sentence-ending punctuation (ideographic full stop, fullwidth exclamation
    and question marks) is included.
    """
    positions: list[int] = []
    for m in _SENTENCE_END_RE.finditer(text):
        pos = m.start()
        if text[pos] == ".":
            # Walk backwards to find the preceding word
            word_start = pos - 1
            while word_start >= 0 and text[word_start].isalpha():
                word_start -= 1
            word = text[word_start + 1 : pos].lower()
            if word in _ABBREVIATIONS:
                continue
            # Skip decimal numbers (digit immediately before the period)
            if word_start >= 0 and text[word_start].isdigit():
                continue
        if _inside_bracket_tag(text, pos):
            continue
        positions.append(pos)
    positions.extend(m.start() for m in _CJK_SENTENCE_END_RE.finditer(text))
    return iter(sorted(positions))


def _find_last_sentence_end(text: str) -> int:
    """Return the index of the last sentence-ending punctuation in *text*, or -1."""
    return max(_iter_sentence_ends(text), default=-1)


def _iter_clause_boundaries(text: str) -> Iterator[int]:
    """Yield the indices of clause-boundary punctuation in *text*, ascending."""
    for m in _CLAUSE_BOUNDARY_RE.finditer(text):
        pos = m.start()
        if _inside_bracket_tag(text, pos):
            continue
        yield pos


def _find_last_clause_boundary(text: str) -> int:
    """Return the index of the last clause-boundary punctuation, or -1."""
    return max(_iter_clause_boundaries(text), default=-1)


def _inside_bracket_tag(text: str, pos: int) -> bool:
    """Return True if *pos* falls inside a ``[...]`` tag."""
    return any(m.start() < pos < m.end() for m in _PARA_TAG_RE.finditer(text))


def _safe_hard_cut(segment: str, max_chars: int) -> int:
    """Find a hard-cut position that doesn't split a ``[tag]``."""
    cut = max_chars - 1
    # Check if the cut falls inside a bracket tag; if so, move before it
    for m in _PARA_TAG_RE.finditer(segment):
        if m.start() < cut < m.end():
            return m.start() - 1 if m.start() > 0 else cut
    return cut


def concatenate_audio_chunks(
    chunks: list[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 50,
) -> np.ndarray:
    """Concatenate audio arrays with a short crossfade to eliminate clicks.

    Each chunk is expected to be a 1-D float32 ndarray at *sample_rate* Hz.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result = np.array(chunks[0], dtype=np.float32, copy=True)

    for chunk in chunks[1:]:
        if len(chunk) == 0:
            continue
        overlap = min(crossfade_samples, len(result), len(chunk))
        if overlap > 0:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            result[-overlap:] = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
            result = np.concatenate([result, chunk[overlap:]])
        else:
            result = np.concatenate([result, chunk])

    return result


class CrossfadeJoiner:
    """Streaming equivalent of :func:`concatenate_audio_chunks`.

    The last ``crossfade_ms`` of audio pushed so far is held back because the
    next chunk may still blend into it.  :meth:`push` returns the samples that
    are safe to emit and :meth:`flush` returns the held tail at the end.
    Concatenating every returned array equals ``concatenate_audio_chunks`` on
    the same chunks, sample for sample: the blend uses the identical fade
    expressions on identical values.

    Sub-chunk pieces (``new_chunk=False``) are appended without blending.  The
    head of a new chunk is buffered until ``crossfade_ms`` of it has arrived,
    because the overlap depends on the chunk's length, not on its first piece.
    """

    def __init__(self, crossfade_ms: int = 50) -> None:
        self.crossfade_ms = crossfade_ms
        self.sample_rate: int | None = None
        self._crossfade_samples = 0
        self._held = np.array([], dtype=np.float32)
        self._total = 0
        self._started = False
        # Head of a new chunk whose crossfade has not been applied yet: the
        # overlap depends on the chunk's length, not on its first piece's.
        self._pending: list[np.ndarray] | None = None
        self._pending_len = 0

    def push(self, chunk: np.ndarray, sample_rate: int, *, new_chunk: bool = True) -> np.ndarray:
        """Add audio and return the samples that can be emitted now.

        ``new_chunk=False`` marks *chunk* as a continuation of the current
        chunk (no crossfade at the seam), which is how pieces from a streaming
        backend are joined.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if not self._started:
            self._started = True
            self.sample_rate = sample_rate
            self._crossfade_samples = int(sample_rate * self.crossfade_ms / 1000)
            self._held = np.array(chunk, dtype=np.float32, copy=True)
            self._total = len(chunk)
            return self._emit_ready()

        out: list[np.ndarray] = []
        if new_chunk:
            if self._pending is not None:
                out.append(self._commit_pending())
            self._pending = [chunk]
            self._pending_len = len(chunk)
        elif self._pending is not None:
            self._pending.append(chunk)
            self._pending_len += len(chunk)
        else:
            out.append(self._append(chunk))

        if self._pending is not None and self._pending_len >= self._crossfade_samples:
            out.append(self._commit_pending())
        return np.concatenate(out) if out else np.array([], dtype=np.float32)

    def _commit_pending(self) -> np.ndarray:
        """Blend the buffered chunk head into the held tail, exactly like the offline join."""
        pending = self._pending or []
        head = (
            np.concatenate(pending) if len(pending) > 1 else (pending[0] if pending else np.array([], dtype=np.float32))
        )
        self._pending = None
        self._pending_len = 0
        if len(head) == 0:
            return np.array([], dtype=np.float32)

        overlap = min(self._crossfade_samples, self._total, len(head))
        if overlap > 0:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            self._held[-overlap:] = self._held[-overlap:] * fade_out + head[:overlap] * fade_in
            self._held = np.concatenate([self._held, head[overlap:]])
        else:
            self._held = np.concatenate([self._held, head])
        self._total += len(head) - overlap
        return self._emit_ready()

    def _append(self, chunk: np.ndarray) -> np.ndarray:
        if len(chunk) == 0:
            return np.array([], dtype=np.float32)
        self._held = np.concatenate([self._held, chunk])
        self._total += len(chunk)
        return self._emit_ready()

    def _emit_ready(self) -> np.ndarray:
        # Keep exactly the samples a future chunk could still overlap with.
        keep = min(self._crossfade_samples, self._total)
        if keep >= len(self._held):
            return np.array([], dtype=np.float32)
        ready = self._held[: len(self._held) - keep]
        self._held = self._held[len(self._held) - keep :]
        return ready

    def flush(self) -> np.ndarray:
        """Return everything still held; call once after the last :meth:`push`."""
        out: list[np.ndarray] = []
        if self._pending is not None:
            out.append(self._commit_pending())
        out.append(self._held)
        self._held = np.array([], dtype=np.float32)
        return np.concatenate(out) if len(out) > 1 else out[0]


def _plan_chunks(
    text: str,
    seed: int | None,
    max_chunk_chars: int,
    first_chunk_chars: int | None = None,
) -> list[tuple[str, int | None]]:
    """Split *text* and assign each chunk a deterministic seed.

    A single chunk keeps the original text and seed so short inputs behave
    exactly like a direct ``backend.generate()`` call.
    """
    chunks = split_text_into_chunks(text, max_chunk_chars, first_chunk_chars=first_chunk_chars)
    if len(chunks) <= 1:
        return [(text, seed)]

    logger.info(
        "Splitting %d chars into %d chunks (max %d chars each)",
        len(text),
        len(chunks),
        max_chunk_chars,
    )
    # Vary the seed per chunk to avoid correlated RNG artefacts, but keep it
    # deterministic so the same (text, seed) pair always produces the same
    # output.
    return [(chunk_text, (seed + i) if seed is not None else None) for i, chunk_text in enumerate(chunks)]


async def _await_inflight(inflight: asyncio.Future) -> None:
    """Wait for an engine call that a cancellation left running.

    Backends synthesize in a worker thread that cannot be interrupted, so the
    caller waits for it before propagating the cancellation.  Otherwise the
    serial queue would start the next job while this chunk still holds the
    GPU.
    """
    if not inflight.done():
        await asyncio.wait({inflight})
    if not inflight.cancelled():
        inflight.exception()  # mark retrieved; the cancellation takes precedence


async def _generate_one_chunk(
    backend,
    chunk_text: str,
    chunk_seed: int | None,
    *,
    voice_prompt: dict,
    language: str,
    instruct: str | None,
    crossfade_ms: int,
    trim_fn: Callable | None,
    runaway_detector: Callable | None,
    retry_depth: int = 0,
) -> tuple[np.ndarray, int]:
    """Synthesize one chunk, retrying unstable output on smaller pieces and trimming."""
    inflight = asyncio.ensure_future(
        backend.generate(
            chunk_text,
            voice_prompt,
            language,
            chunk_seed,
            instruct,
        )
    )
    try:
        chunk_audio, chunk_sr = await asyncio.shield(inflight)
    except asyncio.CancelledError:
        await _await_inflight(inflight)
        raise

    if runaway_detector is not None and runaway_detector(chunk_audio, chunk_sr):
        if retry_depth >= MAX_RUNAWAY_RETRIES or len(chunk_text) <= MIN_RUNAWAY_RETRY_CHARS:
            raise RuntimeError("TTS output remained unstable after retrying smaller text chunks")

        retry_max_chars = max(MIN_RUNAWAY_RETRY_CHARS, len(chunk_text) // 2)
        retry_chunks = split_text_into_chunks(chunk_text, retry_max_chars)
        if len(retry_chunks) <= 1:
            raise RuntimeError("Unable to split unstable TTS output for retry")

        logger.warning(
            "Detected unstable TTS output for %d chars; retrying as %d smaller chunks",
            len(chunk_text),
            len(retry_chunks),
        )
        retry_audio: list[np.ndarray] = []
        sample_rate = chunk_sr
        for i, retry_text in enumerate(retry_chunks):
            retry_seed = chunk_seed + ((retry_depth + 1) * 1000) + i if chunk_seed is not None else None
            audio, sample_rate = await _generate_one_chunk(
                backend,
                retry_text,
                retry_seed,
                voice_prompt=voice_prompt,
                language=language,
                instruct=instruct,
                crossfade_ms=crossfade_ms,
                trim_fn=trim_fn,
                runaway_detector=runaway_detector,
                retry_depth=retry_depth + 1,
            )
            retry_audio.append(np.asarray(audio, dtype=np.float32))

        return (
            concatenate_audio_chunks(retry_audio, sample_rate, crossfade_ms=crossfade_ms),
            sample_rate,
        )

    if trim_fn is not None:
        chunk_audio = trim_fn(chunk_audio, chunk_sr)
    return np.asarray(chunk_audio, dtype=np.float32), chunk_sr


async def iter_generated_chunks(
    backend,
    text: str,
    voice_prompt: dict,
    *,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    first_chunk_chars: int | None = None,
    crossfade_ms: int = 50,
    trim_fn: Callable | None = None,
    runaway_detector: Callable | None = None,
) -> ChunkStream:
    """Yield ``(audio, sample_rate)`` for each text chunk as soon as it is synthesized.

    Chunks come out after runaway retry and trimming, exactly as
    :func:`generate_chunked` sees them before crossfading; no crossfade is
    applied between yields.
    """
    plan = _plan_chunks(text, seed, max_chunk_chars, first_chunk_chars)
    for i, (chunk_text, chunk_seed) in enumerate(plan):
        if len(plan) > 1:
            logger.info("Generating chunk %d/%d (%d chars)", i + 1, len(plan), len(chunk_text))
        yield await _generate_one_chunk(
            backend,
            chunk_text,
            chunk_seed,
            voice_prompt=voice_prompt,
            language=language,
            instruct=instruct,
            crossfade_ms=crossfade_ms,
            trim_fn=trim_fn,
            runaway_detector=runaway_detector,
        )


async def generate_chunked(
    backend,
    text: str,
    voice_prompt: dict,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    crossfade_ms: int = 50,
    trim_fn=None,
    runaway_detector=None,
) -> tuple[np.ndarray, int]:
    """Generate audio with automatic chunking for long text.

    For text shorter than *max_chunk_chars* this is a thin wrapper around
    ``backend.generate()`` with zero overhead.

    For longer text the input is split at natural sentence boundaries,
    each chunk is generated independently, optionally trimmed (useful for
    Chatterbox engines that hallucinate trailing noise), and the results
    are concatenated with a crossfade (or hard cut if *crossfade_ms* is 0).

    Parameters
    ----------
    backend : TTSBackend
        Any backend implementing the ``generate()`` protocol.
    text : str
        Input text (may be arbitrarily long).
    voice_prompt, language, seed, instruct
        Forwarded to ``backend.generate()`` verbatim.
    max_chunk_chars : int
        Maximum characters per chunk (default 800).
    crossfade_ms : int
        Crossfade duration in milliseconds between chunks.  0 for a hard
        cut with no overlap (default 50).
    trim_fn : callable | None
        Optional ``(audio, sample_rate) -> audio`` post-processing
        function applied to each chunk before concatenation (e.g.
        ``trim_tts_output`` for Chatterbox engines).
    runaway_detector : callable | None
        Optional ``(audio, sample_rate) -> bool`` detector. When it flags
        unstable output, the affected text is split in half and retried.

    Returns
    -------
    (audio, sample_rate) : Tuple[np.ndarray, int]
    """
    audio_chunks: list[np.ndarray] = []
    sample_rate: int | None = None

    async for chunk_audio, chunk_sr in iter_generated_chunks(
        backend,
        text,
        voice_prompt,
        language=language,
        seed=seed,
        instruct=instruct,
        max_chunk_chars=max_chunk_chars,
        crossfade_ms=crossfade_ms,
        trim_fn=trim_fn,
        runaway_detector=runaway_detector,
    ):
        audio_chunks.append(chunk_audio)
        if sample_rate is None:
            sample_rate = chunk_sr

    return concatenate_audio_chunks(audio_chunks, sample_rate, crossfade_ms=crossfade_ms), sample_rate


def _backend_stream_fn(backend) -> Callable | None:
    """Return the backend's optional ``generate_stream`` method, if it has one."""
    fn = getattr(backend, "generate_stream", None)
    return fn if callable(fn) else None


async def _stream_backend_chunk(
    stream_fn: Callable,
    chunk_text: str,
    chunk_seed: int | None,
    *,
    voice_prompt: dict,
    language: str,
    instruct: str | None,
    runaway_cut_fn: Callable | None,
) -> AsyncIterator[tuple[np.ndarray, int, bool]]:
    """Yield ``(piece, sample_rate, new_chunk)`` from a backend's own audio stream.

    When *runaway_cut_fn* reports that the audio produced so far contains
    speech, a long silence and then more output (a missed end-of-speech), the
    chunk is cut at the start of that silence and the rest of its stream is
    discarded.  Nothing after the cut has been emitted yet, because detection
    needs the post-silence output that arrives in the same piece.
    """
    so_far = np.array([], dtype=np.float32)
    first = True
    stream = stream_fn(chunk_text, voice_prompt, language, chunk_seed, instruct)
    try:
        async for piece, sample_rate in stream:
            piece = np.asarray(piece, dtype=np.float32)
            if runaway_cut_fn is not None:
                combined = np.concatenate([so_far, piece])
                cut = runaway_cut_fn(combined, sample_rate)
                if cut is not None:
                    keep = max(0, cut - len(so_far))
                    logger.warning(
                        "Detected unstable streamed TTS output; cutting chunk at %.2fs",
                        cut / sample_rate,
                    )
                    if keep:
                        yield piece[:keep], sample_rate, first
                    return
                so_far = combined
            yield piece, sample_rate, first
            first = False
    finally:
        await stream.aclose()


async def generate_chunked_stream(
    backend,
    text: str,
    voice_prompt: dict,
    *,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    first_chunk_chars: int | None = DEFAULT_FIRST_CHUNK_CHARS,
    crossfade_ms: int = 50,
    trim_fn: Callable | None = None,
    runaway_detector: Callable | None = None,
    runaway_cut_fn: Callable | None = None,
    allow_backend_stream: bool = True,
) -> ChunkStream:
    """Yield playable ``(audio, sample_rate)`` pieces as each chunk finishes.

    Text is split exactly like :func:`generate_chunked` (plus the optional
    small first chunk), chunks are synthesized one after another, and the
    seams are crossfaded on the fly by :class:`CrossfadeJoiner`.  When the
    backend offers ``generate_stream`` and no per-chunk trimming is needed,
    sub-chunk pieces are forwarded as the engine produces them; runaway
    detection then uses *runaway_cut_fn* (cut and continue) instead of
    *runaway_detector* (retry), because audio that was already sent cannot
    be regenerated.
    """
    plan = _plan_chunks(text, seed, max_chunk_chars, first_chunk_chars)
    joiner = CrossfadeJoiner(crossfade_ms)

    stream_fn = None
    if allow_backend_stream and trim_fn is None and (runaway_detector is None or runaway_cut_fn is not None):
        stream_fn = _backend_stream_fn(backend)

    for i, (chunk_text, chunk_seed) in enumerate(plan):
        if len(plan) > 1:
            logger.info("Streaming chunk %d/%d (%d chars)", i + 1, len(plan), len(chunk_text))

        if stream_fn is not None:
            async for piece, piece_sr, new_chunk in _stream_backend_chunk(
                stream_fn,
                chunk_text,
                chunk_seed,
                voice_prompt=voice_prompt,
                language=language,
                instruct=instruct,
                runaway_cut_fn=runaway_cut_fn if runaway_detector is not None else None,
            ):
                ready = joiner.push(piece, piece_sr, new_chunk=new_chunk)
                if len(ready):
                    yield ready, piece_sr
            continue

        chunk_audio, chunk_sr = await _generate_one_chunk(
            backend,
            chunk_text,
            chunk_seed,
            voice_prompt=voice_prompt,
            language=language,
            instruct=instruct,
            crossfade_ms=crossfade_ms,
            trim_fn=trim_fn,
            runaway_detector=runaway_detector,
        )
        ready = joiner.push(chunk_audio, chunk_sr)
        if len(ready):
            yield ready, chunk_sr

    tail = joiner.flush()
    if len(tail) and joiner.sample_rate is not None:
        yield tail, joiner.sample_rate
