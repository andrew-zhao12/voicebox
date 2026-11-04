# Speech additions: local validation, 2026-11-03

These checks ran on Apple Silicon macOS with Python 3.12. The optional Nemotron
runtime used Transformers 5.13 in an isolated environment; the main server retained
its existing Transformers 4.57 environment. No CUDA hardware was used.

| Path | Validation |
|---|---|
| faster-whisper Base | Real CPU INT8 inference, exact synthetic sentence, nine aligned words, prompt and vocabulary terms |
| Nemotron English 0.6B | Full checkpoint on CPU, incremental 100 ms input, cached encoder/RNNT state, partial and final text |
| Nemotron multilingual 0.6B | Actual Transformers 5.13 tiny architecture; independent caches, packetization invariance, short final flush, token parity with upstream generation |
| PyTorch Whisper Base | CPU word alignment with prompt and vocabulary, exact synthetic sentence and nine aligned words |
| MLX Whisper Base | Sample-rate/no-hallucination regressions plus real human audio with 17 aligned words; model-specific alignment heads loaded from checkpoint metadata |
| Incremental TTS socket | Real Kokoro audio over repeated flushes, pronunciation replacements, cancellation after audio began |
| Native STT socket | Human speech produced partials before commit and the expected final transcript; unload refused while active, then succeeded after final, clear, and disconnect during inference |

The native CPU socket test paced input slowly enough for that machine. It verifies
protocol behavior and decoder cleanup, not realtime CPU performance. The full
multilingual checkpoint was not downloaded or evaluated for recognition quality.
Frozen executable packaging was updated but a complete frozen release build was
not validated.

## Human-audio baseline

The benchmark used three clips from the [Hugging Face LibriSpeech test fixture](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy),
IDs `1272-128104-0000`, `1272-128104-0001`, and `1272-128104-0002`.
The source corpus is [LibriSpeech](https://www.openslr.org/12), distributed under
CC BY 4.0. These clips have one speaker and do not represent noisy, multilingual,
telephone, or production traffic.

Whisper Base on MLX, warm runs, two repetitions per clip:

| Concurrent HTTP requests | Successful requests | p50 / p95 completion latency | WER | CER |
|---|---|---|---|---|
| 1 | 6 / 6 | 54 / 79 ms | 6.67% | 4.63% |
| 2 | 6 / 6 | 97 / 127 ms | 6.67% | 4.63% |

The benchmark's documented normalization preserves lexical differences such as
`Mr` versus `MISTER`, so abbreviation differences contribute to these error rates.
These small warm-run measurements are a functional baseline, not a capacity claim.
The same first clip transcribed through native Nemotron's WebSocket had zero
normalized word errors.

Use [the benchmark walkthrough](08-speech-benchmarks.md) to repeat measurements on
representative recordings and the intended deployment hardware. GPU latency,
VRAM, higher concurrency, and full multilingual checkpoint accuracy remain to be
measured there. PyTorch allocator metrics exclude the isolated Nemotron worker and
CTranslate2 allocations; use separate host/process GPU telemetry for those.
