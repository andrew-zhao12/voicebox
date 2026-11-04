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

## Native integration adapters

The opt-in `VOICEBOX_TTS_TRANSPORT=websocket` adapters were checked against an
isolated Voicebox server with Kokoro, MLX Whisper Base, and an echo LLM. Each row
below is one warm local session, measured from the end of the user's speech to
the reply's first audio. These are functional comparisons, not evidence of a
latency improvement or production capacity.

| Integration | HTTP reply latency | WebSocket reply latency | Audio/transcript result |
|---|---|---|---|
| LiveKit Agents 1.8.10, real room | 1.480 s | 1.515 s | Greeting and reply WER 0 |
| Pipecat 1.15.0, SmallWebRTC | 1.118 s | 1.099 s | Greeting and reply WER 0 |

LiveKit's reported TTS time to first byte was 0.372 s over HTTP and 0.362 s over
WebSocket. A separate WebSocket barge-in session stopped audio in 0.710 s,
recorded one backend stream cancellation and no abandoned stream, and delivered
the follow-up reply in 1.600 s with WER 0. The room harness accepts
`--tts-transport websocket --barge-in`; Pipecat's harness accepts
`--tts websocket --stt realtime`.

Server-free SDK tests cover incremental text, PCM metadata and sequence checks,
bounded input or pending phrases, interruption, partial failure without replay,
and cleanup. LiveKit was checked against both its minimum 1.8.6 and 1.8.10 SDK;
Pipecat uses its pinned 1.15.0 audio-context API. HTTP remains the default while
voice continuity and latency are evaluated with real LLM output.

After review fixed Pipecat's idle reconnect and completion-count checks, a second
real WebRTC session passed with WER 0 and 1.080 s reply latency. The final suites
passed 64 LiveKit tests on each supported SDK version checked and 39 Pipecat tests.

## Optional-runtime containers

Linux ARM64 CPU images built successfully with `VOICEBOX_STT_RUNTIMES=all` and
the default `none` profile. `scripts/docker-stt-smoke.sh` passed with networking
disabled and no host ports. It checked:

- Startup rejection of missing runtimes and a Nemotron interpreter pointing at
  the incompatible main Transformers environment.
- Actual faster-whisper/PyAV audio decoding and CTranslate2 CPU INT8 support.
- Tiny random-weight English and multilingual Nemotron inference, independent
  caches, packetization parity, and final-buffer handling.
- Server readiness, authenticated model capability discovery, and clean shutdown.

The isolated Nemotron environment also passed `pip check`. Complete dependency
solves passed against the exact CPU Torch pin for AMD64 and ARM64 and the CUDA
12.8 Torch pin for AMD64. That check caught and fixed a CUDA-only `setuptools`
constraint; lock regeneration now verifies all three targets.

The AMD64 and CUDA images were not built or executed locally. The Docker CI
matrix is configured to build CPU and CUDA 12.8 images and run the offline smoke
on AMD64 CPU runners; its result is still pending. Real CUDA inference, full-model
quality, and GPU capacity require separate validation.
