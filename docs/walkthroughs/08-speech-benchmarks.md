# 8. Real-audio accuracy and inference capacity

**Goal.** Measure a chosen Voicebox server with your recordings, compare STT models,
and find how HTTP inference latency changes as concurrency increases. The benchmark
writes reproducible JSON/CSV reports. It can also observe streaming cancellation
and sample the server's CUDA allocator metrics.

This guide supplies commands, not hardware results. No production recordings or
GPU measurements are included. The tests use a local HTTP stub with synthetic
PCM to verify the benchmark itself. The earlier LiveKit/Pipecat echo checks measure
a different workload and do not establish GPU call capacity.

## 1. Prepare a corpus

Copy [`manifest.example.json`](../../examples/speech-benchmarks/manifest.example.json)
to a local directory outside the repository. Add the recordings it names, or
replace those entries with your files. Paths resolve relative to the manifest.
The template contains no audio, so it cannot run until those paths are replaced.

```text
my-corpus/
  manifest.json
  recordings/
    real-clear-001.wav
    real-noisy-001.wav
    synthetic-control-001.wav
```

Each `audio` entry needs a unique `id`, a `path`, an exact reference `text`, and
`source: "real"` or `source: "synthetic"`. Optional `language` supplies the
transcription language. Use your own recordings with representative speakers,
accents, microphones, background noise, interruptions, silence and domain terms.
Transcribe the actual recording by hand for its reference; a TTS script is not a
reference for a different human recording.

Files must be nonempty, uncompressed PCM16 WAV, mono or stereo, at 8 kHz or higher.
The default size limit is 32 MiB per file (`--max-audio-mb` changes it). The script
checks the WAV payload, rejects duplicate IDs and unlabeled inputs, and records
SHA-256 hashes of the manifest and every recording. It checks each recording's
hash again before upload, so editing a fixture during a run fails that request.
The source labels are your declarations; the tool cannot determine whether speech
was recorded from a human.

Delete the synthetic control if you do not need it. Real and synthetic STT results
are always reported separately. TTS output is always labeled synthetic.

Each `tts` entry needs an `id`, `model`, `voice`, and `text`. Choose voices supported
by that model; for example, the supplied `af_heart` preset belongs to Kokoro.
For a cloning engine, use a compatible profile name or ID already on the server.
You can omit `tts` to run only transcription, or omit `audio` to run only speech.
At least one case is required. Include short replies and long multi-sentence
replies when evaluating streaming and cancellation.

## 2. Prepare a dedicated test replica

Use a scratch/development server with the models you intend to evaluate. The script
sends real inference requests and never creates, unloads, deletes or restarts
models explicitly. Requests may lazily load models. Multiple engines can remain
resident, so the model snapshots and the model order matter when comparing VRAM.
Do not interpret a model's allocator peak as its isolated memory footprint.
The allocator gauges cover the main server process. The isolated Nemotron worker
and CTranslate2 allocations require separate host/process GPU telemetry; they are
not represented by these PyTorch allocator gauges.

Set credentials through environment variables. The client key performs inference;
the optional admin key performs only `GET /metrics` and `GET /models/status`.
Neither key is stored in the report or accepted as a command-line option.

```sh
export VOICEBOX_URL=http://127.0.0.1:17494
read -s VOICEBOX_CLIENT_KEY
export VOICEBOX_CLIENT_KEY
read -s VOICEBOX_ADMIN_KEY
export VOICEBOX_ADMIN_KEY
```

Use the backend's Python environment, which already includes `httpx`, or install
`httpx` into a separate Python 3.12 environment. The benchmark imports no backend
models, torch, NumPy or audio ML libraries.

For concurrency tests, configure the test key's inference/request/pending-job
limits to admit the workload, or keep the defaults to measure the limits. A `429`
is reported as an error and makes the command exit nonzero; it is never counted
as successful capacity. Leave unrelated traffic off the test replica if you want
to attribute global metrics and cancellation counters to this run.

## 3. Run a baseline and a concurrency sweep

From the repository root:

```sh
backend/venv/bin/python scripts/speech_benchmark.py /path/to/my-corpus/manifest.json \
  --output /path/to/results/cpu-baseline-001 \
  --server-label 'dedicated CPU replica; Kokoro; Whisper Base' \
  --stt-models whisper-base \
  --concurrency 1,2,4 --repeats 5 --warmup 1
```

`--output` must be a new directory; the script refuses to overwrite old reports.
Every case is repeated `--repeats` times at each concurrency. A bounded set of
workers runs the requests. `max_in_flight` records the actual peak, which may be
less than the requested concurrency when the corpus/repeat count is too small.

Warm-up requests are recorded separately and excluded from measured percentiles
and throughput. A failed warm-up skips that model/source group and makes the run
fail. `--warmup 0` includes whatever loading the server performs in the measured
requests; it does not guarantee a cold model. Use a fresh replica when you need a
known cold start.

Compare explicit STT model IDs sequentially:

```sh
backend/venv/bin/python scripts/speech_benchmark.py /path/to/my-corpus/manifest.json \
  --output /path/to/results/gpu-stt-comparison-001 \
  --server-label 'replace with GPU model, VRAM, image/commit and replica settings' \
  --stt-models whisper-base,whisper-small,whisper-turbo \
  --concurrency 1,2,4,8 --repeats 10 --warmup 1 \
  --timeout 180 --sample-interval 0.5
```

Use IDs exposed by the server you are testing. `whisper-1` is an alias for its
configured current model, so explicit IDs are preferable for comparison. TTS
models/voice pairings come from the manifest. Include hardware, deployment image
or commit, backend variant, generation-worker settings, model versions and key
limits in `--server-label` or retain them beside the report; the client cannot
infer those deployment settings from a remote URL.

## 4. Read the outputs

- `report.json`: validated corpus metadata/hashes, run settings, client versions,
  per-request results, warm-ups, summaries, cancellation probes and observations.
- `requests.csv`: measured requests with status, elapsed time, first PCM payload,
  audio duration, RTF and transcription/error counts.
- `summaries.csv`: one row per operation/model/source/voice/concurrency group,
  including p50/p95, errors, rate limits, throughput and weighted accuracy.
- `cancellations.csv`: optional cancellation probes; nested request/counter data
  is JSON in a CSV cell. Empty when probes were disabled.

The console shows counts and artifact paths, not recording contents or keys.
The reports include reference text and returned transcripts for error review.
A nonzero exit means inference/warm-up errors occurred; JSON/CSV still capture
those completed results. Compare accuracy only alongside the error count: WER
and CER cover successful transcriptions, not requests rejected by the server.

| Field | Meaning |
|---|---|
| `wer` / `cer` | Sum of edit counts divided by total reference words/characters across successful requests; rates can exceed 1 with insertions |
| `latency_s` | HTTP request start through complete response, including upload, server queueing, inference and network |
| `first_audio_s` | First nonempty raw PCM payload on TTS; includes leading silence and is not audible onset |
| `rtf` | Wall seconds divided by audio seconds; **lower is faster** |
| `audio_per_wall_x` | Audio seconds divided by sweep wall time; **higher is faster** |
| `successful_requests_per_s` | Successful completed requests divided by sweep wall time |
| `*_p50`, `*_p95` | Nearest-rank percentiles of successful requests; small sample sizes give coarse estimates |

Text comparison uses Unicode NFKC, case folding, punctuation/symbols replaced by
spaces, and collapsed whitespace. CER includes those normalized spaces. Numbers
are not rewritten as words. Keep this normalization constant across comparisons.

TTS requests use the `/v1` contract's mono PCM16 at 24 kHz. Counting raw payload
avoids treating a WAV header as first audio. HTTP failures, broken transfers,
empty audio and odd byte counts are errors. No retries hide rate limiting or
transient failures.

## 5. Check cancellation separately

Add a probe for each TTS case:

```sh
backend/venv/bin/python scripts/speech_benchmark.py /path/to/my-corpus/manifest.json \
  --output /path/to/results/cancellation-001 \
  --concurrency 1 --repeats 3 --warmup 1 \
  --cancel-after 0.2 --cancel-observe 15
```

The probe cancels at the requested wall time, including before response headers
or after PCM begins. `client_close_s` measures how quickly the local HTTP task
closed. It does **not** prove that the GPU stopped. If the request finished first,
the report says `request_finished_before_cancel` and makes no cancellation claim.
Use a longer TTS case if that happens.

With the admin key, the script polls the global speech-499 and cancelled/abandoned
stream counters. `server_cancel_observed_s_upper_bound` records the first scrape
that observes an increase. It includes polling delay. These counters are global,
can overlap for one request, and do not identify a particular request. Concurrent
traffic invalidates causal attribution. Missing metrics, denied access or a
counter reset are explicit; a null observation means cancellation was not proven
within the observation window. Cancellation results do not by themselves fail an
otherwise successful throughput run; inspect the probe fields before claiming
that cancellation works.

A model's in-flight inference call may finish after the connection closes. Native
model interruption and immediate VRAM release are not implied by either timing.

## 6. Interpret GPU observations and capacity

`observations.snapshots` records models and counters before/after model groups.
`observations.samples` polls `/metrics` during requests. When the server exports
`voicebox_gpu_memory_bytes`, `gpu_allocator_peak_bytes` gives observed allocated
and reserved peaks. Missing data is `null`, never zero.

These are sampled **PyTorch CUDA allocator** values from the server process. They
do not measure whole-device VRAM, other processes, non-PyTorch runtimes, MLX or
ROCm. Brief peaks between scrapes can be missed. The model-status endpoint reports
residency, not an independently verified GPU type or available device memory.
For total-device measurements, collect the host's GPU telemetry beside the report.

This script measures HTTP TTS and complete-utterance HTTP STT. It does not simulate
a LiveKit/Pipecat call, realtime STT sessions, VAD, LLM latency, WebRTC jitter or SIP.
The highest successful HTTP concurrency is not a supported simultaneous-call
count. Use these sweeps to select candidates, then run the integration e2e checks
and a representative multi-call workload on the target hardware.

## 7. Verify the benchmark itself

```sh
backend/venv/bin/python -m pytest backend/tests/test_speech_benchmark.py -q
backend/venv/bin/ruff check --config backend/pyproject.toml \
  scripts/speech_benchmark.py backend/tests/test_speech_benchmark.py
```

These tests run without torch or downloaded models. Their local HTTP stub covers
real/synthetic grouping, weighted accuracy, bounded concurrency, PCM timing,
429/503 failures, interrupted transfers, cancellation before/after headers,
allocator/model observations, artifact writing and credential isolation. Stub
measurements validate code behavior and are not speech-quality or hardware data.
