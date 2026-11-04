# Speech conversation reliability

The fast integration suites use local protocol peers and no speech models. This
runbook adds real speech on repeated turns, interruption cycles, idle gaps and
server restarts. The scripts preserve existing servers and containers; use a
scratch Voicebox instance and a dedicated client key when running directly.

## Repeated conversations on a running server

Install each integration's requirements in its own environment. Set
`VOICEBOX_URL` and `VOICEBOX_CLIENT_KEY`; LiveKit can also use
`VOICEBOX_ADMIN_KEY` to verify server-side cancellation counters. Preload
`kokoro,whisper-base` and wait for `/health/ready`.

```bash
# Three interruption cycles (six caller utterances) in the same LiveKit room.
/tmp/livekit/bin/python integrations/livekit/e2e/run_e2e.py \
  --tts-transport websocket --turns 3 --barge-in --idle-between 18 \
  --container voicebox-reliability-livekit --port-base 7980 --out-dir /tmp/lk-reliability

# Three interruption cycles on the same Pipecat WebRTC connection.
VOICEBOX_TTS_TIMEOUT=10 /tmp/pipecat/bin/python integrations/pipecat/e2e.py \
  --tts websocket --turns 3 --barge-in --idle-between 18 --out-dir /tmp/pc-reliability
```

Pipecat's short TTS progress timeout closes an idle speech socket between turns,
then reconnects when fresh text arrives. It still fails pending speech that makes
no progress. The continuous microphone sends silence while idle, so a quiet gap
alone does **not** prove STT socket reconnection. Server-free protocol tests cover
transport failure, no replay, pending-phrase backpressure and stalled playback.

Every cycle verifies an audible reply, caller-to-reply latency and (in echo mode)
its transcript. Barge-in requires speech to still be audible when interruption
starts, measures the time until silence, rejects resumed speech over the caller,
and checks the follow-up reply. Recordings and completed-turn progress are written
as the call runs; recorder memory is released between cycles.

`--turns` is the minimum number of reply cycles. Pipecat `--duration 1800` keeps
one conversation going for at least 30 minutes; it finishes the current cycle
before stopping. LiveKit `--duration 1800` runs rooms on the same worker until at
least 30 minutes have elapsed; `--turns` controls each room's length. For one long
LiveKit room, use `--turns 60 --idle-between 18 --session-timeout 3600`. Its per-room
hard timeout is independent of the minimum soak duration. Increase latency bounds
only to reflect the hardware under test, and retain the measured values.

## Isolated Docker runs and server restart

Build a CPU image from the checkout. The orchestrator starts it on an automatically
assigned loopback port, creates an ephemeral client key, runs both integrations,
then restarts **its own container** and repeats. Existing services are untouched.
The CPU image downloads Kokoro and Whisper Base on first preload; allow time and
disk space for those weights.

```bash
docker build --build-arg PYTORCH_VARIANT=cpu -t voicebox:reliability .
python3 scripts/run_speech_reliability.py --image voicebox:reliability \
  --livekit-python /tmp/livekit/bin/python --pipecat-python /tmp/pipecat/bin/python \
  --rounds 2 --turns 3 --duration 180 --out-dir /tmp/voicebox-reliability
```

A restart occurs **between calls**, checking readiness and persisted credentials
before new calls. The runner resolves the published port again because Docker
can assign a different ephemeral port after restart. It does not claim transparent
recovery of an interrupted in-flight response. The adapters deliberately report
partial transport failures without replaying text. Timeout and termination cleanup
stop only owned driver process groups and containers. Temporary credential files
are removed; reports, synthetic WAVs and diagnostic logs remain for review.
After each integration pass, the runner requires pending/running queue jobs, realtime/open
WebSocket sessions and pending speech flushes to return to zero. These checks
catch work left behind after the participant and worker disconnect.
The owned server closes idle speech sockets after 10 seconds while preserving the
120-second client allowance for pending synthesis. With multiple Pipecat turns,
the report requires server idle closures and fresh accepted connections across
the 18-second quiet gaps; one-turn passes explicitly skip this check.

`.github/workflows/speech-reliability.yml` provides manual and weekly CPU runs,
separate from PR CI. It builds the checkout's server, installs separate SDK
environments, and retains artifacts for seven days. It needs no paid LLM key.
The default 30-second CPU reply allowance is a failure bound, not a latency SLO.

## Using an actual conversational LLM

Both harnesses accept `--llm configured`. Configure the usual `LLM_BASE_URL`,
`LLM_MODEL` and `LLM_API_KEY` (or `OPENAI_API_KEY`) before invoking it. This mode
uses the normal chat provider and verifies intelligible audio and timing, while
retaining the deterministic greeting check. It skips echo-specific reply WER:
a valid conversational answer need not repeat the caller. Recorded replies still
need a task-specific quality review. Long answers are required to exercise
`--barge-in`; a reply ending before the interruption is an explicit test failure.

Echo remains the default regardless of external credentials. The Docker/CI
orchestrator always selects echo to avoid accidental external model calls.

## Local evidence

On Apple Silicon with Kokoro and MLX Whisper Base, three repeated interruption
cycles passed on each integration using the new harnesses:

| Integration | Same-call duration | Stop latency range | Follow-up latency range | Transcript WER |
|---|---|---|---|---|
| LiveKit 1.8.10 | 92.6 s including greeting | 0.717–0.718 s | 1.357–1.719 s | 0 |
| Pipecat 1.15.0 | 83.4 s after greeting | 0.857–0.879 s | 0.931–0.952 s | 0 |

LiveKit metrics recorded three cancellations and no abandoned stream. Pipecat
used a 10-second speech timeout and 18-second idle gaps. These are synthetic
functional checks, not representative speech-quality or production-capacity
measurements. No external LLM credentials were available for an actual-model run.

The Linux CPU checks also caught idle transcription caused by Pipecat's integer
SoXR resampler adding one-LSB dither to digital silence. The Voicebox realtime
adapter now uses float filtering with explicit PCM rounding, preserving stream
tails and quiet speech while keeping silence exact. Real resampler regressions
cover zero input, one-LSB speech, several packet sizes, flush and reset behavior.

A final run on 2026-11-03 against a Docker Linux ARM64 CPU backend (Kokoro and
PyTorch Whisper Base), with SDK clients on macOS, passed two cycles per integration before and
after restarting the owned server: eight interruption cycles and sixteen caller
utterances in total. Every greeting and follow-up transcript had WER 0.

| Integration | Per-pass call duration | Stop latency range | Follow-up latency range |
|---|---|---|---|
| LiveKit 1.8.10 | 65.8–66.5 s including greeting | 0.709–0.758 s | 3.538–4.150 s |
| Pipecat 1.15.0 | 53.2–53.3 s after greeting | 0.841–0.899 s | 1.410–1.932 s |

Both Pipecat passes recorded one server idle closure and two accepted speech
connections. All pending/running job, realtime session, open WebSocket and pending
flush gauges returned to zero after each of the four passes. Restart reused the
persisted client key through Docker's reassigned host port. The runner exited 0,
removed its containers and temporary credentials, and left the host server and
unrelated containers untouched. Artifacts for this run are in
`/tmp/voicebox-reliability-current`'s referenced directory under
`docker-reliability-restart/` (summary, recordings and logs).

The CPU run also exercises delayed transcription fragments across short VAD
pauses. Realtime pipelines must pair the adapter with
`VoiceboxSpeechTimeoutUserTurnStopStrategy(stt)`, as documented in the Pipecat
README: the turn stays open until every local commit finishes, preserving late
words without a fixed extra delay. These functional measurements use each
harness's audio/VAD timing conventions; they are not a cross-framework benchmark.
