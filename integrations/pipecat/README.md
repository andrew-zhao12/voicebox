# Pipecat with Voicebox

A local browser voice agent using Voicebox for speech and transcription, and an
OpenAI-compatible LLM for replies. Pipecat handles the WebRTC connection, Silero
voice activity detection, silence-based turn endings, and interruptions.

The example is pinned to **Pipecat 1.15.0, Python 3.11+**. Install it in its own
environment; do not add Pipecat to `backend/venv` or the Voicebox server image.

```sh
cd integrations/pipecat
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start a Voicebox server and preload a speech engine and Whisper. Create a client
key from the Voicebox repository root (add `--data-dir` if the server uses a
separate data directory):

```sh
backend/venv/bin/python -m backend.keys create --id pipecat-agent --role client \
  --limit inference=600 --limit requests=1200 \
  --limit max_pending_jobs=16 --limit max_realtime_sessions=8
```

The server's global realtime session limit defaults to four; increase
`VOICEBOX_REALTIME_MAX_SESSIONS` on the server for more simultaneous realtime calls.
Warm models before accepting calls. Batch STT consumes one inference request per
utterance; HTTP speech consumes one per sentence and WebSocket speech one per
flushed phrase. Realtime STT holds one socket per call. WebSocket TTS adds one
speech socket, so a call using both consumes **two** of the client's shared
`max_realtime_sessions` allowance. Speech sockets also count against the server's
`VOICEBOX_SPEECH_MAX_SESSIONS` limit (default four).

From `integrations/pipecat`, with the integration environment active:

```sh
export VOICEBOX_URL=http://127.0.0.1:17493
export VOICEBOX_CLIENT_KEY=<client-key>
export OPENAI_API_KEY=<llm-key>
python bot.py -t webrtc
```

Open **http://localhost:7860/client/**, select SmallWebRTC if prompted, and connect.
Allow microphone access and use headphones. The assistant greets you and answers
each spoken turn. No LiveKit or Daily account is needed. For local testing without
an LLM account, use `export VOICEBOX_LLM=echo`; it answers “You said: …”.

| Setting | Default | Meaning |
|---|---|---|
| `VOICEBOX_URL` | `http://127.0.0.1:17493` | Server root or `/v1` URL; Docker Compose usually publishes port 17600 |
| `VOICEBOX_CLIENT_KEY` | required | Dedicated Voicebox client key; never the server's `VOICEBOX_API_KEY` |
| `VOICEBOX_VOICE` | `af_heart` | Preset voice or cloned profile ID/name from `/v1/voices` |
| `VOICEBOX_TTS_MODEL` | `tts-1` | Voice's own engine, or an explicit compatible engine |
| `VOICEBOX_TTS_TRANSPORT` | `http` | `http` sentence synthesis or `websocket` incremental phrase synthesis |
| `VOICEBOX_PRONUNCIATIONS_FILE` | unset | JSON object of literal whole-term replacements, e.g. `{"API":"A P I"}` |
| `VOICEBOX_TTS_SPEED` | server default | Speech rate, 0.25–4 |
| `VOICEBOX_TTS_INSTRUCTIONS` | unset | Engine-supported delivery instructions, at most 500 characters |
| `VOICEBOX_FIRST_CHUNK_CHARS` | server default | Backend first audio chunk text budget, 20–1000 characters |
| `VOICEBOX_MAX_CHUNK_CHARS` | server default | Backend subsequent chunk text budget, 100–5000 characters |
| `VOICEBOX_TEXT_FLUSH_CHARS` | `120` | WebSocket only: flush at a word boundary near this many characters, 20–1000 |
| `VOICEBOX_TTS_TIMEOUT` | `120` | WebSocket only: seconds allowed for speech progress, 1–1800 |
| `VOICEBOX_STT_MODEL` | `whisper-1` | Server's current Whisper model, or a specific Whisper model |
| `VOICEBOX_STT_REALTIME` | `1` | `1`: GA transcription WebSocket; `0`: one HTTP WAV upload per utterance |
| `VOICEBOX_LANGUAGE` | `en` | Pipecat language code |
| `VOICEBOX_LLM` | `openai` | `openai` or deterministic local `echo` |
| `LLM_BASE_URL` | OpenAI | Chat API, e.g. `http://127.0.0.1:11434/v1` for Ollama |
| `LLM_API_KEY` | `OPENAI_API_KEY` for OpenAI | With a custom URL, defaults to `local`; set the provider's key explicitly |
| `LLM_MODEL` | `gpt-4.1-mini` | Chat model, independent of the Voicebox speech models |

The runner reads exported environment variables. It does not load `.env` files.
Change the listen port with `--port`. The bundled runner is for local development;
an internet deployment needs your application's session authentication and HTTPS
plus appropriate STUN/TURN configuration. The example only configures the WebRTC
transport, even though Pipecat's shared CLI lists other transports.

## Adapter behavior

`services.py` contains the small compatibility layer:

- Pipecat's stock OpenAI TTS service rejects voices outside OpenAI's fixed catalog.
  `VoiceboxTTSService` instead sends the selected Voicebox voice unchanged and
  streams PCM16 mono at **24 kHz**, matching Voicebox's `/v1/audio/speech` contract.
- Batch and realtime STT inherit Pipecat's OpenAI services. The wrappers mark
  completed utterances as final so turn handling does not wait for additional
  transcripts, and close the batch SDK client's connection pool at session end.
- Local Silero VAD commits realtime audio. Server VAD is explicitly disabled.
  Turns finish after a silence window and a completed transcript; this example
  does not download or depend on a semantic end-of-turn model.
- Interruptions cancel synthesis through Pipecat. Closing the HTTP stream lets
  Voicebox abandon the queued request or stop between synthesis chunks. A chunk
  already running on the backend still completes.
- Slow or queued speech may wait up to 120 seconds for HTTP progress. Its audio
  context stays open during that wait; empty responses, malformed PCM, and HTTP
  errors produce pipeline errors and close the context promptly.

For incremental input, set `VOICEBOX_TTS_TRANSPORT=websocket`. The optional
`VoiceboxStreamingTTSService` in `streaming_tts.py` uses Pipecat's **token**
aggregation, flushes at sentence punctuation or the configured phrase length,
and flushes the final remainder at the LLM response boundary. It can start audio
while the LLM is still producing a sentence. Text is held locally until each
phrase can be appended and flushed atomically; Voicebox only synthesizes flushed
text. A long word is split only at the 1000-character safety bound.

The adapter keeps at most four 1000-character phrases outstanding, under the
server's shared 10000-character buffer limit. It limits audio queued ahead of SDK
playback to 64 frames and fails a consumer stalled for five seconds. Interruptions
cancel the affected flush IDs and discard their unsent text, preserving explicitly
uninterruptible SDK speech contexts. A backend chunk already executing may finish.
Connection loss, invalid audio, and partial synthesis failures fail the pipeline;
submitted text is never replayed. Idle sockets may reconnect for subsequent work.
Both transports output PCM16 mono at 24 kHz and close their resources at teardown.

`VoiceboxTTSService.Settings` and `VoiceboxStreamingTTSService.Settings` support
Pipecat `TTSUpdateSettingsFrame` updates for model, voice, language, speed,
instructions, pronunciation replacements, and backend chunk sizes. Streaming
settings are snapshotted per LLM turn; `text_flush_chars` sets its phrase budget.
Pronunciation maps allow at most 100 entries, keys of 128 characters, values of
256 characters, and 8192 total characters. Matches are evaluated within a phrase;
keep multiword replacement terms away from a forced phrase boundary.

HTTP remains the default: phrase size trades startup latency against prosody and
inference request volume. Validate voices and phrase budgets for your workload
before changing it. HTTP accepts backend chunk controls, but rejects the two
WebSocket-only environment settings instead of silently ignoring them.

## Checks

Contract tests run without Voicebox, model downloads, or external services. They
use local HTTP/WebSocket fixtures with the installed SDK and real Pipecat pipelines.

```sh
pip install pytest pytest-asyncio ruff
python -m pytest tests -q
ruff check .
ruff format --check .
```

With `VOICEBOX_URL` and `VOICEBOX_CLIENT_KEY` set and the models warm, check the
actual services without a browser or paid LLM:

```sh
python smoke.py                     # TTS -> batch STT and realtime STT
python smoke.py --stt realtime      # or --stt batch
python smoke.py --tts websocket --stt both
```

The smoke synthesizes a sentence with the adapter, passes it through Pipecat's
transcription services, checks word error rate (default maximum 0.25), and verifies
the realtime handshake and commit in a running pipeline. Its printed STT times
measure an uploaded recording, not live conversational latency.

The full WebRTC check launches `bot.py` with echo replies, connects an aiortc caller,
verifies the greeting, speaks an utterance, records the response, and transcribes
it for a content check. It uses an available local port and stops its own worker:

```sh
python e2e.py --stt realtime
python e2e.py --stt batch
python e2e.py --tts websocket --stt realtime
```

Use `--out-dir /path/to/results` to retain WAV recordings and worker logs. These
checks do not need Docker, a microphone, a browser, or an LLM API key.

Local service and WebRTC checks were run on macOS with Python 3.12, Kokoro, and MLX
Whisper base. The echo agent's greeting and reply had zero word errors in both
realtime and batch WebRTC runs; reply latency was 1.057 s and 1.078 s respectively.
These are individual local checks, not capacity or latency guarantees. Browser
microphone UX, production network traversal, real LLM responses, Linux/GPU capacity,
SIP/telephony, and full-call barge-in timing still need deployment-specific checks.

Upstream references: [Pipecat OpenAI STT](https://reference-server.pipecat.ai/en/stable/api/pipecat.services.openai.stt.html),
[TTS implementation at the tested version](https://github.com/pipecat-ai/pipecat/blob/v1.15.0/src/pipecat/services/openai/tts.py),
[SmallWebRTC](https://docs.pipecat.ai/api-reference/server/services/transport/small-webrtc).
