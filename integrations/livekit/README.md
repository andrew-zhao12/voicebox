# LiveKit Agents with Voicebox

A [LiveKit Agents](https://docs.livekit.io/agents/) voice agent that hears and speaks through a
self-hosted Voicebox, plus an end-to-end check that runs it in a real LiveKit room. The full guide
is the docs page [LiveKit Agents](../../docs/content/docs/integrations/livekit.mdx), served at
`/integrations/livekit`; the runbook
[`docs/walkthroughs/07-livekit.md`](../../docs/walkthroughs/07-livekit.md) does everything below
on one machine, with the output of every command.

The agent defaults to LiveKit's stock OpenAI plugin pointed at Voicebox's OpenAI-compatible API.
An optional native adapter sends text incrementally over Voicebox's speech WebSocket; both
transports use the same installed dependencies:

| Piece | Runs on |
|---|---|
| Speech (`openai.TTS`, `response_format="wav"`) | Voicebox `POST /v1/audio/speech` |
| Speech, incremental (`VoiceboxTTS`) | Voicebox `WS /v1/audio/speech/stream` |
| Transcription (`openai.STT`, `use_realtime=True`) | Voicebox `WS /v1/realtime?intent=transcription` |
| Transcription, batch (`use_realtime=False`) | Voicebox `POST /v1/audio/transcriptions`, one call per utterance |
| Voice activity, end of turn, interruptions | LiveKit, in the worker: Silero VAD and the on-device `v1-mini` end-of-turn model |
| Replies | any OpenAI-compatible LLM (OpenAI, Ollama, vLLM, ...) |

## What is here

| File | What it is |
|---|---|
| `agent.py` | The example to copy alongside `voicebox_tts.py`: an `AgentServer` worker whose session is built by `build_session()` |
| `voicebox_tts.py` | Native LiveKit streaming TTS adapter, including `synthesize()`, cancellation and protocol validation |
| `requirements.txt` | `livekit-agents[openai,silero]>=1.8.6,<2` |
| `e2e/run_e2e.py` | The end-to-end check: livekit-server in Docker, the worker, a scripted caller, assertions |
| `e2e/e2e_agent.py` | The worker the check runs: `agent.build_session()` with an echo LLM |
| `e2e/participant.py` | The scripted caller: speaks WAVs into the room and records the agent |
| `e2e/echo.py` | The echo LLM, as a LiveKit LLM and as a tiny OpenAI-compatible chat server; `python e2e/echo.py` serves the latter on its own |
| `e2e/livekit.yaml` | livekit-server config for a server in Docker with clients on the same host |
| `tests/` | Settings, e2e harness and real SDK lifecycle checks against a local mock WebSocket: `pip install pytest pytest-asyncio`, then `python -m pytest tests` |
| `ruff.toml` | Lint settings (Python 3.10, line length 100) |

## Prerequisites

- **Python 3.10 or newer**, and `pip install -r requirements.txt` in a virtualenv. The `<2` bound
  matters: `livekit-agents` 2.0.0b1 is on PyPI and the 1.8 OpenAI plugin does not import on it.
- **A LiveKit server**: LiveKit Cloud, or your own. `docker run --rm -p 7880:7880 -p 7881:7881
  -p 7882:7882/udp livekit/livekit-server:v1.13.10 --dev --bind 0.0.0.0` runs one with the key
  pair `devkey`/`secret`; with Docker Desktop, also mount `e2e/livekit.yaml` as the comment at its
  top shows.
- **A Voicebox server** the worker can reach (the desktop app and `just dev-backend` listen on
  17493, `docker compose up` on 17600), with its models loaded before the first call. For a
  headless one, from the Voicebox repo root:

  ```sh
  VOICEBOX_PRELOAD_MODELS=kokoro,whisper-base backend/venv/bin/python -m backend.main --port 17493
  curl -fsS http://127.0.0.1:17493/health/ready   # 200 once the models are loaded
  ```

- **A client key for the worker**, with limits raised for an agent. On the Voicebox host, from the
  repo root (add `--data-dir` when the server does not use `./data`; the server picks the new key up
  without a restart):

  ```sh
  backend/venv/bin/python -m backend.keys create --id livekit-agent --role client \
    --limit inference=600 --limit requests=1200 \
    --limit max_pending_jobs=16 --limit max_realtime_sessions=8
  ```

  The defaults for a client key are 30 `inference` calls a minute, 4 pending jobs and 2 realtime
  sessions. An agent spends one `inference` per spoken sentence and per batch-transcribed
  utterance, and holds one realtime session per call, so the defaults run out within a minute or
  two of conversation; the limits above suit about eight calls at once (`unlimited` is accepted
  too). The server also caps realtime sessions at 4 overall: raise `VOICEBOX_REALTIME_MAX_SESSIONS`
  on the server for more simultaneous calls.

  With native WebSocket speech, each active TTS stream uses another realtime session in addition
  to transcription. For eight simultaneously speaking calls, allow at least 16 sessions on both
  the key and server, plus headroom for overlapping speech and reconnects. Each phrase flush
  consumes inference, text and pending-job budgets; shorter phrases increase job frequency.

## Run the example

```sh
export LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret
export VOICEBOX_URL=http://127.0.0.1:17493 VOICEBOX_CLIENT_KEY=<the key printed above>
export OPENAI_API_KEY=sk-...                       # or LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
python agent.py dev                                # or: console, start
```

No model at hand? `python e2e/echo.py &` starts the e2e check's echo chat server and prints the
`LLM_BASE_URL` and `LLM_MODEL` to export; the agent then answers "You said: <what it heard>".

Then join a room on that server from any LiveKit client app: LiveKit dispatches the agent to
every new room (set `LIVEKIT_AGENT_NAME` to require an explicit dispatch instead), where it greets
and answers. Without a client app, the e2e check's scripted caller makes one call:
`python e2e/participant.py --room demo --speak line.wav --out-dir demo` says a 16-bit WAV after
the greeting and records the agent's greeting and reply. `python agent.py console` uses this machine's microphone and speakers instead of a
room. LiveKit Agents 1.8 marks this built-in CLI as deprecated, though it still works; its notice
points to `python -m livekit.agents start agent.py`, which runs this file unchanged, and to the
LiveKit CLI (`lk agent dev`).

| Variable | Default | Meaning |
|---|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | | your LiveKit server |
| `VOICEBOX_URL` | `http://127.0.0.1:17493` | Voicebox server root; `/v1` is appended (`docker compose` publishes 17600) |
| `VOICEBOX_CLIENT_KEY` | (required) | the client key. Not `VOICEBOX_API_KEY`, which is the server's own admin key |
| `VOICEBOX_VOICE` | `af_heart` | a preset voice id, or a cloned voice's name or id (`GET /v1/voices`) |
| `VOICEBOX_TTS_MODEL` | `tts-1` | speech model or engine (`GET /v1/models`); `tts-1` uses the voice's own engine. A preset voice works only with its engine, a cloned voice only with a cloning engine |
| `VOICEBOX_TTS_TRANSPORT` | `http` | `http` uses the OpenAI plugin; `websocket` streams text through the native adapter |
| `VOICEBOX_TTS_SPEED` | `1` | speech speed, `0.25` to `4`, on either transport |
| `VOICEBOX_TTS_INSTRUCTIONS` | unset | delivery instructions, up to 500 characters, on either transport; only supporting engines use them |
| `VOICEBOX_PRONUNCIATIONS_FILE` | unset | WebSocket only: path to a JSON object of literal text-to-spoken replacements |
| `VOICEBOX_FIRST_CHUNK_CHARS` | server default | WebSocket only: first backend chunk target, `20` to `1000` |
| `VOICEBOX_MAX_CHUNK_CHARS` | server default | WebSocket only: backend chunk target, `100` to `5000` |
| `VOICEBOX_TEXT_FLUSH_CHARS` | `160` | WebSocket only: maximum client phrase size, `20` to `2000` |
| `VOICEBOX_TTS_TIMEOUT` | `120` | WebSocket only: seconds without speech progress before failure; connect/handshake use the SDK timeout |
| `VOICEBOX_STT_MODEL` | `whisper-1` | Whisper model; `whisper-1` is the server's current one |
| `VOICEBOX_STT_REALTIME` | `1` | `1`: realtime socket; `0`: batch transcription of each VAD-cut utterance |
| `VOICEBOX_LANGUAGE` | `en` | transcription language and, with native WebSocket TTS, speech language |
| `LLM_BASE_URL` | OpenAI | OpenAI-compatible chat API, e.g. `http://127.0.0.1:11434/v1` for Ollama |
| `LLM_MODEL` | `gpt-4.1-mini` | chat model |
| `LLM_API_KEY` | `OPENAI_API_KEY` | key for that API (any value for servers that ignore it) |

Things the example does on purpose:

- **Turn-taking stays local.** It passes `inference.TurnDetector(version="v1-mini")` and
  `interruption={"mode": "vad"}`. Left to its defaults, LiveKit Agents 1.8 uses LiveKit Cloud's
  end-of-turn and interruption models in `console` and `dev` mode. The `v1-mini` weights ship with
  `livekit-agents` (in `livekit-local-inference`), so nothing is downloaded at run time.
- **`wav`, not the plugin's default `mp3`.** Without ffmpeg on the server, `mp3` arrives only
  once the whole sentence is synthesized; `wav` streams chunk by chunk and carries its sample
  rate in the header. (`pcm` works too: `/v1/audio/speech` sends it at 24 kHz, the rate the
  plugin assumes.)
- **It refuses to start without `VOICEBOX_CLIENT_KEY`,** or without an LLM setting, whichever
  way it is launched: `python agent.py`, `python -m livekit.agents start agent.py` or `lk agent`.
  The OpenAI plugin falls back to `OPENAI_API_KEY` when it gets no key, which would send your
  OpenAI key to Voicebox.
- **Barge-in frees Voicebox.** When the user talks over the agent, LiveKit pauses the speech at
  once and drops it when the user's words come back from STT; Voicebox then gives up the speech
  request in flight (an HTTP 499 before its first chunk, a cancelled streaming job after it).
  The chunk being synthesized still finishes, and synthesis runs on while the user is talking.

`python agent.py start` uses LiveKit's production defaults: one idle process per CPU, a health
server on port 8081, and a one-hour drain on shutdown.

## Incremental speech

```sh
export VOICEBOX_TTS_TRANSPORT=websocket
export VOICEBOX_TEXT_FLUSH_CHARS=160
export VOICEBOX_FIRST_CHUNK_CHARS=80
# Optional JSON file, for example {"LiveKit": "Live Kit", "SDK": "ess dee kay"}:
export VOICEBOX_PRONUNCIATIONS_FILE=/absolute/path/pronunciations.json
python agent.py dev
```

Copy `voicebox_tts.py` alongside `agent.py` when deploying the example. The adapter implements
LiveKit `stream()` and `synthesize()` and was checked against SDK 1.8.6 and 1.8.10. It sends
bounded phrases as the LLM supplies text, flushing at sentence endings after 20 characters or
at the configured phrase limit. The SDK's final flush sends any remaining text. Each phrase
is one Voicebox generation job; this is phrase-based synthesis, not generation from every LLM
token. The backend's chunk settings further divide those phrases during synthesis.

One socket belongs to one SDK stream. The adapter awaits each flush's complete audio before
submitting the next, so it holds at most one backend job and keeps pending text below the
server's 10,000-character cap. SDK input is capped at 10,000 characters per stream and received
audio at ten minutes; these bounds also limit SDK buffering when a consumer falls behind.
Audio must be mono PCM16 at 24 kHz with matching flush IDs, sequence numbers and completion
counts. Errors and partial output fail the stream without automatic replay. Interrupting or
closing the SDK stream sends `input_text.cancel` and closes its socket; an engine call already
running finishes before backend resources can be released.

Pronunciation JSON allows at most 100 nonempty entries (keys up to 128 characters, values up
to 256, 8,192 characters total). Unsupported settings fail startup: the installed OpenAI TTS
plugin exposes speed and instructions over HTTP, but no extra request body for pronunciation
or chunk fields. HTTP remains the default; compare both transports with your models and LLM
before choosing one. Smaller phrases can reduce first-audio latency but increase inference
jobs and change pauses or pronunciation at phrase boundaries.

## End-to-end check

`e2e/run_e2e.py` runs the example's session in a real LiveKit room against a running Voicebox.
It needs Docker and the same client key, and no audio tools from the OS: the user's lines are
synthesized by Voicebox itself (voice `am_adam`). It has been run on macOS only; on Windows, which
has no process groups, it stops the worker process but not the worker's job processes.

```sh
cd integrations/livekit/e2e
export VOICEBOX_URL=http://127.0.0.1:17493 VOICEBOX_CLIENT_KEY=<client key>
export VOICEBOX_ADMIN_KEY=<admin key>   # optional: lets the barge-in check read /metrics
python run_e2e.py --stt realtime --sessions 3
python run_e2e.py --stt batch --sessions 3
python run_e2e.py --barge-in --sessions 2
python run_e2e.py --agent example        # the unmodified agent.py, with an echo LLM server
python run_e2e.py --tts-transport websocket --stt realtime
python run_e2e.py --tts-transport websocket --barge-in
```

Each run starts `livekit/livekit-server:v1.13.10` as container `lk-e2e` on ports 7880, 7881 and
7882/udp (`--port-base` moves them), starts the worker, and runs each session in a new room.
An existing container or occupied port makes the run fail without stopping that service;
use `--container` and `--port-base` to give a separate run its own name and ports.
Teardown stops only the container created by this run. Each session checks:

1. the agent greets; Voicebox's batch STT must transcribe the recording to `agent.GREETING`;
2. the caller says "Hi there. Could you tell me what the weather is like today?";
3. the agent's logged transcript must match it, and its reply, from an echo LLM that answers
   "You said: <what it heard>", must transcribe to that sentence: one check covers STT and TTS.
   Every comparison allows a word error rate up to 0.25 (`--max-wer`);
4. the reply must start after the caller stops, within 5 s, and the agent must not talk over them.

The worker runs with `LIVEKIT_DEV_MODE=1`, where LiveKit's defaults would be its cloud turn
detector and adaptive interruptions, so every session must also report the `v1-mini` turn detector
and VAD interruptions. It registers as `e2e-agent-<run id>`, so a worker left behind by an earlier
run never takes this run's calls. Ctrl-C, `kill`, `timeout` and closing the terminal all end a run
through its teardown, which stops the worker and the container.

With `--barge-in` the first reply runs on for minutes, the caller cuts in 1.5 s into it with a
second sentence, and the check requires that the agent goes quiet within 1.5 s (`--max-stop`),
stays quiet while the caller talks, answers the new sentence, and that Voicebox abandoned the
interrupted speech request: a 499 or a cancelled streaming job in `/metrics` (with
`VOICEBOX_ADMIN_KEY`), or the `abandoned before its first chunk` log line (with
`--voicebox-log`); with neither, the agent's own metrics must show a cancelled TTS request. The
first reply is long so that its synthesis is still running when the caller cuts in; on a server
that synthesizes much faster than this check expects, lengthen `ECHO_TAIL` in `run_e2e.py`.

Each run prints per-session timings (end of the caller's speech to the agent's reply, final
transcript latency, TTS time to first byte, barge-in stop time) and exits 0 when every session
passed. WAVs, reports and logs go to `--out-dir` (a temporary directory by default).

Validated again on 2026-11-03 with Python 3.12, LiveKit Agents and its OpenAI plugin 1.8.10,
OpenAI SDK 3.44.0, and livekit-server 1.13.10, using Kokoro, MLX Whisper Base and the echo LLM
on macOS. All six sessions passed: two realtime, two batch, one barge-in and one through the
unmodified example worker. Median reply latency was 1.53 s with realtime transcription and
1.72 s with batch transcription. Barge-in stopped audio in 0.71 s and cancelled the Voicebox
speech request; the example worker replied in 2.53 s. These checks do not cover a real LLM,
SIP calls, Linux or GPU deployment.

Pitfalls the scripts already handle, worth knowing when you adapt them:

- `livekit.yaml` sets `node_ip: 127.0.0.1`; without it, a server in Docker Desktop advertises its
  container address and media only flows by luck.
- With `RoomOptions(participant_identity=...)`, LiveKit Agents 1.8.8 needs `ctx.connect()`
  before `session.start()`.
- After `docker stop`, ports stay in TIME_WAIT for about 30 s; the check looks for listeners,
  not bindable ports, so back-to-back runs work.
- Keep the machine awake (`caffeinate -i` on macOS): a sleep in the middle of a run stalls the
  room, and the caller's clock (`time.monotonic()`) stops while the wall clock does not.
