# 7. LiveKit Agents on this machine

**Goal.** Run a [LiveKit Agents](https://docs.livekit.io/agents/) voice agent
that hears and speaks through a local Voicebox: prove the pipeline with the
end-to-end check in `integrations/livekit/e2e` (live transcription, batch
transcription, a caller talking over the agent), read its latency numbers, then
run the example worker `integrations/livekit/agent.py` against a LiveKit server
you started yourself and put a call through it.

**Time.** About 30 minutes; the commands themselves ran in about ten here.
**Money.** None. **You need.** The `backend/venv` from `just setup-python`,
Docker Desktop running, `uv` (or any Python 3.10+ with `venv` and `pip`), `jq`,
and an internet connection the first time: Kokoro (about 350 MB), Whisper base
(about 290 MB), the LiveKit packages (a 337 MB virtualenv) and the
`livekit/livekit-server` image (133 MB). An LLM is optional: the example needs
an OpenAI-compatible chat API, and the e2e check brings its own stand-in.
**Was run.** Every command below was executed on 2026-10-17 on this Mac (Apple
Silicon M5 Pro, macOS 27.0.1, Docker Desktop 29.8.1 with 18 CPUs and 9.7 GiB of
VM memory); the outputs are from that run. Models and Python wheels were already
cached, and this Mac had no LLM credentials, so the example answers through the
echo chat server the e2e check uses.

Three terminals: terminal 1 for Voicebox (step 2 onwards), terminal 3 for the
example worker (step 11 onwards), terminal 2 for everything else. Commands run
from the repository root. The reference page is
[LiveKit Agents](../content/docs/integrations/livekit.mdx); the worker's own
notes are in [`integrations/livekit/README.md`](../../integrations/livekit/README.md).

## 1. What this machine has, and where the run keeps its files (terminal 2)

```bash
backend/venv/bin/python --version
uv --version
docker info --format '{{.ServerVersion}}, {{.NCPU}} CPUs, {{.MemTotal}} bytes of VM memory'
for p in 17493 17499 7880 7881; do printf 'tcp %s: ' $p; lsof -nP -iTCP:$p -sTCP:LISTEN -Fc | sed -n 's/^c//p' | head -1 | grep . || echo free; done
printf 'udp 7882: '; lsof -nP -iUDP:7882 -Fc | sed -n 's/^c//p' | head -1 | grep . || echo free
printenv OPENAI_API_KEY >/dev/null && echo "OPENAI_API_KEY is set" || echo "OPENAI_API_KEY is not set"
```

**Check:** `Python 3.12.14`, `uv 0.12.11`, `29.8.1, 18 CPUs, 10414411776
bytes of VM memory`, every port `free`, and `OPENAI_API_KEY is not set`. The
LiveKit server needs 7880 (signalling), 7881 (ICE over TCP) and 7882/udp
(media); Voicebox goes on 17499 here because 17493 belongs to the desktop app
whenever it runs (it happened to be closed). A taken port shows the name of the
process holding it instead of `free`.

Now the run's variables. `W` holds the logs, the virtualenv and the e2e
artifacts, `DATA` is a fresh data directory for this Voicebox, and the `env`
file lets every terminal load the same values:

```bash
export PORT=17499 B=http://127.0.0.1:17499
export W=~/voicebox-livekit DATA=~/voicebox-livekit/data
mkdir -p $W $DATA
printf 'export PORT=%s B=%s W=%s DATA=%s\n' $PORT $B $W $DATA > $W/env
```

(This run set `W` and `DATA` to two new, empty scratch directories instead;
nothing else depends on where they are, as long as they are outside the
repository.)

## 2. Voicebox with the models loaded (terminal 1)

The worker calls Voicebox for every sentence it speaks and streams the caller's
audio to it for the whole call, so the models have to be in memory before the
first call: otherwise the first request waits for the model to load (a client
key never downloads one), and the plugin gives up on a sentence whose audio has
not started after 30 s.

```bash
source ~/voicebox-livekit/env
VOICEBOX_PRELOAD_MODELS=kokoro,whisper-base backend/venv/bin/python -m backend.main --port $PORT --data-dir $DATA 2>&1 | tee $W/voicebox.log
```

In terminal 2:

```bash
source ~/voicebox-livekit/env
curl -s $B/health; echo
until [ "$(curl -s -o /dev/null -w '%{http_code}' $B/health/ready)" = 200 ]; do sleep 2; done
curl -s $B/health/ready | jq -c .
```

**Check:** `{"status":"healthy","service":"voicebox"}`, then
`{"ready":true,"draining":false,"stopping":false,"worker":true,"models":{"ready":["kokoro","whisper-base"],"pending":{},"failed":[]},"startup":{"done":[],"pending":{},"failed":[]}}`.
Both models were in the Hugging Face cache, so the loop returned at once.
Terminal 1 shows `Created local API key file …/api_key` (in `$DATA`),
`Backend: MLX`, `Preloading models: kokoro, whisper-base`, `Uvicorn running on
http://127.0.0.1:17499`, `Loading Kokoro-82M on cpu...`, `Kokoro-82M loaded
successfully`, `Loading MLX Whisper model base...` and `MLX Whisper model base
loaded successfully`. Two torch warnings about LSTM dropout and `weight_norm`
come from Kokoro and are harmless.

**If not:** `models.failed` names a model whose load crashed, and terminal 1
has the reason; a readiness that stays `503` with `pending` entries is a first
download in progress.

## 3. The worker's key

Every request the worker makes carries a client key. Give the worker its own,
with limits for an agent: one `inference` per sentence spoken and per batch
utterance, one realtime session per call. The admin key is only for reading
`/metrics` in step 9.

```bash
ADMIN_KEY=$(backend/venv/bin/python -m backend.keys local --data-dir $DATA)
CLIENT_KEY=$(backend/venv/bin/python -m backend.keys create --id livekit-agent --role client --data-dir $DATA \
  --limit inference=600 --limit requests=1200 --limit max_pending_jobs=16 --limit max_realtime_sessions=8)
printf 'export ADMIN_KEY=%s CLIENT_KEY=%s\n' "$ADMIN_KEY" "$CLIENT_KEY" >> $W/env; chmod 600 $W/env
backend/venv/bin/python -m backend.keys list --data-dir $DATA
sleep 1
curl -s -H "Authorization: Bearer $CLIENT_KEY" $B/auth/whoami | jq -c '{key_id,role,limits}'
```

**Check:** stderr says `This is the admin key for this data directory; keep it
private.`, then `Created client key 'livekit-agent' in …/api_keys.json` and
`Store it now; it will not be shown again:` (the keys themselves go into the
variables and the `env` file, never to the screen). `list` prints two rows:

```
local                admin   local  -                          requests=1200, inference=120, uploads_bytes=2147483648, tts_chars=600000, max_pending_jobs=unlimited, max_voices=unlimited, max_realtime_sessions=unlimited
livekit-agent        client  file   2026-10-17T10:29:55+00:00  requests=1200, inference=600, uploads_bytes=268435456, tts_chars=60000, max_pending_jobs=16, max_voices=50, max_realtime_sessions=8
```

and `whoami` answers
`{"key_id":"livekit-agent","role":"client","limits":{"requests":1200,"inference":600,"uploads_bytes":268435456,"tts_chars":60000,"max_pending_jobs":16,"max_voices":50,"max_realtime_sessions":8}}`.
The running server picked the new key up without a restart (it re-reads
`api_keys.json` within a second). The defaults for a client key (30 `inference`
a minute, 4 pending jobs, 2 realtime sessions) run out within a minute or two
of conversation.

## 4. What the plugin will call, by hand

LiveKit's OpenAI plugin fetches `GET /v1/` to warm its connection, speaks
through `POST /v1/audio/speech` and transcribes through
`POST /v1/audio/transcriptions` (batch) or the realtime socket. Check each
with the worker's key before involving LiveKit:

```bash
curl -s -w ' HTTP %{http_code}\n' -H "Authorization: Bearer $CLIENT_KEY" $B/v1
curl -s -w ' HTTP %{http_code}\n' $B/v1
curl -s -H "Authorization: Bearer $CLIENT_KEY" $B/v1/models | jq -c '[.data[].id]'
curl -s -H "Authorization: Bearer $CLIENT_KEY" $B/v1/voices | jq -c '{count: (.data|length), picked: [.data[]|select(.id=="af_heart" or .id=="am_adam")|{id,name,engine}]}'
S="Hello from Voicebox. This sentence checks the speech and transcription a LiveKit agent will use."
for fmt in wav pcm; do curl -s -o $W/check.$fmt -D - -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"kokoro\",\"voice\":\"af_heart\",\"input\":\"$S\",\"response_format\":\"$fmt\"}" $B/v1/audio/speech | grep -iE '^HTTP|x-voicebox-sample-rate'; done
ls -l $W/check.wav $W/check.pcm | awk '{print $5, $NF}'
curl -s -H "Authorization: Bearer $CLIENT_KEY" -F file=@$W/check.wav -F model=whisper-1 -F language=en $B/v1/audio/transcriptions | jq -c .
```

**Check:**

```
{"object":"service","name":"voicebox","version":"0.5.0"} HTTP 200
{"error":{"message":"Authentication required","type":"authentication_error","param":null,"code":"invalid_api_key"}} HTTP 401
["qwen-tts-1.7B","qwen-tts-0.6B","qwen-custom-voice-1.7B","qwen-custom-voice-0.6B","luxtts","chatterbox-tts","chatterbox-turbo","tada-1b","tada-3b-ml","kokoro","whisper-base","whisper-small","whisper-medium","whisper-large","whisper-turbo"]
{"count":63,"picked":[{"id":"af_heart","name":"Heart","engine":"kokoro"},{"id":"am_adam","name":"Adam","engine":"kokoro"}]}
HTTP/1.1 200 OK
x-voicebox-sample-rate: 24000
HTTP/1.1 200 OK
x-voicebox-sample-rate: 24000
313200 …/check.pcm
313244 …/check.wav
{"text":"Hello from VoiceBox. This sentence checks the speech and transcription a live kit agent will use."}
```

`af_heart` is the example's voice and `am_adam` the e2e caller's. `pcm` is
always 24 kHz on `/v1`, the rate the plugin assumes for raw audio; `wav` (what
the example asks for) carries the engine's rate in its header, which is also
24 kHz for Kokoro, so the two files differ only by the 44-byte header (6.5 s of
audio). `whisper-1` means "the Whisper size the server has loaded", here base.

**If not:** a `403` on `GET /v1` is a server from before this integration;
`404` `voice_not_found` is a voice id typo.

## 5. The integration's virtualenv

The worker runs in its own Python environment, not in `backend/venv`:

```bash
uv venv --python 3.12 $W/venv
time uv pip install --python $W/venv/bin/python -r integrations/livekit/requirements.txt
uv pip list --python $W/venv/bin/python 2>/dev/null | grep -E '^(livekit[a-z-]*|openai|onnxruntime) '
```

**Check:** `Using CPython 3.12.14`, `Resolved 76 packages`, `Installed 76
packages` (in 0.1 s here, from uv's cache), then:

```
livekit                                  1.1.20
livekit-agents                           1.8.8
livekit-api                              1.2.2
livekit-blingfire                        1.2.0
livekit-local-inference                  0.2.7
livekit-plugins-openai                   1.8.8
livekit-plugins-silero                   1.8.8
livekit-protocol                         1.1.27
onnxruntime                              1.31.0
openai                                   3.33.0
```

`requirements.txt` asks for `livekit-agents[openai,silero]>=1.8.6,<2`; the upper
bound keeps installers off the 2.0 beta, where the 1.8 OpenAI plugin does not
import. `livekit-local-inference` carries the on-device `v1-mini` end-of-turn
model the example pins, so nothing is downloaded at run time. Without `uv`:
`python3 -m venv $W/venv && $W/venv/bin/pip install -r
integrations/livekit/requirements.txt` with any Python from 3.10 on.

## 6. The end-to-end check: live transcription

`run_e2e.py` does a whole run by itself: it has Voicebox synthesize the
caller's two lines in `am_adam`, starts `livekit/livekit-server:v1.13.10` in
Docker as container `lk-e2e` with `e2e/livekit.yaml` (ports 7880, 7881 and
7882/udp, so they must be free), starts a worker that runs the example's
`build_session()` with an echo LLM (it answers "You said: <what it heard>"),
and puts each session in a new room with a scripted caller. Every comparison is
a word error rate against the expected text, through Voicebox's own batch
transcription of what the caller recorded.

```bash
export VOICEBOX_URL=$B VOICEBOX_CLIENT_KEY=$CLIENT_KEY VOICEBOX_ADMIN_KEY=$ADMIN_KEY
caffeinate -i $W/venv/bin/python integrations/livekit/e2e/run_e2e.py --stt realtime --sessions 3 --out-dir $W/runs/realtime --voicebox-log $W/voicebox.log
```

**Check:**

```
[e2e 05:30:54] Voicebox ready; user lines in am_adam: 3.42 s and 3.95 s
[e2e 05:30:55] livekit-server livekit/livekit-server:v1.13.10 ready in 0.35 s (ws://127.0.0.1:7880)
[e2e 05:31:02] worker registered in 7.21 s: e2e_agent.py start
[e2e 05:31:21] session 1: PASS {"user_end_to_reply_s": 1.515, "stt_final_s": 0.528, "transcription_delay_s": 0.327, "end_of_turn_delay_s": 0.577, "tts_ttfb_s": 0.362, "llm_ttft_s": 0.012}
[e2e 05:31:39] session 2: PASS {"user_end_to_reply_s": 1.519, "stt_final_s": 0.511, ...}
[e2e 05:31:57] session 3: PASS {"user_end_to_reply_s": 1.512, "stt_final_s": 0.535, ...}
[e2e 05:31:58] torn down: worker and livekit-server stopped

3 session(s), stt=realtime agent=e2e; timings in seconds (min / median / max):
  end_of_turn_delay_s      0.577 / 0.577 / 0.577
  llm_ttft_s               0.012 / 0.012 / 0.012
  stt_final_s              0.511 / 0.528 / 0.535
  transcription_delay_s    0.327 / 0.329 / 0.331
  tts_ttfb_s               0.359 / 0.362 / 0.381
  user_end_to_reply_s      1.512 / 1.515 / 1.519
  session 1: Voicebox {'speech_200': 3.0, 'speech_499': 0.0, 'transcriptions': 0.0, 'stream_cancelled': 0.0, 'stream_abandoned': 0.0}; agent TTS cancelled: 0
  ...
[e2e 05:31:58] RESULT PASS: 3/3 sessions passed
```

The command exits 0. Each session passed four checks: the greeting the caller
recorded transcribes to the example's `GREETING`; the agent's logged transcript
matches "Hi there. Could you tell me what the weather is like today?"; its reply
transcribes to "You said: …" that sentence (one check of Voicebox's STT and TTS
together); and the reply started after the caller stopped, within 5 s, without
talking over them. The worker runs with `LIVEKIT_DEV_MODE=1`, where LiveKit's
defaults would be its cloud turn detector and adaptive interruptions, and it
logged the `turn-detector-v1-mini` detector and VAD interruptions, as the check
requires. Each session folder holds the WAVs and what was heard:

```bash
jq -c '.evidence | {greeting: ."greeting.wav_transcript", heard: .agent_user_turns, reply: ."reply.wav_transcript", turn_detector, interruption}' $W/runs/realtime/session-1/summary.json
```

gives `{"greeting":"Hello, I'm a voice assistant and I'm listening. How can I
help you today?","heard":["Hi there, could you tell me what the weather is like
today?"],"reply":"You said, hi there, could you tell me what the weather is like
today?","turn_detector":"turn-detector-v1-mini","interruption":"vad"}`. `afplay` plays
`$W/runs/realtime/session-1/reply.wav` if you want to hear it.

**If not:** `ports [...] are in use` means another LiveKit server holds them
(step 10's, if you ran it first; ports left in `TIME_WAIT` by a previous run do
not count, the driver looks for listeners); a session failing on `greeting` or
`reply` WER prints what was heard; `worker.log` in the run folder has the
worker's side.

## 7. Batch transcription

The same check with `use_realtime=False`: LiveKit's Silero VAD cuts each
utterance and the plugin posts it to `/v1/audio/transcriptions`.

```bash
caffeinate -i $W/venv/bin/python integrations/livekit/e2e/run_e2e.py --stt batch --sessions 3 --out-dir $W/runs/batch --voicebox-log $W/voicebox.log
```

**Check:** `RESULT PASS: 3/3 sessions passed`, with

```
  end_of_turn_delay_s      0.640 / 0.643 / 0.649
  stt_final_s              0.798 / 0.846 / 0.847
  transcription_delay_s    0.638 / 0.641 / 0.647
  tts_ttfb_s               0.353 / 0.358 / 0.359
  user_end_to_reply_s      1.679 / 1.729 / 1.730
  session 1: Voicebox {'speech_200': 3.0, 'speech_499': 0.0, 'transcriptions': 1.0, ...}
```

`transcriptions: 1.0` per session is the agent's upload of the caller's
utterance (the realtime run shows `0.0`: its audio went over the socket). The
reply comes about 0.2 s later than with live transcription, because the
transcript can only start once Silero has heard the caller stop.

## 8. Barge-in

The caller cuts in 1.5 s into a reply that would otherwise run for minutes. The
check requires the agent to fall silent within 1.5 s, stay silent while the
caller talks, answer the new sentence, and Voicebox to give up the speech
request that was in flight (a `499` or a cancelled streaming job in `/metrics`,
or the `abandoned before its first chunk` line in its log).

```bash
caffeinate -i $W/venv/bin/python integrations/livekit/e2e/run_e2e.py --barge-in --sessions 2 --out-dir $W/runs/barge-realtime --voicebox-log $W/voicebox.log
caffeinate -i $W/venv/bin/python integrations/livekit/e2e/run_e2e.py --barge-in --stt batch --sessions 2 --out-dir $W/runs/barge-batch --voicebox-log $W/voicebox.log
```

**Check:** both end `RESULT PASS: 2/2 sessions passed`. The realtime run:

```
  barge_end_to_reply_s     1.509 / 1.512 / 1.515
  barge_stop_s             0.690 / 0.692 / 0.693
  user_end_to_reply_s      1.527 / 1.529 / 1.531
  session 1: Voicebox {'speech_200': 10.0, 'speech_499': 0.0, 'transcriptions': 0.0, 'stream_cancelled': 1.0, 'stream_abandoned': 0.0}; agent TTS cancelled: 1
  session 2: Voicebox {'speech_200': 10.0, 'speech_499': 0.0, 'transcriptions': 0.0, 'stream_cancelled': 1.0, 'stream_abandoned': 0.0}; agent TTS cancelled: 1
```

and the batch run `barge_stop_s 0.741 / 0.744 / 0.748`, `barge_end_to_reply_s
1.509 / 1.514 / 1.519`. `stream_cancelled: 1.0` is Voicebox stopping the
interrupted sentence after the chunk it was synthesizing; a `speech_499` instead
means the cut came before that request's first chunk, just after the request
for the next sentence went out. The barge-in reply transcribes to `You said
sorry to interrupt. What time is it in Tokyo right now?` and the first reply is
marked `interrupted: true` in the evidence. The first reply is long so that its
synthesis is still running when the caller cuts in; on a server much faster
than this Mac's CPU Kokoro, lengthen `ECHO_TAIL` in `run_e2e.py`.

## 9. Reading the timings

The driver prints these per run; across runs they are easier to compare side
by side, and Voicebox's own counters say what it did:

```bash
for r in realtime batch barge-realtime barge-batch; do jq -r --arg r $r '
  def med: sort | (if length % 2 == 1 then .[length/2|floor] else (.[length/2-1] + .[length/2]) / 2 end) * 1000 | round / 1000;
  "\($r): reply \(map(.timings.user_end_to_reply_s)|med) s, final transcript \(map(.timings.stt_final_s)|med) s, first TTS byte \(map(.timings.tts_ttfb_s)|med) s"
  + (if .[0].timings.barge_stop_s then ", quiet \(map(.timings.barge_stop_s)|med) s into the barge-in" else "" end)' $W/runs/$r/summary.json; done
curl -s -H "Authorization: Bearer $ADMIN_KEY" $B/metrics | grep -E '^voicebox_(generations_total|stream_first_chunk_seconds_(sum|count)|queue_wait_seconds_(sum|count)|realtime_utterances_total)|^voicebox_http_requests_total\{method="POST"'
```

**Check:**

```
realtime: reply 1.515 s, final transcript 0.528 s, first TTS byte 0.362 s
batch: reply 1.729 s, final transcript 0.846 s, first TTS byte 0.358 s
barge-realtime: reply 1.529 s, final transcript 0.533 s, first TTS byte 0.367 s, quiet 0.692 s into the barge-in
barge-batch: reply 1.75 s, final transcript 0.843 s, first TTS byte 0.365 s, quiet 0.744 s into the barge-in
voicebox_http_requests_total{method="POST",route="/v1/audio/speech",status="200"} 68.0
voicebox_http_requests_total{method="POST",route="/v1/audio/transcriptions",status="200"} 28.0
voicebox_queue_wait_seconds_count 70.0
voicebox_queue_wait_seconds_sum 2.8370020806323737
voicebox_generations_total{engine="kokoro",kind="stream",status="completed"} 64.0
voicebox_generations_total{engine="kokoro",kind="stream",status="cancelled"} 4.0
voicebox_stream_first_chunk_seconds_count{engine="kokoro"} 68.0
voicebox_stream_first_chunk_seconds_sum{engine="kokoro"} 23.925287789083086
voicebox_realtime_utterances_total 7.0
```

What the numbers are, all measured from the moment the caller's speech ended
(as the caller's own recording saw it):

- `user_end_to_reply_s`: to the first audio of the agent's reply arriving at
  the caller. This is the number a person on the call feels.
- `stt_final_s`: to the final transcript in the worker. Besides the audio's trip
  through LiveKit to the worker, with live transcription that is Voicebox's
  350 ms of silence plus one Whisper pass on the utterance; in batch mode,
  Silero's silence plus one upload and one Whisper pass.
- `end_of_turn_delay_s` and `transcription_delay_s`: LiveKit's own metrics for
  the same turn (end of speech to the turn being committed, and to the
  transcript).
- `tts_ttfb_s`: from the plugin's speech request to Voicebox's first audio
  byte, for the first sentence of the reply. Voicebox's side of it is
  `voicebox_stream_first_chunk_seconds`: 23.93 s over 68 requests, 0.35 s on
  average, with 0.04 s of queue wait on average (`queue_wait`, which also
  counts the two preloads).
- `llm_ttft_s`: the echo LLM's 12 ms. A real model adds its time to the first
  sentence to every reply number here, less the part that overlaps the turn
  confirmation: LiveKit starts the LLM as soon as the final transcript is in,
  and sends the speech request once the turn is committed.
- `barge_stop_s`: from the caller starting to talk over the agent to the
  agent's audio stopping at the caller.

The counters add up: 68 speech requests are 3 per plain session (two greeting
sentences and the reply), 10 per barge-in session, the caller's two lines per
run and step 4's two; 28 transcriptions are the driver's two per session, the
batch agent's 7 utterances and step 4's one; 7 realtime utterances are one per
realtime session plus two per realtime barge-in session; 4 cancelled streams
are the 4 barge-ins. A real call adds network time between the worker and
Voicebox, and an engine slower than Kokoro on this CPU adds to `tts_ttfb_s`.

## 10. A LiveKit server by hand

The example worker needs a LiveKit server that stays up. This is the e2e
check's own configuration: `--dev` supplies the key pair `devkey`/`secret`,
`--bind 0.0.0.0` is needed inside a container, and `livekit.yaml` sets
`node_ip: 127.0.0.1` so the server advertises the ports Docker publishes rather
than its container address.

```bash
docker run --rm -d --name livekit -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  -v "$PWD/integrations/livekit/e2e/livekit.yaml:/etc/livekit.yaml:ro" livekit/livekit-server:v1.13.10 \
  --dev --bind 0.0.0.0 --config /etc/livekit.yaml | cut -c1-12
until curl -s -o /dev/null http://127.0.0.1:7880/; do sleep 0.5; done; curl -s http://127.0.0.1:7880/; echo
docker logs livekit 2>&1 | grep -E 'placeholder keys|explicit node IP|starting LiveKit server'
```

**Check:** a container id (`c3c4378dbb9b` here), `OK`, and three log lines
(timestamps trimmed):

```
INFO  livekit  server/main.go:214  no keys provided, using placeholder keys  {"API Key": "devkey", "API Secret": "secret"}
INFO  livekit  rtcconfig/webrtc_config.go:112  using explicit node IP for NAT1To1Ips  {"ip": {"V4":"127.0.0.1","V6":""}, "advertiseInternalIP": false}
INFO  livekit  service/server.go:309  starting LiveKit server  {"portHttp": 7880, ..., "nodeIP": "127.0.0.1", "version": "1.13.10", "bindAddresses": ["0.0.0.0"], "rtc.portTCP": 7881, "rtc.portUDP": {"Start":7882,"End":0}}
```

**If not:** without `node_ip` the server advertises `172.17.x.x`, which the Mac
cannot reach, and media flows only when Docker's NAT happens to find another
path; on LiveKit Cloud or a real server, skip this step and use its URL and
keys in step 11.

## 11. The example worker (terminal 3)

First without an LLM, to see the configuration check, then with the echo chat
server from the e2e check standing in for one:

```bash
source ~/voicebox-livekit/env
export LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret
export VOICEBOX_URL=$B VOICEBOX_CLIENT_KEY=$CLIENT_KEY
$W/venv/bin/python integrations/livekit/agent.py dev
$W/venv/bin/python integrations/livekit/e2e/echo.py &
export LLM_BASE_URL=http://127.0.0.1:55902/v1 LLM_MODEL=echo      # the two values echo.py printed
$W/venv/bin/python integrations/livekit/agent.py dev
```

With a real model, skip `echo.py` and export `OPENAI_API_KEY` instead of the
two `LLM_` values (the example defaults to `gpt-4.1-mini`), or
`LLM_BASE_URL=http://127.0.0.1:11434/v1 LLM_MODEL=llama3.1` for a local Ollama.
This Mac had neither, which is why the echo server is used.

**Check:** the first `agent.py dev` exits 1 with `Set LLM_API_KEY (or
OPENAI_API_KEY), or LLM_BASE_URL for a local server.` (it also refuses to start
without `VOICEBOX_CLIENT_KEY`, because the plugin would otherwise send
`OPENAI_API_KEY` to Voicebox). `echo.py` prints `echo chat API:
LLM_BASE_URL=http://127.0.0.1:55902/v1 LLM_MODEL=echo` (the port is any free
one). The second `agent.py dev` prints two deprecation notices (`dev mode is
deprecated ... use lk agent dev instead` and `in-process auto-reload has been
removed from the Python CLI`; both modes still work in 1.8), then these lines,
with a PyJWT `InsecureKeyLengthWarning` before the last one (the dev secret
`secret` is only 6 bytes):

```
INFO livekit.agents - starting worker {"version": "1.8.8", "rtc-version": "1.1.20"}
INFO livekit.agents - plugin registered {"plugin": "livekit.plugins.openai", "version": "1.8.8"}
INFO livekit.agents - plugin registered {"plugin": "livekit.plugins.silero", "version": "1.8.8"}
INFO livekit.agents - HTTP server listening on :55904
INFO livekit.agents - registered worker {"agent_name": "", "deployment": "", "id": "AW_DKJS5YnRzPPr", "url": "ws://127.0.0.1:7880", "region": "", "protocol": 17}
```

`agent_name: ""` means automatic dispatch: LiveKit sends this worker to every
new room (set `LIVEKIT_AGENT_NAME` to require an explicit dispatch, as the e2e
check does). `dev` uses a random port for the worker's health server; `start`
would take 8081 and keep one idle process per CPU.

## 12. A call into the example (terminal 2)

Any LiveKit client can join a room on this server now. The e2e check's scripted
caller is the quickest one here: it joins room `demo`, waits for the greeting,
says the line the e2e run synthesized in step 6, records the agent and leaves.

```bash
$W/venv/bin/python integrations/livekit/e2e/participant.py --room demo --speak $W/runs/realtime/user.wav --out-dir $W/runs/demo | jq -c '{phases, greeting_s, reply_s}'
for f in greeting reply; do printf '%s: ' $f; curl -s -H "Authorization: Bearer $CLIENT_KEY" -F file=@$W/runs/demo/$f.wav -F model=whisper-1 -F language=en $B/v1/audio/transcriptions | jq -r .text; done
```

**Check:**

```
{"phases":{"greeting_start":2.209,"greeting_end":6.458,"user_start":7.972,"user_end":10.392,"reply_start":11.908,"reply_end":15.078},"greeting_s":6.48,"reply_s":6.69}
greeting: Hello, I'm a voice assistant and I'm listening. How can I help you today?
reply: You said, hi there, could you tell me what the weather is like today?
```

The reply started 1.52 s after the caller stopped (`reply_start - user_end`),
in line with step 6. Before the JSON, stderr carries a PyJWT
`InsecureKeyLengthWarning` and six `Exception ignored in: <function
FfiHandle.__del__ ...> ... AssertionError` tracebacks: livekit-rtc 1.1.20
releasing handles while the script exits, harmless (the e2e driver keeps them
in each session's `participant.log`). Terminal 3 shows the call (trimmed):

```
INFO livekit.agents - received job request {"job_id": "AJ_8aLRPxTxTbAK", ..., "room": "demo", "agent_name": "", ...}
WARNING livekit.agents - no warmed process available for job, waiting for one to be created
INFO livekit.agents - process initialized {"pid": 34624, "elapsed_time": 0.86}
INFO livekit.agents - RoomIO linked to participant {"lk.pii.participant_identity": "e2e-user", "lk.pii.room_name": "demo", ...}
DEBUG livekit.agents - conversation_item_added {"role": "assistant", "lk.pii.text": "Hello! I'm a voice assistant, and I'm listening. How can I help you today?", ...}
DEBUG voicebox-agent - heard: Hi there, could you tell me what the weather is like today? {...}
DEBUG livekit.agents - eot prediction {"probability": 0.7790672779083252, "unlikely_threshold": 0.36, ..., "trigger": "vad", ...}
DEBUG livekit.agents - conversation_item_added {"role": "assistant", "lk.pii.text": "You said: Hi there, could you tell me what the weather is like today?", ...}
INFO livekit.agents - closing agent session due to participant disconnect (disable via `RoomInputOptions.close_on_disconnect=False`) {...}
```

`heard:` is the example's own log line for each final transcript (`dev` logs at
debug level). The `eot prediction` line is the `v1-mini` turn detector running
inside the worker: nothing in this call went to LiveKit Cloud. The warning is
`dev` keeping no idle process; the job's process started in 0.86 s. The inline
agent on the docs page behaves the same way: run on 2026-10-17 against this
server with `OPENAI_API_KEY=unused OPENAI_BASE_URL=<the echo URL>`, it greeted
through `generate_reply` and answered the caller with the same sentence.

## 13. Clean up

In terminal 3, Ctrl-C stops the worker (it logs `shutting down worker` and
exits at once with no call in progress); `kill %1` stops `echo.py`. In terminal
1, Ctrl-C stops Voicebox. Then, in terminal 2:

```bash
docker stop livekit
docker ps -a --filter name=livekit --filter name=lk-e2e --format '{{.Names}} {{.Status}}'
for p in $PORT 7880 7881; do lsof -nP -iTCP:$p -sTCP:LISTEN >/dev/null && echo "$p busy" || echo "$p free"; done; lsof -nP -iUDP:7882 >/dev/null && echo "7882/udp busy" || echo "7882/udp free"
rm -rf $W
```

**Check:** terminal 1 ends `Draining (signal SIGINT): refusing new jobs,
finishing the ones in flight`, `Generation queue drained`, `MLX Whisper model
unloaded`, `Application shutdown complete.`; `docker stop` prints `livekit`, the
`docker ps` line prints nothing (the e2e runs removed their `lk-e2e` containers
themselves), and every port is `free`. `rm -rf $W` deletes the data directory,
the keys, the virtualenv and the recordings; keep it if you want to listen to
the WAVs or rerun the check.

## Done when

- [ ] `/health/ready` listed `kokoro` and `whisper-base`, `GET /v1` answered the worker's key with `200` and no key with `401`.
- [ ] The e2e check passed in all four runs: 3/3 live, 3/3 batch, 2/2 barge-in on each transcription path, with Voicebox counting a cancelled stream for every barge-in.
- [ ] The example registered with your LiveKit server, joined a room, greeted, logged `heard:` with the caller's sentence and answered it.
- [ ] Nothing listens on 7880, 7881, 7882/udp or your Voicebox port, and no `livekit` or `lk-e2e` container is left.

**Not covered here.** A real LLM (this Mac had no credentials; the replies came
from the echo server, so a model's time to first sentence is missing from every
number above); `agent.py console`, which needs a microphone and speakers; the
LiveKit CLI's `lk agent dev`; LiveKit Cloud and phone calls through LiveKit SIP;
the Voicebox Docker image (the [docs page's quick start](../content/docs/integrations/livekit.mdx)
uses it, with `docker compose exec -u voicebox` for the key); GPU engines
(Qwen, Chatterbox, LuxTTS), whose time to first audio decides how fast a reply
starts on a server; and several calls at once, which share Voicebox's serial
queue and its one Whisper slot (see the docs page's Sizing section).
