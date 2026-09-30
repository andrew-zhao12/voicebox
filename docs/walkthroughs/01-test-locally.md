# 1. Test locally

**Goal.** Exercise everything the server does on this Mac, from the test
suite to a two-replica fleet behind a load balancer, and leave behind the
two inputs the cloud documents mount: `./seed` (the voice catalog) and
`./secrets/api_keys.json` (the keys file).

**Time.** About 2 hours, less if the Docker image is already built.
**Money.** None. **You need.** The `backend/venv` from `just setup-python`
(present), Docker Desktop running, about 10 GB of disk, an internet
connection for the model downloads (Kokoro and Whisper base are already in
this Mac's Hugging Face cache). **Was run.** Every command below was
executed on 2026-09-29 on this Mac (M5 Pro, Docker Desktop with 7.75 GiB
of VM memory); the outputs are from that run.

Two terminals: one for the server (steps 4 and 13 onwards), one for
everything else. Commands run from the repository root.

## 1. What this machine has

```bash
backend/venv/bin/python --version
docker info --format '{{.ServerVersion}}, {{.NCPU}} CPUs, {{.MemTotal}} bytes of VM memory'
for t in just bun gcloud az helm ffmpeg uv jq aws kubectl; do printf '%s: ' "$t"; command -v "$t" >/dev/null && echo yes || echo no; done
```

**Check:** `Python 3.12.14`; Docker `29.7.2, 18 CPUs, 8317267968 bytes`.
On this Mac `just`, `bun`, `gcloud`, `az`, `helm` and `ffmpeg` are `no`,
`uv`, `jq`, `aws`, `kubectl` are `yes`. Consequences: every `just` recipe
is spelled out below; the system `python3` is 3.14 without the project's
packages, so scripts run as `backend/venv/bin/python scripts/...`; without
`ffmpeg` the `aac` output format is unavailable (the others are); the venv
has no `pip` (it was made with `uv`: `uv pip install --python
backend/venv/bin/python PKG` installs into it). 7.75 GiB of Docker memory is
enough for the fleet with Kokoro and Whisper base; raise it in Docker
Desktop → Settings → Resources before trying larger models in containers.

## 2. The test suite and the linter

```bash
backend/venv/bin/python -m pytest backend/tests -q -p no:cacheprovider
backend/venv/bin/ruff check $(grep -v '^#' backend/ruff-clean-paths.txt)
```

**Check:** `442 passed, 9 skipped, 8 warnings in 10.76s` (the count grows
with the code; zero failures is the point) and `All checks passed!`. The
skips are the two `ffmpeg` tests in `test_encode.py`, the Windows and
Python 3.13 cases, and the ROCm build tests. The warnings include a
`PytestReturnNotNoneWarning` from `test_generation_download.py`, a script
that expects a live server and returns `False` without one.

**If not:** a failure in `test_mlx_smoke.py` means the MLX packages
broke; anything else is a real regression, read the assertion.

## 3. The models

```bash
backend/venv/bin/python -m backend.preload --list
backend/venv/bin/python -m backend.preload kokoro whisper-base
```

**Check:** the list prints one line per model (`kokoro  Kokoro 82M
(hexgrad/Kokoro-82M)`, `whisper-base ...`); the preload ends with `ready:
kokoro` and `ready: whisper-base` in a few seconds because both are cached
(a first download is about 350 MB and 290 MB). Note that on this Mac the
Qwen TTS entries point at `mlx-community/...` repositories: a cache filled
here is not the cache a CUDA server needs, which is why the cloud documents
preload on the platform.

## 4. A scratch server (terminal 1)

Port 17493 is the project's port, but the desktop app owns it while it
runs (this Mac had `voicebox-server` from `/Volumes/Voicebox/Voicebox.app`
listening there). Check, and pick 17494 if it is taken:

```bash
lsof -nP -iTCP:17493 -sTCP:LISTEN
export PORT=17494          # or 17493 when it is free
VOICEBOX_PRELOAD_MODELS=kokoro VOICEBOX_MCP_STATELESS=1 VOICEBOX_SHUTDOWN_DELAY_S=10 backend/venv/bin/uvicorn backend.main:app --port $PORT
```

(`python -m backend.main` defaults to port 8000; the uvicorn form is what
`just dev-backend` runs, minus `--reload`.) In terminal 2:

```bash
export PORT=17494 B=http://127.0.0.1:$PORT
curl -s $B/health; echo
until [ "$(curl -s -o /dev/null -w '%{http_code}' $B/health/ready)" = 200 ]; do sleep 2; done
curl -s $B/health/ready | jq -c .
```

**Check:** `{"status":"healthy","service":"voicebox"}` at once (the
anonymous body is minimal), then
`{"ready":true,"draining":false,"stopping":false,"worker":true,"models":{"ready":["kokoro"],"pending":{},"failed":[]},"startup":{"done":[],"pending":{},"failed":[]}}`
after a couple of seconds. Terminal 1 shows `Created local API key file
.../data/api_key`, `Loading Kokoro-82M on cpu...` and `Kokoro-82M loaded
successfully`.

**If not:** `address already in use` is the port; `models.failed:
["kokoro"]` means the load crashed and the reason is in terminal 1.

## 5. Keys

```bash
ADMIN_KEY=$(backend/venv/bin/python -m backend.keys local)
CLIENT_KEY=$(backend/venv/bin/python -m backend.keys create --id myapp --role client)
LOAD_KEY=$(backend/venv/bin/python -m backend.keys create --id loadtest --role client \
  --limit inference=unlimited --limit requests=unlimited --limit tts_chars=unlimited --limit max_pending_jobs=unlimited)
backend/venv/bin/python -m backend.keys list
sleep 1
curl -s -H "Authorization: Bearer $ADMIN_KEY" $B/auth/whoami | jq -c '{key_id,role,via}'
curl -s -H "Authorization: Bearer $CLIENT_KEY" $B/auth/whoami | jq -c '{key_id,role,via}'
curl -s -H "Authorization: Bearer $LOAD_KEY" $B/auth/whoami | jq -c .limits
curl -s -w ' HTTP %{http_code}\n' $B/profiles
curl -s -w ' HTTP %{http_code}\n' -H "Authorization: Bearer $CLIENT_KEY" $B/history
```

**Check:** `create` prints the key on stdout and `Store it now; it will
not be shown again` on stderr; `list` shows three rows (`local admin
local ...`, `myapp client file ... inference=30, ... max_pending_jobs=4,
... max_realtime_sessions=2`, `loadtest client file ...
inference=unlimited, ...`); then `{"key_id":"local","role":"admin","via":"header"}`,
`{"key_id":"myapp","role":"client","via":"header"}`, the load key's limits
with `null` for the unlimited ones, `{"detail":"Authentication required"}
HTTP 401` and `{"detail":"Admin key required"} HTTP 403`. The server
re-reads `data/api_keys.json` within a second of a change; the one-second
sleep covers it.

## 6. A catalog: one preset voice, one cloned voice, exported as the seed

The preset proves the seed path; the cloned voice is what the cloning
engines need in [document 6](06-test-on-the-cloud.md). Its sample is a
Kokoro rendering of a sentence, which is fine for a plumbing test.

```bash
T="The quick brown fox jumps over the lazy dog while the river runs quietly past the old stone bridge, and the evening light settles over the valley."
curl -s -H "Authorization: Bearer $ADMIN_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"Narrator","language":"en","voice_type":"preset","preset_engine":"kokoro","preset_voice_id":"af_heart"}' $B/profiles | jq -c '{id,name,voice_type,default_engine}'
curl -s -o ref.wav -D - -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"kokoro\",\"voice\":\"af_heart\",\"input\":\"$T\",\"response_format\":\"wav\"}" $B/v1/audio/speech | grep -iE '^HTTP|x-voicebox'
ls -l ref.wav | awk '{print $5" bytes"}'; afplay ref.wav
PID=$(curl -s -H "Authorization: Bearer $ADMIN_KEY" -H 'Content-Type: application/json' -d '{"name":"CloneTest","language":"en"}' $B/profiles | jq -r .id)
curl -s -H "Authorization: Bearer $ADMIN_KEY" -F file=@ref.wav -F "reference_text=$T" $B/profiles/$PID/samples | jq -c '{id: (.id|.[0:8]), profile_id: (.profile_id|.[0:8])}'
backend/venv/bin/python -m backend.voices export ./seed
ls -l seed
```

**Check:** the profile with `"voice_type":"preset"` and
`"default_engine":"kokoro"`; `HTTP/1.1 200 OK` with `x-voicebox-engine:
kokoro`, `x-voicebox-voice: kokoro:af_heart`, `x-voicebox-sample-rate:
24000`; `438044 bytes` (9.1 s of audio) that you can hear; the sample id;
then `CloneTest: seed/CloneTest.voicebox.zip (331885 bytes)` and
`Narrator: seed/Narrator.voicebox.zip (445 bytes)`. Keep `ref.wav` for
the next steps.

## 7. Speech: formats, speed, errors, the native stream

```bash
curl -s -o /dev/null -w 'speed 2: %{size_download} bytes\n' -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"kokoro\",\"voice\":\"af_heart\",\"input\":\"$T\",\"response_format\":\"wav\",\"speed\":2.0}" $B/v1/audio/speech
curl -s -o /dev/null -w 'mp3: HTTP %{http_code} %{size_download} bytes %{content_type}\n' -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","voice":"af_heart","input":"A short mp3 check.","response_format":"mp3"}' $B/v1/audio/speech
curl -s -w ' HTTP %{http_code}\n' -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","voice":"af_heart","input":"x","response_format":"aac"}' $B/v1/audio/speech
curl -s -w ' HTTP %{http_code}\n' -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' -d '{"model":"kokoro","voice":"af_heart","input":"x","speed":5}' $B/v1/audio/speech
curl -s -w ' HTTP %{http_code}\n' -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' -d '{"model":"kokoro","voice":"nobody","input":"x"}' $B/v1/audio/speech
NID=$(curl -s -H "Authorization: Bearer $ADMIN_KEY" $B/profiles | jq -r '.[]|select(.name=="Narrator").id')
curl -s -o stream.wav -D - -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' \
  -d "{\"profile_id\":\"$NID\",\"text\":\"Streaming from the native endpoint. Each sentence arrives as soon as it is synthesized.\",\"engine\":\"kokoro\"}" $B/generate/stream | grep -iE '^HTTP|x-voicebox'
curl -s -H "Authorization: Bearer $CLIENT_KEY" $B/v1/voices | jq -c '{count: (.data|length), profiles: [.data[]|select(.kind=="profile")|{name,engine,owner,shared}]}'
```

**Check:** `speed 2: 243644 bytes` (Kokoro speaks faster natively, so
about 56 % of the bytes; other engines are time-stretched after
synthesis); `mp3: HTTP 200 14232 bytes audio/mpeg`; the `aac` line is an
OpenAI error envelope, `"code":"unsupported_format"` with `available:
mp3, opus, flac, wav, pcm`, `HTTP 400`; `speed` 5 gives `"speed: Input
should be less than or equal to 4"` `HTTP 400`; the unknown voice gives
`"code":"voice_not_found"` `HTTP 404`; the native stream answers with
`x-voicebox-stream-mode: chunked`, `x-voicebox-sample-format: s16le` and
a job id (`"engine":"kokoro"` is required there: the field defaults to
`qwen`); the voices list has `count: 65` (the two profiles, `owner: null,
shared: true`, plus the presets).

## 8. Transcription with timestamps

```bash
curl -s -H "Authorization: Bearer $CLIENT_KEY" -F file=@ref.wav -F model=whisper-base -F response_format=srt $B/v1/audio/transcriptions
curl -s -H "Authorization: Bearer $CLIENT_KEY" -F file=@ref.wav -F model=whisper-base -F response_format=verbose_json $B/v1/audio/transcriptions | jq -c '{task,language,duration,segments: [.segments[]|{id,start,end}]}'
curl -s -w ' HTTP %{http_code}\n' -H "Authorization: Bearer $CLIENT_KEY" -F file=@ref.wav -F model=whisper-base -F response_format=verbose_json -F 'timestamp_granularities[]=word' $B/v1/audio/transcriptions
```

**Check:** two SRT cues (`1` / `00:00:00,000 --> 00:00:06,220` / `The
quick brown fox ...`, then the second sentence); a `verbose_json` object
with `"duration":9.125` and three segments; then `"timestamp_granularities
'word' is not available; only segment timestamps are"` with
`"code":"unsupported_value"` `HTTP 400`. (Whisper base mishears
"light" as "lights"; the medium model in the cache is more accurate and
slower, `-F model=whisper-medium`.)

## 9. Live transcription over a WebSocket

The example client streams a file at real-time pace and prints the
server's events; deltas appear inline while the audio plays.

```bash
backend/venv/bin/python scripts/realtime_client.py --url ws://127.0.0.1:$PORT --key "$CLIENT_KEY" --model whisper-base --language en ref.wav
backend/venv/bin/python scripts/realtime_client.py --url ws://127.0.0.1:$PORT --key "$CLIENT_KEY" --model whisper-base --language en --no-vad --speed 2 ref.wav
backend/venv/bin/python scripts/realtime_client.py --url ws://127.0.0.1:$PORT --key vbx_wrong ref.wav 2>&1 | tail -1
```

**Check:** the first run prints `< input_audio_buffer.speech_started`,
deltas such as `The quick brown` while it streams, `speech_stopped`,
`committed`, then `< completed [item_...]: The quick brown fox jumps over
the lazy dog while the river runs quietly past the old stone bridge.` and
a second completed item for `and the evening lights settles over the
valley.`, ending `2 utterance(s) in 12.7 s` (the server VAD split at the
comma). The manual-turn run commits once and prints one completed item in
about 5.7 s; with Whisper base and a single 30 s window it appended
`Thank you for watching.`, a known Whisper hallucination over the padding
that the server VAD path trims and the base model sometimes still
produces. The bad key ends with `server rejected WebSocket connection:
HTTP 403`.

## 10. MCP without a session, and the metrics

`VOICEBOX_MCP_STATELESS=1` at start is what lets a single `tools/call`
work without `initialize`, which is how a load-balanced fleet has to run.

```bash
curl -s -H "Authorization: Bearer $CLIENT_KEY" -H 'Accept: application/json, text/event-stream' -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"voicebox.list_profiles","arguments":{}}}' $B/mcp/ | head -c 300; echo
curl -s -H "Authorization: Bearer $CLIENT_KEY" -H 'Accept: application/json, text/event-stream' -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' $B/mcp/ | sed -n 's/^data: //p' | jq -c '[.result.tools[].name]'
curl -s -H "Authorization: Bearer $ADMIN_KEY" $B/metrics | grep -E '^voicebox_(queue_running_jobs|http_requests_in_flight|generations_total|realtime_utterances_total)'
curl -s -w ' HTTP %{http_code}\n' -H "Authorization: Bearer $CLIENT_KEY" $B/metrics
```

**Check:** an SSE frame `event: message` / `data: {"jsonrpc":"2.0","id":1,
"result":{"content":[{"type":"text","text":"{\"profiles\":[...CloneTest...Narrator...`;
`["voicebox.speak","voicebox.transcribe","voicebox.list_captures","voicebox.list_profiles"]`;
metric lines such as `voicebox_queue_running_jobs{lane="all"} 0.0`,
`voicebox_generations_total{engine="kokoro",kind="stream",status="completed"} 4.0`
and `voicebox_realtime_utterances_total 3.0`; then `{"detail":"Admin key
required"} HTTP 403`.

## 11. Limits, and the CPU baseline

Six requests at once on the default client key (4 pending jobs allowed),
then the load test with the unlimited key:

```bash
for i in 1 2 3 4 5 6; do curl -s -o /dev/null -w '%{http_code} %{header_json}\n' -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"kokoro\",\"voice\":\"af_heart\",\"input\":\"Parallel request number $i, testing the per-key pending-job limit with a sentence long enough to take a moment.\"}" $B/v1/audio/speech | cut -c1-3 & done; wait; echo
backend/venv/bin/python scripts/load_test.py --url $B --key "$LOAD_KEY" --voice af_heart --model kokoro --concurrency 2 --requests 10
```

**Check:** a mix of `200` and `429` (four and two on this Mac; the `429`
carries `retry-after: 5`), then:

```
requests      10 total, 2 concurrent, 10 ok, 0 x 429, 0 failed
wall time     4.5 s  (133.4 requests/min)
first byte    p50 0.89 s   p95 0.94 s   max 0.94 s
realtime      median 6.56x   (audio seconds per wall second, per request)
audio         57.4 s produced, 12.75x realtime across the run
```

Write those two `realtime` numbers down: they are the CPU baseline the
GPU table in document 6 is compared against.

## 12. A graceful stop

Start a long request, then send the server `SIGTERM` (Ctrl-C in terminal
1, or `kill -TERM`):

```bash
LONG=$(backend/venv/bin/python -c "print(' '.join('Sentence number %d of a deliberately long text that keeps the synthesizer busy for a while.' % i for i in range(1, 41)))")
(curl -s -o long.wav -w 'long request: HTTP %{http_code} %{size_download} bytes in %{time_total}s\n' -H "Authorization: Bearer $LOAD_KEY" -H 'Content-Type: application/json' -d "{\"model\":\"kokoro\",\"voice\":\"af_heart\",\"input\":\"$LONG\",\"response_format\":\"wav\"}" $B/v1/audio/speech &)
sleep 3; kill -TERM $(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t); sleep 1
curl -s -w ' HTTP %{http_code}\n' $B/health/ready
curl -s -o /dev/null -w 'new request: HTTP %{http_code}\n' -H "Authorization: Bearer $LOAD_KEY" -H 'Content-Type: application/json' -d '{"model":"kokoro","voice":"af_heart","input":"Still accepted while stopping."}' $B/v1/audio/speech
wait
```

**Check:** readiness answers `{"ready":false,...,"stopping":true,...} HTTP
503` while `/health` still says healthy; the new request is `HTTP 200`;
the long one finishes `HTTP 200 11274044 bytes in 19.9s`; terminal 1
prints `Stopping in 10 s (signal SIGTERM): readiness answers 503, requests
are still served`, then `Draining (signal SIGTERM): refusing new jobs,
finishing the ones in flight`, `Generation queue drained`, and exits.
That sequence is what lets a load balancer stop routing before the
listener closes; a second signal skips the delay.

## 13. The Docker image and its smoke test

```bash
docker build -t voicebox:local --build-arg PYTORCH_VARIANT=cpu .
SMOKE_PORT=17601 scripts/docker-smoke.sh voicebox:local
```

**Check:** the first build takes 10–25 minutes (a rebuild with the layer
cache took 3.5 minutes); the smoke test ends `smoke test passed: 218444
bytes of Kokoro audio, stopped in 1s` after booting the image with a
Kokoro preload, creating a client key, streaming speech through it,
checking a client key is refused on an admin route, and stopping the
container gracefully. `SMOKE_PORT` keeps it off 17493. The published
`ghcr.io/andrew-zhao12/voicebox:main-cpu` is linux/amd64 only, which is
why a native `voicebox:local` is built here.

**If not:** the script prints the container's last 60 log lines. A
readiness timeout on the first run is the Kokoro download; `SMOKE_READY_TIMEOUT=1800`
gives it longer.

## 14. One container with the UI, and an app-owned voice

```bash
docker tag voicebox:local voicebox-voicebox:latest     # the name `docker compose build` would give it
docker compose up -d --no-build
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:17600/health/ready)" = 200 ]; do sleep 5; done
DK=$(docker compose exec -T voicebox cat /app/data/api_key | tr -d '\r\n')
APPKEY=$(docker compose exec -T --user voicebox voicebox python -m backend.keys create --id myapp --role client --data-dir /app/data 2>/dev/null | tr -d '\r\n')
sleep 2; curl -s -H "Authorization: Bearer $APPKEY" http://127.0.0.1:17600/auth/whoami | jq -c '{key_id,role}'
curl -s -H "Authorization: Bearer $APPKEY" -F name=AppVoice -F file=@ref.wav -F "text=$T" -F engine=qwen http://127.0.0.1:17600/v1/voices | jq -c '{name,kind,engine,owner,shared}'
curl -s -H "Authorization: Bearer $DK" http://127.0.0.1:17600/profiles | jq -c '.[]|{name,owner_key_id,voice_type}'
```

Open http://localhost:17600, paste the admin key (`echo $DK`) into the
Connect screen, and look at Voice Profiles: `AppVoice` carries an owner
badge with `myapp`, the key that created it through the API. Then:

```bash
docker compose down
```

**Check:** `{"key_id":"myapp","role":"client"}`, then
`{"name":"AppVoice","kind":"profile","engine":"qwen","owner":"myapp","shared":false}`
and `{"name":"AppVoice","owner_key_id":"myapp","voice_type":"cloned"}`.
The `--user voicebox` matters: `docker compose exec` runs as root, and a
keys file written by root is unreadable to the server, which then answers
`401` for the new key. Speaking with `AppVoice` needs the Qwen model,
which this container has not downloaded (`400`, `Model 1.7B is not
downloaded yet`); Kokoro presets work in the UI right away.

## 15. Engine concurrency (Linux only) and the metrics exporter

`VOICEBOX_ENGINE_CONCURRENCY` is ignored on the MLX backend, so it is
shown in the Linux image: two Kokoro jobs run at once and the running-jobs
gauge says so.

```bash
CK=vbx_conc_$(openssl rand -hex 8)
docker run -d --name vb-conc -p 127.0.0.1:17602:17493 -p 127.0.0.1:9464:9464 -e VOICEBOX_API_KEY=$CK \
  -e VOICEBOX_PRELOAD_MODELS=kokoro -e VOICEBOX_ENGINE_CONCURRENCY=kokoro=2 -e VOICEBOX_METRICS_PORT=9464 voicebox:local
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:17602/health/ready)" = 200 ]; do sleep 5; done
docker logs vb-conc 2>&1 | grep -E 'Engine concurrency|Metrics exporter'
(backend/venv/bin/python scripts/load_test.py --url http://127.0.0.1:17602 --key $CK --voice af_heart --model kokoro --concurrency 2 --requests 12 &)
MAX=0; for i in $(seq 1 60); do v=$(curl -s http://127.0.0.1:9464/metrics | awk '/^voicebox_queue_running_jobs\{lane="all"\}/{print int($2)}'); [ -n "$v" ] && [ "$v" -gt "$MAX" ] && MAX=$v; sleep 0.5; done; echo "max running jobs: $MAX"
wait; docker rm -f vb-conc
```

**Check:** `Engine concurrency: kokoro=2` and `Metrics exporter
listening on 0.0.0.0:9464` in the log, `max running jobs: 2` while the
load test runs, and the load test's own `12 ok, 0 x 429, 0 failed`. (On
this Mac's CPU the two jobs share the cores, so per-request speed halves
and the run's throughput stays about the same, `6.09x` against `5.99x`
serial: exactly the measurement that has to be repeated on a GPU before
the flag is worth enabling anywhere.)

## 16. The fleet: two replicas behind Caddy

The fleet takes the same three inputs the cloud recipes take: the seed
from step 6, a keys file, and a model cache its `models` service fills
once. It publishes only Caddy on 17600.

```bash
CLIENT_KEY=$(backend/venv/bin/python -m backend.keys create --id myapp --role client --data-dir ./secrets)
LOAD_KEY=$(backend/venv/bin/python -m backend.keys create --id loadtest --role client --data-dir ./secrets \
  --limit inference=unlimited --limit requests=unlimited --limit tts_chars=unlimited --limit max_pending_jobs=unlimited)
printf 'export CLIENT_KEY=%s\nexport LOAD_KEY=%s\n' "$CLIENT_KEY" "$LOAD_KEY" >> ~/.voicebox-cloud.env
export VOICEBOX_API_KEY=$(openssl rand -base64 32) VOICEBOX_MEDIA_TOKEN_SECRET=$(openssl rand -base64 48)
VOICEBOX_PRELOAD_MODELS=kokoro,whisper-base VOICEBOX_IMAGE=voicebox:local docker compose -f docker-compose.fleet.yml up -d
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:17600/health/ready)" = 200 ]; do sleep 5; done
docker compose -f docker-compose.fleet.yml ps --format '{{.Service}} {{.Status}}'
curl -s http://127.0.0.1:17600/health/ready | jq -c '{ready, models: .models.ready, startup: .startup.done}'
scripts/fleet-check.sh http://127.0.0.1:17600 "$CLIENT_KEY" --rounds 6
```

**Check:** the fleet was ready 18–34 s after `up` (the cache was warm; a
first run downloads the two models once, for both replicas), both
replicas show `(healthy)`, the readiness body says
`{"ready":true,"models":["kokoro","whisper-base"],"startup":["seed_profiles"]}`,
and the six rounds print a constant `catalog: 65 3a8cd20108b5b61e`
fingerprint, `speech: ... bytes via kokoro` and a `transcription:` line
each, ending `fleet check passed: 6 round(s) against http://127.0.0.1:17600`.
The keys just made are the ones the cloud documents reuse, which is why
they also go into `~/.voicebox-cloud.env`.

The three things a fleet has to get right: the same voice ids on every
replica (an application stores them), MCP calls that land on any replica,
and a WebSocket through the balancer.

```bash
for r in voicebox-a voicebox-b; do printf '%s: ' $r; docker compose -f docker-compose.fleet.yml exec -T $r curl -s -H "Authorization: Bearer $VOICEBOX_API_KEY" localhost:17493/v1/voices | jq -c '[.data[]|select(.kind=="profile")|{name,id: (.id|.[0:8])}]'; done
for i in 1 2 3 4; do curl -s -o /dev/null -w '%{http_code} ' -H "Authorization: Bearer $CLIENT_KEY" -H 'Accept: application/json, text/event-stream' -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"voicebox.list_profiles","arguments":{}}}' http://127.0.0.1:17600/mcp/; done; echo
backend/venv/bin/python scripts/realtime_client.py --url ws://127.0.0.1:17600 --key "$CLIENT_KEY" --model whisper-base --language en ref.wav | grep -E 'completed|utterance'
TOKEN=$(curl -s -X POST -H "Authorization: Bearer $CLIENT_KEY" http://127.0.0.1:17600/auth/media-token | jq -r .token)
curl -s -o /dev/null --max-time 4 -w '%{http_code}\n' --http1.1 -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H 'Sec-WebSocket-Version: 13' -H "Sec-WebSocket-Key: $(openssl rand -base64 16)" "http://127.0.0.1:17600/v1/realtime/transcription?token=$TOKEN"
```

**Check:** both replicas list `CloneTest` and `Narrator` with the same
ids (the bundle carries the id); `200 200 200 200`; two completed
utterances through Caddy; `101` for the handshake with a media token
(curl then times out on purpose; the media secret is shared, so the token
issued by one replica verified on whichever replica Caddy picked).

Now the part that matters for scaling: a replica goes away and comes back
while requests flow, and a replica is replaced (a rolling update) while
requests flow.

```bash
(backend/venv/bin/python scripts/load_test.py --url http://127.0.0.1:17600 --key "$LOAD_KEY" --voice af_heart --model kokoro --concurrency 2 --requests 40 > stop.txt &)
sleep 4; docker compose -f docker-compose.fleet.yml stop voicebox-b; wait; grep -E 'requests|failure' stop.txt
docker compose -f docker-compose.fleet.yml logs --no-log-prefix voicebox-b | grep -E 'Stopping|Draining'
docker compose -f docker-compose.fleet.yml start voicebox-b
(backend/venv/bin/python scripts/load_test.py --url http://127.0.0.1:17600 --key "$LOAD_KEY" --voice af_heart --model kokoro --concurrency 2 --requests 40 > recreate.txt &)
sleep 4; docker compose -f docker-compose.fleet.yml up -d --no-deps --force-recreate voicebox-a; wait; grep -E 'requests|failure' recreate.txt
docker compose -f docker-compose.fleet.yml exec -T voicebox-b curl -s localhost:9464/metrics | grep -E '^voicebox_(queue_pending_jobs|http_requests_in_flight|generations_total)'
docker compose -f docker-compose.fleet.yml down -v
```

**Check:** both load tests end `40 total, 2 concurrent, 40 ok, 0 x 429, 0
failed` (no `failure` line; on this Mac the wall time was 52 s with one
replica stopping and 48 s with one being recreated, against 43–44 s
undisturbed); the stopped replica's log shows `Stopping in 10 s (signal
SIGTERM): readiness answers 503, requests are still served` and, ten
seconds later, `Draining (signal SIGTERM): refusing new jobs, finishing
the ones in flight`, which is the window in which Caddy stopped routing
to it; the metrics exporter on the replicas' private port answers with
`voicebox_queue_pending_jobs`, `voicebox_http_requests_in_flight` and the
`voicebox_generations_total` counter; `down -v` removes the containers and
the three volumes (the models cache included; drop `-v` to keep it for the
next run).

**If not:** a `failure HTTP 502` in a run has had two causes on this Mac,
both fixed in the repository on 2026-09-29 and both worth knowing about
because a cloud balancer can show the same symptom. First, uvicorn's
default keep-alive of 5 s let Caddy reuse an idle connection the replica
had just closed (`EOF` in Caddy's log); the image now starts uvicorn with
`--timeout-keep-alive 650`. Second, two CPU replicas each started a torch
thread per core, the host's load average passed 70, and both replicas
missed the 2 s readiness probe at the same moment (`no upstreams
available` in Caddy's log); `docker-compose.fleet.yml` now caps each
replica at `VOICEBOX_REPLICA_THREADS` (default 4) and the probe timeout is
5 s. Run `uptime` during a load test: the load average should stay near
the number of threads you allowed, not the number of cores times two.

## Done when

- [ ] The suite and ruff are green; the scratch server answered every check in steps 4–12 with the expected bodies and codes.
- [ ] `ref.wav` plays; `seed/` holds `Narrator.voicebox.zip` and `CloneTest.voicebox.zip`.
- [ ] The smoke test passed; the UI showed the owner badge; the running-jobs gauge reached 2.
- [ ] `fleet-check.sh` passed six rounds against the fleet; both replicas carry the same profile ids; the stop and the recreate ran with `0 failed`.
- [ ] `secrets/api_keys.json` lists `myapp` and `loadtest`, and `~/.voicebox-cloud.env` holds `CLIENT_KEY` and `LOAD_KEY`.

**Not covered here.** The desktop app (needs `bun` and Rust; see the
developer setup page) and every GPU engine: this Mac runs Kokoro and
Whisper on CPU and MLX, and Qwen, LuxTTS, Chatterbox and TADA are measured
in [document 6](06-test-on-the-cloud.md). Continue with
[document 2](02-cloud-prep.md).
