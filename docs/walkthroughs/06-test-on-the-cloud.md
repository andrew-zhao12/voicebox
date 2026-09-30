# 6. Test on the cloud

**Goal.** Prove the deployment from [document 3](03-deploy-gcp-cloud-run.md),
[4](04-deploy-aws-ecs.md) or [5](05-deploy-kubernetes-keda.md) the way an
application would use it, measure every engine on the GPU, watch the
platform scale out and back in, survive a rolling update under load, look
at the bill, and tear everything down.

**Time.** About an hour for the functional part, one to two hours for the
engine measurements (models load once each), half an hour for the scaling
checks. **Money.** The deployment while you test, plus the extra GPU
replicas a scale-out adds for a few minutes. **You need.** `URL` (and the
rest) in `~/.voicebox-cloud.env` from the deploy document, `CLIENT_KEY`,
`LOAD_KEY` and `ADMIN_KEY` from [document 2](02-cloud-prep.md), `jq`,
`backend/venv/bin/python` for the scripts. **What was run.** The
platform-independent checks in steps 2 and 3 are the same commands
[document 1](01-test-locally.md) ran against the local fleet on 2026-09-29;
the scaling and rolling-update sections have not been run on a platform yet.

```bash
source ~/.voicebox-cloud.env
WS_URL=${URL/#http/ws}          # https:// -> wss://, http:// -> ws://
echo "$URL $WS_URL"
```

## 1. Health and identity

```bash
curl -s $URL/health/ready | jq -c '{ready, draining, models: .models.ready, failed: .models.failed, startup: .startup.done}'
curl -s -H "Authorization: Bearer $CLIENT_KEY" $URL/auth/whoami | jq -c '{key_id, role, limits: .limits.inference}'
curl -s -H "Authorization: Bearer $CLIENT_KEY" $URL/v1/models | jq -c '[.data[]|select(.downloaded)|.id]'
curl -s -H "Authorization: Bearer $CLIENT_KEY" $URL/v1/voices | jq -c '[.data[]|select(.kind=="profile")|.name]'
curl -s -o /dev/null -w '%{http_code}\n' $URL/profiles
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $CLIENT_KEY" $URL/history
```

**Check:** `ready: true` with `startup` containing `seed_profiles` and
`gpu` and `failed: []`; `key_id: myapp`, `role: client`, `inference: 30`;
the model list includes `kokoro` and the Whisper you preloaded (the other
engines show `downloaded: true` once the cache is filled and `loaded`
only after first use); the voices list is `["CloneTest","Narrator"]`; then
`401` (no key) and `403` (a client key on an admin route).

**If not:** `failed` naming a model means the cache lacks it on that
replica; `startup` without `gpu` means `VOICEBOX_REQUIRE_GPU` is not set
in the manifest you deployed (the recipes set it).

## 2. The API, request by request

Each line is one thing an application does. The byte counts are from the
local run and only the ratios matter.

```bash
T="The quick brown fox jumps over the lazy dog while the river runs quietly past the old stone bridge, and the evening light settles over the valley."
curl -s -o clip.wav -D - -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"kokoro\",\"voice\":\"af_heart\",\"input\":\"$T\",\"response_format\":\"wav\"}" $URL/v1/audio/speech | grep -iE '^HTTP|x-voicebox-(engine|voice|sample-rate)'
ls -l clip.wav | awk '{print $5" bytes"}'
curl -s -o /dev/null -w '%{size_download} bytes at speed 2\n' -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"kokoro\",\"voice\":\"af_heart\",\"input\":\"$T\",\"response_format\":\"wav\",\"speed\":2.0}" $URL/v1/audio/speech
curl -s -o /dev/null -w '%{http_code} %{content_type}\n' -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","voice":"af_heart","input":"An mp3 check.","response_format":"mp3"}' $URL/v1/audio/speech
curl -s -H "Authorization: Bearer $CLIENT_KEY" -F file=@clip.wav -F model=whisper-1 -F response_format=srt $URL/v1/audio/transcriptions | head -4
curl -s -H "Authorization: Bearer $CLIENT_KEY" -F file=@clip.wav -F model=whisper-1 -F response_format=verbose_json $URL/v1/audio/transcriptions | jq -c '{language, duration, segments: (.segments|length)}'
curl -s -w ' %{http_code}\n' -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' -d '{"model":"kokoro","voice":"nobody","input":"x"}' $URL/v1/audio/speech
```

**Check:** `HTTP/2 200` (or `HTTP/1.1 200`) with `x-voicebox-engine:
kokoro`, `x-voicebox-voice: kokoro:af_heart`, `x-voicebox-sample-rate:
24000`; about 438 000 bytes, then about 244 000 at speed 2 (Kokoro
speaks faster natively, so the audio is shorter, not resampled); `200
audio/mpeg`; an SRT cue block starting `1` / `00:00:00,000 -->`; a
`verbose_json` object with `segments` above 0; and the OpenAI error
envelope `{"error":{...,"code":"voice_not_found"}} 404`.

Now the same as an application fleet would, six rounds through the
balancer, and the per-key limit:

```bash
scripts/fleet-check.sh "$URL" "$CLIENT_KEY" --rounds 6
for i in 1 2 3 4 5 6; do curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $CLIENT_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"kokoro\",\"voice\":\"af_heart\",\"input\":\"Parallel request $i, long enough to take a moment on the queue.\"}" $URL/v1/audio/speech & done; wait
```

**Check:** `fleet check passed: 6 round(s) against ...` with the same
`catalog:` fingerprint every round (with more than one replica this is the
proof that every replica applied the same seed), then a mix of `200` and
`429` (the `myapp` key allows 4 pending jobs; locally 2 of 6 were refused
with `retry-after: 5`). If every one of the six is `200`, the queue
emptied faster than the requests arrived, which is fine on a GPU.

## 3. The things a browser or a proxy would trip on

```bash
curl -s -o /dev/null -w '%{http_code}\n' -H 'Host: evil.example' -H "Authorization: Bearer $CLIENT_KEY" $URL/v1/models
TOKEN=$(curl -s -X POST -H "Authorization: Bearer $CLIENT_KEY" $URL/auth/media-token | jq -r .token)
curl -s -o /dev/null --max-time 4 -w '%{http_code}\n' --http1.1 -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H 'Sec-WebSocket-Version: 13' \
  -H "Sec-WebSocket-Key: $(openssl rand -base64 16)" "$URL/v1/realtime/transcription?token=$TOKEN"
curl -s -o /dev/null --max-time 4 -w '%{http_code}\n' --http1.1 -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H 'Sec-WebSocket-Version: 13' \
  -H "Sec-WebSocket-Key: $(openssl rand -base64 16)" "$URL/v1/realtime/transcription"
curl -s -H "Authorization: Bearer $ADMIN_KEY" $URL/metrics | grep -E '^voicebox_(queue_pending_jobs|gpu_memory_bytes|realtime_sessions) '
```

**Check:** `400` for the foreign `Host` (Cloud Run's front end and
ingress-nginx may answer `404` first; either way nothing is served), `101`
for the handshake with a media token (curl then times out on purpose, the
socket was open) and `403` without one, and the metric lines including
`voicebox_gpu_memory_bytes{kind="allocated"}` above zero. With several
replicas the media-token check is also the proof that
`VOICEBOX_MEDIA_TOKEN_SECRET` is shared: the token was issued by one
replica and verified by whichever the balancer picked.

## 4. Live transcription over the balancer

```bash
backend/venv/bin/python scripts/realtime_client.py --url "$WS_URL" --key "$CLIENT_KEY" --model whisper-1 --language en clip.wav
backend/venv/bin/python scripts/realtime_client.py --url "$WS_URL" --key "$CLIENT_KEY" --model whisper-1 --language en --no-vad clip.wav
```

**Check:** the first run prints `< completed [item_...]: The quick brown
fox ...` for each utterance the server VAD found (two, with a pause at the
comma, in the local run) and ends `2 utterance(s) in ... s`, exit 0; the
second commits once and prints one completed item. On Cloud Run a session
is one request: the server closes it cleanly after
`VOICEBOX_REALTIME_MAX_SESSION_S` (840 s in the recipe) so the platform's
900 s request timeout never cuts it mid-word; an application reconnects.

**If not:** `HTTP 403` at the handshake is a bad key; a close with code
`1013` is the per-key session cap (2 for a client key); a close with
`1008` and `model_not_found` means the Whisper model is not on that
replica; a hang with no events means the balancer did not upgrade the
connection (the recipes set the timeouts and Caddy, Cloud Run, the ALB and
ingress-nginx all proxy WebSockets; a corporate proxy on your side may not).

## 5. The engine measurements

These numbers are what decides `VOICEBOX_ENGINE_CONCURRENCY` (and
`VOICEBOX_GENERATION_WORKERS=2`), so they are taken on one replica, with
the `loadtest` key (no per-minute limits; the replica's queue depth of 4
still applies and shows up as `429`).

Pin the deployment to one replica for the duration:

| Platform | Pin | Unpin |
|----------|-----|-------|
| Cloud Run | `gcloud run services update voicebox --region $REGION --max-instances 1` | `--max-instances 3` |
| ECS | `aws application-autoscaling register-scalable-target --service-namespace ecs --scalable-dimension ecs:service:DesiredCount --resource-id service/voicebox/voicebox --min-capacity 1 --max-capacity 1` | `--max-capacity 4` |
| Kubernetes | `kubectl -n voicebox patch scaledobject voicebox --type merge -p '{"spec":{"maxReplicaCount":1}}'` | `{"maxReplicaCount":4}` |

Then, for each row of the table, run the load test at concurrency 1 and
2. The first request of each engine loads it (allow minutes: the cache is
on a bucket or a network volume), which is why `--warmup 1` and a long
timeout are there.

| Engine (`--model`) | Voice (`--voice`) | Why |
|--------------------|-------------------|-----|
| `kokoro` | `af_heart` | the CPU-friendly baseline |
| `qwen_custom_voice` | `Ryan` | Qwen with a preset speaker |
| `qwen` | `CloneTest` | Qwen cloning from the seed's sample |
| `luxtts` | `CloneTest` | the small cloning engine |
| `chatterbox_turbo` | `CloneTest` | reviewed as unsafe for concurrency (per-call state on the model) |
| `tada` | `CloneTest` | the largest of the set |

```bash
for c in 1 2; do
  backend/venv/bin/python scripts/load_test.py --url $URL --key "$LOAD_KEY" --model ENGINE --voice VOICE --format wav --warmup 1 --requests 12 --concurrency $c --timeout 600
done
curl -s -H "Authorization: Bearer $ADMIN_KEY" $URL/metrics | grep '^voicebox_gpu_memory_bytes'
```

Record per engine and concurrency: `first byte p50 / p95`, `total p50`,
`realtime median` (per request), `audio ... x realtime across the run`
(the throughput figure), the `429` and `failed` counts, and the allocated
GPU bytes after the run. For reference, Kokoro on this Mac's CPU at
concurrency 2 gave `first byte p50 0.89 s`, `realtime median 6.56x` and
`12.75x realtime across the run`; an L4 should do considerably better and
the other engines considerably worse than Kokoro.

Free the GPU between engines so the next one is measured alone (the
route answers `409` while the engine is busy; the names are the ones
`backend/venv/bin/python -m backend.preload --list` prints):

```bash
curl -s -X POST -H "Authorization: Bearer $ADMIN_KEY" $URL/models/MODEL-NAME/unload
```

Then the same rows with the flag on, at concurrency 2 and 4. Only the
engines the code reviewed as safe accept a value above 1 (`kokoro`, `qwen`,
`qwen_custom_voice`, `luxtts`, `tada`; Chatterbox is forced to 1 with a
warning in the log), and the setting is applied by a new revision:

| Platform | Apply |
|----------|-------|
| Cloud Run | `gcloud run services update voicebox --region $REGION --update-env-vars VOICEBOX_ENGINE_CONCURRENCY=kokoro=2,qwen=2,qwen_custom_voice=2,luxtts=2,tada=2` |
| ECS | add the variable to `~/voicebox-cloud/task-definition.json`, `register-task-definition` again, then `aws ecs update-service --cluster voicebox --service voicebox --task-definition voicebox` |
| Kubernetes | `kubectl -n voicebox patch configmap voicebox-config --type merge -p '{"data":{"VOICEBOX_ENGINE_CONCURRENCY":"kokoro=2,qwen=2,qwen_custom_voice=2,luxtts=2,tada=2"}}' && kubectl -n voicebox rollout restart deploy/voicebox` |

**The rule.** Turn the flag on for an engine only if all three hold at
concurrency 2 compared with its own serial run at concurrency 2:

1. `audio ... x realtime across the run` improves by at least 1.3×;
2. `first byte p95` is not worse;
3. `voicebox_gpu_memory_bytes{kind="allocated"}` stays below 75 % of the
   card (18 GB of an L4's 24 GB) with the engine loaded and two jobs
   running, so a third engine or a second voice does not push it over.

Otherwise leave it off: on a GPU that is already saturated by one job,
two jobs only share the same throughput and add latency. Unpin the
deployment when the table is complete.

**If not:** a run with `failed` above zero is not a measurement; read the
replica's log for the error (an out-of-memory on the card shows as a
`CUDA out of memory` line) and retry after an unload.

## 6. Scale-out and scale-in

Sustained load with a slow engine so jobs queue on the first replica and
the platform has a reason to add one:

```bash
backend/venv/bin/python scripts/load_test.py --url $URL --key "$LOAD_KEY" --model qwen --voice CloneTest --concurrency 6 --requests 90 --timeout 600
```

While it runs, in a second terminal, watch the replica count:

| Platform | Watch | Expect |
|----------|-------|--------|
| Cloud Run | Console → Cloud Run → voicebox → Metrics → *Container instance count*, or `watch -n 15 "gcloud run services describe voicebox --region $REGION --format='value(status.traffic[0].latestRevision)'"` and the revision's instances in the console | a second instance within a minute of the queue filling; GPU instances start in seconds, then load the models from the bucket |
| ECS | `watch -n 15 "aws ecs describe-services --cluster voicebox --services voicebox --query 'services[0].{running:runningCount,desired:desiredCount}'"` and `aws autoscaling describe-scaling-activities --auto-scaling-group-name voicebox-gpu --max-items 3` | `desired` rises after the alarm has been high for its evaluation period (a minute or two), then the capacity provider launches a g6 (three to five minutes to join), then `running` follows |
| Kubernetes | `kubectl -n voicebox get hpa -w` and `kubectl get nodes -w` | the HPA target leaves `0/2` within 15 s of the queue filling, a pod is Pending until the pool adds a GPU node (minutes), then Running |

**Check:** the load test ends with `0 failed` and the platform ran more
than one replica at some point. A few `429` lines are the queue-depth cap
doing its job while the platform catches up; the OpenAI SDKs retry them.
On ECS a scale-out that never happens usually means the target-tracking
alarm is fine but the Auto Scaling group hit the quota from document 4
step 1.

Then stop the load and wait for scale-in (Cloud Run idles instances out in
about 15 minutes; ECS has `ScaleInCooldown 300`; KEDA `cooldownPeriod 300`
plus a 300 s stabilisation window). While it happens, a light load proves
that removing a replica drops nothing:

```bash
backend/venv/bin/python scripts/load_test.py --url $URL --key "$LOAD_KEY" --model kokoro --voice af_heart --concurrency 1 --requests 60
```

**Check:** `60 ok, 0 x 429, 0 failed`, and the replica count is back to
the minimum afterwards. That is `VOICEBOX_SHUTDOWN_DELAY_S` at work: the
retiring replica fails readiness first, keeps serving while the balancer
stops routing to it, then drains.

## 7. A rolling update under load

Start a load test and one live session, then trigger a new revision with
an unrelated setting:

```bash
backend/venv/bin/python scripts/load_test.py --url $URL --key "$LOAD_KEY" --model kokoro --voice af_heart --concurrency 2 --requests 200 &
backend/venv/bin/python scripts/realtime_client.py --url "$WS_URL" --key "$CLIENT_KEY" --model whisper-1 --speed 0.5 clip.wav &
```

| Platform | Trigger |
|----------|---------|
| Cloud Run | `gcloud run services update voicebox --region $REGION --update-env-vars VOICEBOX_RETENTION_DAYS=2` |
| ECS | `aws ecs update-service --cluster voicebox --service voicebox --force-new-deployment` (needs room for a second task: the capacity provider adds an instance first, so this takes minutes) |
| Kubernetes | `kubectl -n voicebox rollout restart deploy/voicebox && kubectl -n voicebox rollout status deploy/voicebox` (`maxSurge 1`, so a second GPU node if the pool has none free) |

**Check:** the load test ends with `0 failed`; the live session ends with
the client printing an error or close with code `1012` (the retiring
replica tells sockets to reconnect); the old replica's log shows
`Stopping in N s (signal SIGTERM): readiness answers 503, requests are
still served` and then `Draining`. On ECS, scale-in is suspended during
the deployment, so do this after the scale-in check, not during.

## 8. Cost, then teardown

Look at the bill before tearing down, so you know what a day of testing
costs on this platform: Cloud Run → Billing → Reports filtered by the
project; AWS → Billing → Cost Explorer (daily, by service: EC2, ELB, EFS);
GKE → Billing → Reports (Kubernetes Engine and Compute Engine).

Then the teardown section of the document you deployed with:
[Cloud Run](03-deploy-gcp-cloud-run.md#teardown),
[ECS](04-deploy-aws-ecs.md#teardown),
[Kubernetes](05-deploy-kubernetes-keda.md#teardown). Delete `clip.wav`
and, if you created it for the test, the `loadtest` key
(`backend/venv/bin/python -m backend.keys revoke --id loadtest --data-dir ./secrets`).

## 9. Write it down

| Engine | Voice | c | first byte p50 / p95 | total p50 | realtime (median) | run throughput | 429 / failed | GPU bytes | flag |
|--------|-------|---|----------------------|-----------|-------------------|----------------|--------------|-----------|------|
| kokoro | af_heart | 1 | | | | | | | off |
| kokoro | af_heart | 2 | | | | | | | off |
| kokoro | af_heart | 2 | | | | | | | 2 |
| ... | | | | | | | | | |

- The table and the platform go into
  [`remote-mode.mdx`](../content/docs/overview/remote-mode.mdx) next to
  the Kokoro figure it already has.
- The `VOICEBOX_ENGINE_CONCURRENCY` decision, the scale-out timings you
  observed and anything a recipe got wrong go into
  [`PROJECT_STATUS.md`](../PROJECT_STATUS.md) (and the recipe).

## Done when

- [ ] Steps 1–4 pass over the public URL (every check line matched).
- [ ] The measurement table is filled for every engine at c=1 and c=2, with and without the flag.
- [ ] A scale-out and a scale-in were observed with `0 failed`; a rolling update ran with `0 failed` and a `1012` close.
- [ ] The bill was looked at and the teardown ran; `URL` no longer answers.
