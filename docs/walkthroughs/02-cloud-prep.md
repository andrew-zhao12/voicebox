# 2. Cloud prep: once, before any cloud

**Goal.** Everything the three cloud documents share, done once: the CUDA
image published, the keys and secrets generated and saved, the voice seed
and model set decided, the settings and quotas every platform needs.

**Time.** About 45 minutes of your attention, plus a 25 minute image
build on GitHub that you only wait for (24.6 minutes on 2026-09-29). **Money.** None. **You need.**
[Document 1](01-test-locally.md) done (it leaves `./seed` and
`./secrets/api_keys.json` behind), `gh` logged in to the fork, `docker`,
`jq`, `openssl`. **At the end.** `ghcr.io/andrew-zhao12/voicebox:main-cu128`
exists and is public, `~/.voicebox-cloud.env` holds `CLIENT_KEY`,
`LOAD_KEY`, `ADMIN_KEY` and `MEDIA_SECRET`, and you know which quota to
request before touching a cloud.

## 1. Publish the CUDA image

The CPU image is published on every push to `main`; the CUDA image only on
release tags or on demand. Trigger the on-demand build:

```bash
gh workflow run docker.yml --ref main -f variant=cu128
sleep 10
RUN_ID=$(gh run list --workflow docker.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```

**Check:** the run shows the `cuda` job `completed/success` and the `cpu`
job `skipped` (a dispatch builds only the variant you asked for). Then
confirm the tag exists **without any credentials**, which is also the proof
that the package is public and every platform can pull it:

```bash
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:andrew-zhao12/voicebox:pull" | jq -r .token)
curl -s -H "Authorization: Bearer $TOKEN" https://ghcr.io/v2/andrew-zhao12/voicebox/tags/list
```

**Check:** `{"name":"andrew-zhao12/voicebox","tags":["main-cpu","main-cu128"]}`.

**If not:** a 401 or 403 here means the package is private (GitHub →
your profile → Packages → `voicebox` → Package settings → Change visibility).
A missing `main-cu128` with a green run means the run was dispatched from a
branch other than `main` (the tag rules only fire on `main`). A red `cuda`
job is almost always the runner's disk: open the run log; the workflow
already frees the usual directories, and a retry (`gh run rerun "$RUN_ID"`)
often succeeds.

Cloud Run refuses an image whose single layer exceeds 9.9 GB when it comes
from an external registry, so record the layer sizes once:

```bash
docker manifest inspect ghcr.io/andrew-zhao12/voicebox:main-cu128 > /tmp/cu128.json
DIGEST=$(jq -r '(.manifests // [])[] | select(.platform.architecture=="amd64" and .platform.os=="linux") | .digest' /tmp/cu128.json)
curl -s -H "Authorization: Bearer $TOKEN" \
  -H 'Accept: application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
  "https://ghcr.io/v2/andrew-zhao12/voicebox/manifests/${DIGEST:-main-cu128}" \
  | jq '[.layers[].size] | {layers: length, largest_gb: (max/1e9*100|round/100), total_gb: (add/1e9*100|round/100)}'
```

**Check:** `largest_gb` under 9.9; on 2026-09-29 the image had 14 layers, the
largest 4.67 GB and 4.89 GB in total (compressed). **If not:** the image
must be split (smaller `RUN` steps in the `Dockerfile`) before Cloud Run
can use it; ECS and Kubernetes do not care.

## 2. Keys: the file the replicas mount and the two keys your apps use

Document 1 created `./secrets/api_keys.json` with the ids `myapp` (default
client limits) and `loadtest` (no per-minute limits, for the measurements).
Confirm, and re-create them if the file is missing:

```bash
backend/venv/bin/python -m backend.keys list --data-dir ./secrets
```

**Check:** two lines, `myapp  client  file ...` and `loadtest  client  file
...`, and no plaintext anywhere (the file holds SHA-256 digests). If the
file is missing, create the keys now and keep what they print; they are
printed once:

```bash
CLIENT_KEY=$(backend/venv/bin/python -m backend.keys create --id myapp --role client --data-dir ./secrets)
LOAD_KEY=$(backend/venv/bin/python -m backend.keys create --id loadtest --role client --data-dir ./secrets \
  --limit inference=unlimited --limit requests=unlimited --limit tts_chars=unlimited --limit max_pending_jobs=unlimited)
printf 'export CLIENT_KEY=%s\nexport LOAD_KEY=%s\n' "$CLIENT_KEY" "$LOAD_KEY" >> ~/.voicebox-cloud.env
```

Whatever happens later, `source ~/.voicebox-cloud.env` must define both:

```bash
source ~/.voicebox-cloud.env && echo "${CLIENT_KEY:0:4}... ${LOAD_KEY:0:4}..."
```

**Check:** `vbx_... vbx_...`. **If not:** the keys were lost; revoke and
re-create (`python -m backend.keys revoke --id myapp --data-dir ./secrets`,
then `create` again); the store re-reads the file, nothing else changes.

## 3. The admin key and the media-token secret

Each platform stores these as secrets. Generate them once so you know them:

```bash
ADMIN_KEY=$(openssl rand -base64 32)
MEDIA_SECRET=$(openssl rand -base64 48)
printf 'export ADMIN_KEY=%s\nexport MEDIA_SECRET=%s\n' "$ADMIN_KEY" "$MEDIA_SECRET" >> ~/.voicebox-cloud.env
```

**Check:** `source ~/.voicebox-cloud.env; echo ${#ADMIN_KEY} ${#MEDIA_SECRET}`
prints `44 64`. The media secret must be at least 32 characters or the
server ignores it with a warning; it is what lets a media token issued by
one replica verify on another (browsers use `?token=` for the live
transcription socket and the SSE status stream).

## 4. The voice seed

`./seed` holds the two bundles document 1 exported: `Narrator` (a Kokoro
preset, 445 bytes) and `CloneTest` (a cloned voice with one 9 s sample,
about 330 KB). Together they are well under the 1 MiB a Kubernetes
ConfigMap can hold, and `CloneTest` is what the cloning engines in
[document 6](06-test-on-the-cloud.md) need.

```bash
ls -l seed && du -sh seed
```

**Check:** both `.voicebox.zip` files, total under 1 MiB. **If not:** export
again from the local server's data directory
(`backend/venv/bin/python -m backend.voices export ./seed --data-dir data`).
Preset voices need no seed at all: `af_heart`, `Ryan` and the rest are
accepted by `/v1/audio/speech` on any replica.

## 5. Models

The recipes preload `kokoro,whisper-turbo`. For the engine measurements add
`qwen-tts-1.7B qwen-custom-voice-1.7B luxtts chatterbox-turbo tada-1b` (the
names are the ones `backend/venv/bin/python -m backend.preload --list`
prints). Two rules:

- **Fill the cache on the platform with the Linux image**, never from this
  Mac: on Apple Silicon `qwen-tts-1.7B` resolves to an MLX repository the
  CUDA image cannot load (`--list` shows the `mlx-community/...` id here).
  Each cloud document has its own preload step (a Cloud Run Job, the
  Kubernetes Job, a `docker run` on the ECS instance). A Mac preload is only
  correct for `kokoro` and the `whisper-*` models.
- **`HF_HUB_OFFLINE=1` on the replicas**, so a replica that finds a model
  missing fails readiness instead of downloading at boot. The readiness body
  then names it under `models.failed`.

## 6. Settings that are the same on every platform

The recipes already carry these; the table is what to look for when a
manifest is edited. Full explanations are in
[`deploy/README.md`](../../deploy/README.md).

| Setting | Value | Why |
|---------|-------|-----|
| `VOICEBOX_PRELOAD_MODELS` | the set from step 5 | readiness waits for them |
| `VOICEBOX_MODELS_DIR=/models`, `HF_HUB_OFFLINE=1` | | the filled cache, no downloads |
| `VOICEBOX_SEED_PROFILES=/seed` | | the catalog from step 4 |
| `VOICEBOX_API_KEYS_JSON=/secrets/api_keys.json` | | the file from step 2 |
| `VOICEBOX_API_KEY`, `VOICEBOX_MEDIA_TOKEN_SECRET` | secrets from step 3 | admin access; media tokens across replicas |
| `VOICEBOX_MCP_STATELESS=1` | | MCP calls work on any replica |
| `VOICEBOX_REQUIRE_GPU=1` | | a replica without a GPU never becomes ready |
| `VOICEBOX_MAX_QUEUE_DEPTH=4` | | 429 + `Retry-After` when the balancer overshoots |
| `VOICEBOX_SHUTDOWN_DELAY_S` | 5 Cloud Run, 15 ECS, 10 Kubernetes | readiness 503 before the listener closes |
| `VOICEBOX_REALTIME_MAX_SESSION_S` | 840 Cloud Run and ingress-nginx | the server ends a live session before the proxy cuts it |
| `VOICEBOX_ALLOWED_HOSTS` | the public hostname, set once it exists | foreign `Host` headers get 400; probes stay exempt |
| `VOICEBOX_LOG_FORMAT=json`, `VOICEBOX_METRICS_PORT=9464`, `VOICEBOX_RETENTION_DAYS=1`, `FORWARDED_ALLOW_IPS=*` | | logs with `request_id`, metrics for the scaler, nothing to back up, real client IPs |

Placeholders to replace in the recipe you deploy: `OWNER` → `andrew-zhao12`,
`VERSION` → `main`, plus the platform ones (`PROJECT`, `ACCOUNT`, `REGION`,
`fs-EFS`, `subnet-A`, `CHANGE-ME`, `voice.example.com`). After editing:

```bash
grep -rn -E 'OWNER|VERSION|PROJECT|ACCOUNT|CHANGE-ME|example\.com|fs-EFS|subnet-A|TG_ARN|ALB-ID' deploy/cloudrun deploy/aws deploy/k8s
```

**Check:** no line for the platform you are about to deploy (lines for the
other two are fine).

## 7. Quotas that take days, and money

Ask for these before the day you want to deploy:

| Platform | Quota | Where |
|----------|-------|-------|
| Cloud Run | L4 GPUs: 3 per region without zonal redundancy are granted at first deployment; more needs a request (`run.googleapis.com/nvidia_l4_gpu_allocation_no_zonal_redundancy`), often days | IAM & Admin → Quotas ([GPU docs](https://docs.cloud.google.com/run/docs/configuring/services/gpu)) |
| AWS | "Running On-Demand G and VT instances" vCPUs (quota code `L-DB2E81BA`; confirm the code in the console), often 0 on a new account; a g6.xlarge needs 4, two of them 8 | Service Quotas → EC2 |
| GKE | `NVIDIA_L4_GPUS` in the region, often 0 | IAM & Admin → Quotas |

Money: a minimum of one GPU replica is billed around the clock on every
platform (Cloud Run bills a min-instance with a GPU at the full rate while
idle). Each cloud document ends with a teardown; run it the same day. Check
the platform's pricing page for the instance you pick rather than any
number written here.

## Done when

- [ ] `main-cu128` is listed by the anonymous `tags/list` call and its largest layer is under 9.9 GB.
- [ ] `source ~/.voicebox-cloud.env` defines `CLIENT_KEY`, `LOAD_KEY`, `ADMIN_KEY`, `MEDIA_SECRET`.
- [ ] `./secrets/api_keys.json` lists `myapp` and `loadtest`; `./seed` holds the two bundles.
- [ ] The quota request for your platform is filed (or confirmed sufficient).
