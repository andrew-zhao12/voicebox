# Deployment recipes

Reviewed starting points for running Voicebox as a fleet of interchangeable
replicas on a managed platform. They all rely on the same three inputs and
the same probes; only the platform glue differs.

| Recipe | Platform | GPU | Scale to zero | Scales on |
|--------|----------|-----|---------------|-----------|
| [`k8s/`](k8s/) | Any Kubernetes (GKE, AKS, EKS, on-prem) | node pool | node pools can | KEDA on `voicebox_queue_pending_jobs` (or an HPA) |
| [`cloudrun/`](cloudrun/) | Google Cloud Run | NVIDIA L4 | yes | request concurrency per instance |
| [`azure/`](azure/) | Azure Container Apps | serverless T4 / A100 | yes | HTTP concurrency per replica (KEDA) |
| [`aws/`](aws/) | Amazon ECS on EC2 (Fargate for CPU-only) | g5 / g6 instances | keep min 1 | ALB requests per target |

The local stand-in is `docker-compose.fleet.yml` at the repository root:
two replicas behind Caddy with the same inputs, checked by
`scripts/fleet-check.sh`.

## What every replica needs

1. **Voices**: a directory of `.voicebox.zip` bundles
   (`python -m backend.voices export ./seed`) mounted at `/seed` with
   `VOICEBOX_SEED_PROFILES=/seed`. Applied at boot, idempotent by name;
   `GET /health/ready` waits for it. Preset voices (`af_heart`, `Ryan`, ...)
   need no seed at all.
2. **Keys**: `api_keys.json` (`python -m backend.keys create --id myapp
   --role client --data-dir ./secrets`) mounted as a file and named by
   `VOICEBOX_API_KEYS_JSON`; the admin key as the `VOICEBOX_API_KEY` secret.
   The file holds SHA-256 digests only, never plaintext keys.
3. **Models**: either baked into the image (`docker build --build-arg
   VOICEBOX_BAKE_MODELS=kokoro,whisper-turbo`) or on a volume filled once
   with `python -m backend.preload ...` and mounted at `/models`
   (`VOICEBOX_MODELS_DIR=/models`, `HF_HUB_OFFLINE=1` so a replica never
   downloads).

## Settings that are the same everywhere

| Setting | Value | Why |
|---------|-------|-----|
| Liveness probe | `GET /health` every 30 s | process alive |
| Readiness / startup probe | `GET /health/ready` every 5 s, up to 5 min at start | models resident, seed applied, not draining |
| `VOICEBOX_SHUTDOWN_DELAY_S` | 10 (15 on ECS, 5 on Cloud Run) | after SIGTERM, readiness answers 503 but requests are still served, so the load balancer stops routing before the listener closes |
| Termination grace | 90 s | shutdown delay + 40 s for open streams + `VOICEBOX_DRAIN_TIMEOUT_S` (30) + margin |
| Scaling target | 2 concurrent requests per replica (4 with `VOICEBOX_GENERATION_WORKERS=2`) | one job runs, one waits per lane |
| `VOICEBOX_MAX_QUEUE_DEPTH` | scaling target + 2 | the replica answers 503 + `Retry-After` only when the platform overshoots; OpenAI SDKs retry 503 |
| Request timeout at the load balancer | at least 600 s where configurable | long generations stream for minutes; Azure's fixed 240 s means capping `tts_chars` per key |
| Minimum replicas | 1 | cold start = image pull + model load; 0 only when that latency is acceptable |
| Maximum replicas | your GPU quota | Cloud Run gives 3 L4 per region by default |
| `VOICEBOX_RETENTION_DAYS` | 1 | replicas keep nothing worth backing up |
| `VOICEBOX_LOG_FORMAT` | `json` | `request_id` and `key_id` on every line |
| `VOICEBOX_METRICS_PORT` | 9464 | scraped by a sidecar or the platform agent; never exposed publicly |
| `VOICEBOX_REQUIRE_GPU` | `1` on GPU platforms | a replica scheduled without a GPU never becomes ready |
| `VOICEBOX_MCP_STATELESS` | `1` | MCP tool calls carry no session, so any replica can answer them |
| `FORWARDED_ALLOW_IPS` | `*` behind the platform's load balancer | per-IP limits see the real client |

## What stays single-replica

`POST /generate` with `GET /generate/{id}/status` and `GET /audio/{id}`,
`POST /speak` and the MCP `speak` tool (they queue a generation and return
an id to poll), history, stories and the admin UI read state that lives in
one replica's SQLite. Route those to one replica (session affinity, or a
separate single-replica service with the same image) or keep them for the
desktop app. The `/v1` API, `/generate/stream` and the MCP `transcribe`
tool are stateless and scale; run MCP with `VOICEBOX_MCP_STATELESS=1` so
tool calls do not depend on a session held by one replica.

Rate limits and queue caps are per replica: a key's budget is multiplied by
the number of replicas it reaches. Set per-key limits with the maximum
replica count in mind when a fleet-wide budget matters.

## Verifying a deployment

```bash
scripts/fleet-check.sh https://voice.example.com "$CLIENT_KEY" --rounds 6
scripts/load_test.py --url https://voice.example.com --key "$CLIENT_KEY" --voice af_heart --model kokoro --concurrency 4 --requests 20
```

Then watch `voicebox_queue_pending_jobs`, the 503 rate and
`voicebox_stream_first_chunk_seconds` while the load test runs; the scaling
target is right when new replicas appear before requests queue for more
than a few seconds.
