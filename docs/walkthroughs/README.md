# Walkthroughs

Step-by-step runbooks for one person at one keyboard: test Voicebox on this
machine, put it on a cloud, and test it there. They are written to be
followed top to bottom, with a check after every step, and they name the
exact commands, files and outputs of this repository at the time of writing
(2026-09-29). The reference pages they lean on are
[Deployment](../content/docs/overview/deployment.mdx),
[Scaling](../content/docs/overview/scaling.mdx),
[API reference](../content/docs/overview/api-reference.mdx) and the recipes
under [`deploy/`](../../deploy/README.md); the walkthroughs sequence those
pages, they do not replace them.

| Order | Document | What you have at the end | Time | Money |
|-------|----------|--------------------------|------|-------|
| 1 | [Test locally](01-test-locally.md) | Every feature exercised on this Mac, from the test suite to a two-replica fleet behind Caddy; the `./seed` and `./secrets` inputs the cloud documents reuse | ~2 h | none |
| 2 | [Cloud prep](02-cloud-prep.md) | The CUDA image published, the keys and secrets generated once, the settings and quotas every platform needs | ~45 min plus a 20–40 min image build you only wait for | none |
| 3 | [Deploy to GCP Cloud Run](03-deploy-gcp-cloud-run.md) | One L4 instance behind a `run.app` URL, scaling 1–3 | ~1.5 h | a GPU instance while it runs |
| 4 | [Deploy to AWS ECS](04-deploy-aws-ecs.md) | ECS tasks on g6 instances behind an Application Load Balancer, scaling 1–4 | ~3 h the first time | GPU instances, the ALB and EFS while they run |
| 5 | [Deploy to Kubernetes with KEDA](05-deploy-kubernetes-keda.md) | A GPU node pool, TLS ingress, Prometheus and KEDA scaling on queue depth | ~2 h | the cluster fee and GPU nodes while they run |
| 6 | [Test on the cloud](06-test-on-the-cloud.md) | The functional checks through the public URL, the per-engine GPU numbers, autoscaling observed, and everything torn down | ~1 h per platform plus the measurement runs | the deployment above while you test |

Do 1 first: it produces files the others mount, and it is where you learn
what a healthy server looks like. Do 2 once. Then pick 3, 4 or 5 (you only
need one cloud) and finish with 6.

## How to read a step

Every step has the same shape:

1. One sentence saying what the step is for.
2. A command block. Commands run from the repository root unless the step
   says otherwise, and `backend/venv/bin/python` is spelled out because the
   system `python3` on this machine has none of the packages.
3. **Check:** what proves the step worked, quoting the real output where
   the program prints a fixed line.
4. **If not:** the likely cause and the command or log that shows it.

Values you collect along the way (keys, URLs, ids) are exported once and
appended to `~/.voicebox-cloud.env`, a file outside the repository that
every later block loads with `source ~/.voicebox-cloud.env`. Create it once:

```bash
touch ~/.voicebox-cloud.env && chmod 600 ~/.voicebox-cloud.env
```

Placeholders are `UPPER_CASE` shell variables, never angle brackets inside a
command, so a block either runs or fails loudly on an unset variable.

## What was actually run

Document 1 was executed end to end on this Mac on 2026-09-29 and its
expected outputs are copied from that run. Documents 3, 4 and 5 have **not**
been run on an account yet: every vendor-side claim carries the page it was
checked against, and anything that could not be checked without an account
says "confirm at run time". Treat the first run of a cloud document as a
test of the document too, and fix it as you go.

## What cannot be tested locally

- **GPU engines and their throughput.** This Mac runs the MLX and CPU paths.
  Kokoro is quick everywhere; Qwen, LuxTTS, Chatterbox and TADA on an L4 are
  what document 6 measures, and those numbers decide whether
  `VOICEBOX_ENGINE_CONCURRENCY` is worth turning on.
- **Autoscaling.** The local fleet has two fixed replicas; scale-out, scale-in
  and their timings only exist on a platform.
- **A real load balancer and DNS.** Caddy in `docker-compose.fleet.yml` is
  the stand-in; it proxies WebSockets and honours readiness like the cloud
  balancers do, but timeouts and TLS are per platform.
- **Cost.** Only a bill shows it. Every cloud document ends with a teardown
  and document 6 repeats it; a minimum of one GPU replica is billed around
  the clock on every platform, so tear down the same day.

Local cloud emulators (LocalStack, Floci) do not help with any of this:
Voicebox calls no cloud API (SQLite and local disk only) and nothing
emulates a GPU. They may become useful for the deferred shared-state phase
(object storage and Postgres in CI), not for these walkthroughs.

## Where the results go

- The engine throughput table from document 6 belongs in
  [`remote-mode.mdx`](../content/docs/overview/remote-mode.mdx), which
  currently carries only the Kokoro figure.
- The decision on `VOICEBOX_ENGINE_CONCURRENCY` and anything a cloud run
  taught you goes into [`PROJECT_STATUS.md`](../PROJECT_STATUS.md).
- A recipe under `deploy/` that a walkthrough had to correct gets the fix in
  the recipe, not a note in the walkthrough.
