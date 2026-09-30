# 3. Deploy to GCP Cloud Run with an L4

**Goal.** The `deploy/cloudrun/` recipe running: one Cloud Run service with
an NVIDIA L4 per instance, scaling between 1 and 3, reachable at a public
`https://voicebox-....run.app` URL, with the keys, seed and models from
[document 2](02-cloud-prep.md).

**Time.** About 1.5 hours the first time, most of it waiting for the model
preload job and the first instance. **Money.** One L4 instance is billed
the whole time the service has `minScale: "1"`, including while idle; step
11 shows how to park it. **You need.** Document 2 done, a Google Cloud
account with billing enabled, `gcloud` (installed in step 1). **Not run
yet.** This document has not been executed on an account; vendor facts
carry their source, and steps that could not be verified without an account
say so.

Everything below reads `~/.voicebox-cloud.env`; start each session with
`source ~/.voicebox-cloud.env`.

## 1. Install gcloud, pick the project and region, enable the APIs

```bash
brew install --cask google-cloud-sdk
gcloud init                      # log in, choose or create the project
export PROJECT=$(gcloud config get-value project) REGION=us-central1
gcloud billing projects describe "$PROJECT" --format='value(billingEnabled)'
gcloud services enable run.googleapis.com secretmanager.googleapis.com storage.googleapis.com iam.googleapis.com logging.googleapis.com
printf 'export PROJECT=%s\nexport REGION=%s\n' "$PROJECT" "$REGION" >> ~/.voicebox-cloud.env
```

**Check:** `billingEnabled` prints `True`, and `gcloud services list
--enabled | grep -E 'run|secretmanager'` shows both. L4 GPUs on Cloud Run
exist in `us-central1`, `us-east4`, `europe-west1`, `europe-west4` and
`asia-southeast1` ([GPU
docs](https://docs.cloud.google.com/run/docs/configuring/services/gpu));
the recipe assumes `us-central1`.

**If not:** `billingEnabled: False` means the project has no billing
account; attach one in the console before anything else.

## 2. Confirm the GPU quota

Three L4 instances without zonal redundancy are granted at the first
deployment, which is exactly the recipe's `maxScale: "3"`. Look at the
quota once so a scale-out does not surprise you:

- Console → IAM & Admin → Quotas → filter
  `nvidia_l4_gpu_allocation_no_zonal_redundancy` for `run.googleapis.com`.

**Check:** the limit is 3 or more in `$REGION`. **If not:** request an
increase from that page; it can take days, and until then keep `maxScale`
at the limit.

## 3. Buckets for the seed and the models

```bash
source ~/.voicebox-cloud.env
gcloud storage buckets create gs://$PROJECT-voicebox-models gs://$PROJECT-voicebox-seed --location=$REGION --uniform-bucket-level-access
gcloud storage rsync -r ./seed gs://$PROJECT-voicebox-seed
gcloud storage ls gs://$PROJECT-voicebox-seed
```

**Check:** both `.voicebox.zip` bundles are listed. **If not:** bucket names
are global; if `create` fails with a name clash, add a suffix and use the
same name in `service.yaml` later.

## 4. Secrets: the keys file, the admin key, the media secret

```bash
source ~/.voicebox-cloud.env
gcloud secrets create voicebox-keys-json --data-file=./secrets/api_keys.json
printf '%s' "$ADMIN_KEY"    | gcloud secrets create voicebox-admin-key --data-file=-
printf '%s' "$MEDIA_SECRET" | gcloud secrets create voicebox-media-token-secret --data-file=-
for s in voicebox-keys-json voicebox-admin-key voicebox-media-token-secret; do gcloud secrets versions list $s --format='value(name,state)'; done
```

**Check:** each secret has version `1` in state `enabled`. **If not:** an
empty `$ADMIN_KEY` means the env file was not sourced; a secret that already
exists needs `gcloud secrets versions add NAME --data-file=-` instead of
`create`.

## 5. The service account and what it may read

The models bucket is mounted read-write (engines write lock files into the
cache on first use), the seed bucket read-only, and the three secrets are
read at start:

```bash
source ~/.voicebox-cloud.env
gcloud iam service-accounts create voicebox-sa
SA=voicebox-sa@$PROJECT.iam.gserviceaccount.com
gcloud storage buckets add-iam-policy-binding gs://$PROJECT-voicebox-models --member=serviceAccount:$SA --role=roles/storage.objectUser
gcloud storage buckets add-iam-policy-binding gs://$PROJECT-voicebox-seed   --member=serviceAccount:$SA --role=roles/storage.objectViewer
for s in voicebox-keys-json voicebox-admin-key voicebox-media-token-secret; do
  gcloud secrets add-iam-policy-binding $s --member=serviceAccount:$SA --role=roles/secretmanager.secretAccessor
done
gcloud storage buckets get-iam-policy gs://$PROJECT-voicebox-models --format=json | jq -r '.bindings[]|select(.role=="roles/storage.objectUser").members[]'
```

**Check:** the last line prints `serviceAccount:voicebox-sa@...`. **If
not:** a freshly created service account can take a minute to be usable in
bindings; re-run the failed command.

## 6. Fill the models bucket with a Cloud Run Job (the Linux image)

The recipe's CPU image runs the same preload the server runs at boot, so the
cache layout matches the CUDA image exactly. The job writes into the bucket
through the same volume mount the service uses. Kokoro and Whisper turbo are
the readiness set; the five engines behind them are for
[document 6](06-test-on-the-cloud.md) and load on first use.

```bash
source ~/.voicebox-cloud.env
gcloud run jobs create voicebox-preload --region $REGION \
  --image ghcr.io/andrew-zhao12/voicebox:main-cpu \
  --args=python,-m,backend.preload,kokoro,whisper-turbo,qwen-tts-1.7B,qwen-custom-voice-1.7B,luxtts,chatterbox-turbo,tada-1b \
  --set-env-vars VOICEBOX_MODELS_DIR=/models,HF_HUB_OFFLINE=0,NUMBA_CACHE_DIR=/tmp/numba_cache \
  --add-volume name=models,type=cloud-storage,bucket=$PROJECT-voicebox-models \
  --add-volume-mount volume=models,mount-path=/models \
  --service-account voicebox-sa@$PROJECT.iam.gserviceaccount.com \
  --cpu 4 --memory 16Gi --task-timeout 3600 --max-retries 0
gcloud run jobs execute voicebox-preload --region $REGION --wait
gcloud storage ls gs://$PROJECT-voicebox-models/
```

**Check:** the execution ends `Succeeded` (expect 15–40 minutes: it is
downloading on the order of 15 GB and loading each model once) and the
bucket lists `models--hexgrad--Kokoro-82M/`,
`models--openai--whisper-large-v3-turbo/` and one directory per engine.
Read the job's own lines with
`gcloud logging read 'resource.type=cloud_run_job AND resource.labels.job_name=voicebox-preload' --limit 40 --format='value(textPayload)'`:
one `ready: NAME` per model.

**If not:** the volume flags are from the Cloud Run Jobs CLI reference and
could not be verified from here (**confirm at run time**; `gcloud run jobs
create --help` lists `--add-volume` and `--add-volume-mount`). `unknown
model` in the log means a name that is not in
`backend/venv/bin/python -m backend.preload --list`. An out-of-memory kill
(`Memory limit ... exceeded` in the logs) means raising `--memory` to 32Gi;
each model is loaded into RAM once to prove it works.

## 7. Prepare the service definition

Work on a copy so the recipe in the repository keeps its placeholders:

```bash
source ~/.voicebox-cloud.env
mkdir -p ~/voicebox-cloud
sed -e "s/PROJECT/$PROJECT/g" \
    -e "s#ghcr.io/OWNER/voicebox:VERSION-cu128#ghcr.io/andrew-zhao12/voicebox:main-cu128#" \
    deploy/cloudrun/service.yaml > ~/voicebox-cloud/cloudrun-service.yaml
grep -nE 'PROJECT|OWNER|VERSION' ~/voicebox-cloud/cloudrun-service.yaml
```

**Check:** the `grep` prints nothing. The copy keeps the recipe's
`containerConcurrency: 2`, `timeoutSeconds: 900`,
`VOICEBOX_REALTIME_MAX_SESSION_S: "840"` (a live-transcription session is
one request to Cloud Run and is cut at the request timeout, [WebSockets
docs](https://docs.cloud.google.com/run/docs/triggering/websockets)), the
GPU annotations and the media-token secret.

## 8. Deploy and wait for readiness

```bash
source ~/.voicebox-cloud.env
gcloud run services replace ~/voicebox-cloud/cloudrun-service.yaml --region $REGION
gcloud run services add-iam-policy-binding voicebox --region $REGION --member=allUsers --role=roles/run.invoker
URL=$(gcloud run services describe voicebox --region $REGION --format='value(status.url)')
echo "export URL=$URL" >> ~/.voicebox-cloud.env
until [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health/ready)" = 200 ]; do printf .; sleep 10; done; echo
curl -s $URL/health/ready | jq
```

The `add-iam-policy-binding` line is what makes the URL answer without a
Google identity; `ingress: all` in the manifest only allows internet
traffic, it does not grant invocation (`gcloud run deploy
--allow-unauthenticated` does both, `services replace` does neither). Every
route still needs a Voicebox key.

**Check:** the body has `"ready": true`, `models.ready` containing
`kokoro` and `whisper-turbo`, and `startup.done` containing
`seed_profiles` and `gpu`. The first instance pulls about 8 GB and loads two
models from the bucket, so expect several minutes of dots.

**If not:**

- `gcloud run services logs read voicebox --region $REGION --limit 100`
  shows the container's own lines (`Loading Kokoro-82M on cuda`, the seed
  result, `Application startup complete`).
- `startup.failed: ["gpu"]` in the readiness body: the instance has no GPU;
  the `nodeSelector` and the GPU annotations were lost in the edit, or the
  region has no L4.
- `models.failed` names a model: the bucket lacks it (step 6 did not finish)
  or the service account cannot read the bucket (step 5).
- `add-iam-policy-binding` refused with an organisation policy error: the
  org forbids `allUsers`; keep the service authenticated and put an
  ID-token proxy in front, or test through
  `gcloud run services proxy voicebox --region $REGION` (localhost only).
- The revision never becomes ready and the logs stop after the pull: the
  startup probe's 5 minutes ran out; raise `failureThreshold` in the copy
  and `replace` again.

## 9. First contact

```bash
source ~/.voicebox-cloud.env
curl -s -H "Authorization: Bearer $CLIENT_KEY" $URL/auth/whoami | jq -c .
scripts/fleet-check.sh "$URL" "$CLIENT_KEY" --rounds 2
```

**Check:** `{"key_id":"myapp","role":"client",...}` and `fleet check
passed: 2 round(s) against https://voicebox-....run.app`. The full battery
is [document 6](06-test-on-the-cloud.md).

## 10. Lock the Host header to the public name

```bash
source ~/.voicebox-cloud.env
gcloud run services update voicebox --region $REGION --update-env-vars VOICEBOX_ALLOWED_HOSTS=${URL#https://}
until [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health/ready)" = 200 ]; do sleep 10; done
curl -s -o /dev/null -w '%{http_code}\n' -H 'Host: evil.example' -H "Authorization: Bearer $CLIENT_KEY" $URL/v1/models
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $CLIENT_KEY" $URL/v1/models
```

**Check:** `400` (or `404`: Google's front end may drop a foreign `Host`
before the container sees it; either way it is not served) followed by
`200`. Probes stay exempt, so readiness kept answering during the update.

## 11. Between sessions

The GPU instance is billed at the full rate while idle
([GPU docs](https://docs.cloud.google.com/run/docs/configuring/services/gpu)).
Park the service when you stop for the day and wake it up before the next
run:

```bash
gcloud run services update voicebox --region $REGION --min-instances 0   # park: next request cold-starts in a minute or two
gcloud run services update voicebox --region $REGION --min-instances 1   # wake
```

## Teardown

```bash
source ~/.voicebox-cloud.env
gcloud run services delete voicebox --region $REGION --quiet
gcloud run jobs delete voicebox-preload --region $REGION --quiet
gcloud storage rm -r gs://$PROJECT-voicebox-models gs://$PROJECT-voicebox-seed
for s in voicebox-keys-json voicebox-admin-key voicebox-media-token-secret; do gcloud secrets delete $s --quiet; done
gcloud iam service-accounts delete voicebox-sa@$PROJECT.iam.gserviceaccount.com --quiet
gcloud run services list --region $REGION
```

**Check:** the last command lists nothing, and the billing page for the
project shows Cloud Run and Cloud Storage usage ending today.

## Done when

- [ ] `curl $URL/health/ready` answers 200 with `ready: true` and `startup.done` ⊇ `seed_profiles, gpu`.
- [ ] `fleet-check.sh` passes over the `run.app` URL with `CLIENT_KEY`.
- [ ] A foreign `Host` header is refused; `URL` is saved in `~/.voicebox-cloud.env`.
- [ ] You know the park/wake commands and the teardown is scheduled.
