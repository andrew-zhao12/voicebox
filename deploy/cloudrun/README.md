# Voicebox on Cloud Run (GPU)

Cloud Run gives request-based autoscaling, scale to zero and managed TLS
with one NVIDIA L4 (24 GB) per instance; new projects get 3 L4 per region.
Everything below is `gcloud`; the service definition is `service.yaml`.

## 1. Inputs

```bash
export PROJECT=my-project REGION=us-central1
gcloud config set project $PROJECT

# Models: fill a bucket once from a machine that can download them.
VOICEBOX_MODELS_DIR=./models-cache python -m backend.preload kokoro whisper-turbo
gsutil mb -l $REGION gs://$PROJECT-voicebox-models
gsutil -m rsync -r ./models-cache gs://$PROJECT-voicebox-models

# Voices: the exported catalog.
python -m backend.voices export ./seed
gsutil mb -l $REGION gs://$PROJECT-voicebox-seed
gsutil -m rsync -r ./seed gs://$PROJECT-voicebox-seed

# Keys: api_keys.json (hashes only) and the admin key as secrets.
python -m backend.keys create --id myapp --role client --data-dir ./secrets   # prints the client key once
gcloud secrets create voicebox-keys-json --data-file=./secrets/api_keys.json
openssl rand -base64 32 | gcloud secrets create voicebox-admin-key --data-file=-

# A service account that may read both buckets and both secrets.
gcloud iam service-accounts create voicebox-sa
for b in models seed; do gsutil iam ch serviceAccount:voicebox-sa@$PROJECT.iam.gserviceaccount.com:objectViewer gs://$PROJECT-voicebox-$b; done
for s in voicebox-keys-json voicebox-admin-key; do gcloud secrets add-iam-policy-binding $s --member=serviceAccount:voicebox-sa@$PROJECT.iam.gserviceaccount.com --role=roles/secretmanager.secretAccessor; done
```

Alternatively bake a small model set into the image
(`docker build --build-arg VOICEBOX_BAKE_MODELS=kokoro,whisper-turbo ...`);
Google recommends images under 10 GB, larger sets belong in the bucket.
The models bucket is mounted read-write because some engines write lock
files into the HuggingFace cache on first use; `HF_HUB_OFFLINE=1` still
prevents downloads.

## 2. Deploy

Edit `service.yaml` (PROJECT, OWNER/VERSION of the image, bucket names),
then:

```bash
gcloud run services replace deploy/cloudrun/service.yaml --region $REGION
gcloud run services describe voicebox --region $REGION --format='value(status.url)'
```

The equivalent flags for `gcloud run deploy` are `--gpu 1 --gpu-type nvidia-l4
--cpu 8 --memory 32Gi --concurrency 2 --timeout 900 --min-instances 1
--max-instances 3 --no-cpu-throttling --no-gpu-zonal-redundancy`.

Access: the service is public (`ingress: all`) and every route needs a
Voicebox key anyway; add `--no-allow-unauthenticated` plus an ID-token
proxy only if you want Google IAM in front as well.

## 3. Scale and observe

- Cloud Run adds an instance when every instance has `containerConcurrency`
  requests in flight; a queue depth of 4 inside the replica absorbs brief
  overshoot, and the OpenAI SDKs retry the rare 503.
- Shutdown: Cloud Run stops routing to an instance before it sends SIGTERM
  and allows 10 s before SIGKILL, so the recipe keeps a short
  `VOICEBOX_SHUTDOWN_DELAY_S=5` and relies on `minScale` to avoid shutting
  down instances with long streams; scale-in happens when instances idle.
- Startup: about 5 s for the GPU instance plus the model load from the
  bucket (Kokoro and Whisper turbo load in well under a minute; larger
  models take longer, which is what `minScale: 1` hides).
- Metrics: add an OpenTelemetry collector sidecar that scrapes
  `localhost:9464` and writes to Cloud Monitoring (Managed Prometheus), or
  scrape `GET /metrics` with the admin key from outside. JSON logs land in
  Cloud Logging with `request_id` and `key_id` as fields.

WebSockets: Cloud Run proxies them; a live-transcription session is one
request, so `timeoutSeconds` (900 in the recipe) is also the longest session
and `VOICEBOX_REALTIME_MAX_SESSION_S` should stay below it.

## 4. Verify

```bash
URL=$(gcloud run services describe voicebox --region $REGION --format='value(status.url)')
scripts/fleet-check.sh "$URL" "$CLIENT_KEY" --rounds 6
scripts/load_test.py --url "$URL" --key "$CLIENT_KEY" --voice af_heart --model kokoro --concurrency 4 --requests 20
```
