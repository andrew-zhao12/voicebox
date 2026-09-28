# Voicebox on Azure Container Apps (serverless GPU)

Container Apps runs the image on a scale-to-zero, per-second-billed
NVIDIA T4 (16 GB, enough for Kokoro, Whisper and the 0.6B Qwen) or A100
(for the larger engines), with KEDA scaling rules and managed TLS. The
one hard limit is the HTTP ingress request timeout of 240 s: keep a
single speech request well inside it (cap `tts_chars` per key, or split
long texts in the application) and transcribe files shorter than a few
minutes per call.

## 1. Environment and GPU profile

```bash
export RG=voicebox-rg LOC=westus3
az group create -n $RG -l $LOC
az containerapp env create -n voicebox-env -g $RG -l $LOC --enable-workload-profiles
az containerapp env workload-profile add -n voicebox-env -g $RG \
  --workload-profile-name gpu-t4 --workload-profile-type Consumption-GPU-NC8as-T4
```

Quota for serverless GPUs is enabled by default for pay-as-you-go and EA
subscriptions in the supported regions; check the environment's Quota page
if the profile fails to add.

## 2. Inputs

```bash
# Storage account with two file shares: models (filled once) and seed.
az storage account create -n voiceboxstore -g $RG -l $LOC --sku Premium_LRS --kind FileStorage
for share in models seed; do az storage share-rm create --storage-account voiceboxstore -n $share --quota 100; done
KEY=$(az storage account keys list -n voiceboxstore -g $RG --query '[0].value' -o tsv)
for share in models seed; do
  az containerapp env storage set -n voicebox-env -g $RG --storage-name voicebox-$share \
    --azure-file-account-name voiceboxstore --azure-file-account-key "$KEY" \
    --azure-file-share-name $share --access-mode ReadWrite
done

# Fill the shares from a machine with the models and the catalog.
VOICEBOX_MODELS_DIR=./models-cache python -m backend.preload kokoro whisper-turbo
python -m backend.voices export ./seed
az storage file upload-batch --account-name voiceboxstore --account-key "$KEY" -d models -s ./models-cache
az storage file upload-batch --account-name voiceboxstore --account-key "$KEY" -d seed -s ./seed

# Keys.
python -m backend.keys create --id myapp --role client --data-dir ./secrets   # prints the client key once
```

Put the admin key, a media-token secret and the content of
`./secrets/api_keys.json` into the `secrets:` block of `containerapp.yaml`
(or pass them with `az containerapp secret set` after creation; a Key Vault
reference works too). The secret volume mounts each secret as a file, so
`VOICEBOX_API_KEYS_JSON=/secrets/api-keys-json`.

Cold starts: host the image in a Premium Azure Container Registry with
artifact streaming, and keep the models on the share rather than in the
image.

## 3. Deploy and scale

```bash
az containerapp create -g $RG --yaml deploy/azure/containerapp.yaml
az containerapp show -n voicebox -g $RG --query properties.configuration.ingress.fqdn -o tsv
```

The `http` scale rule adds a replica when each replica has two requests in
flight; `minReplicas: 1` avoids the cold start, `maxReplicas` is bounded by
the GPU quota. For a queue-depth signal instead, replace the rule with a
KEDA `prometheus` rule against Azure Monitor managed Prometheus scraping
port 9464 (`query: sum(voicebox_queue_pending_jobs)`, `threshold: "2"`).

## 4. Verify

```bash
URL=https://$(az containerapp show -n voicebox -g $RG --query properties.configuration.ingress.fqdn -o tsv)
scripts/fleet-check.sh "$URL" "$CLIENT_KEY" --rounds 6
```
