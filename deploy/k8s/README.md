# Voicebox on Kubernetes

Works on GKE, AKS and EKS (and any cluster with the NVIDIA device plugin).
Replace `ghcr.io/OWNER/voicebox:VERSION-cu128` in the manifests with your
image; use the `-cpu` tag and drop the GPU lines for a CPU-only Kokoro
service.

## 1. Inputs

```bash
kubectl apply -f deploy/k8s/namespace.yaml
# Keys: the admin key, an optional shared media-token secret, and the
# api_keys.json produced by `python -m backend.keys create ... --data-dir ./secrets`
CLIENT_KEY=$(python -m backend.keys create --id myapp --role client --data-dir ./secrets)   # printed once; keep it
ADMIN_KEY=$(openssl rand -base64 32)   # keep it: the admin key of the fleet
kubectl -n voicebox create secret generic voicebox-keys \
  --from-literal=VOICEBOX_API_KEY="$ADMIN_KEY" \
  --from-literal=VOICEBOX_MEDIA_TOKEN_SECRET="$(openssl rand -base64 48)" \
  --from-file=api_keys.json=./secrets/api_keys.json
# Voices (small catalogs; ConfigMaps hold 1 MiB):
kubectl -n voicebox create configmap voicebox-seed --from-file=./seed
# Models: an RWX volume filled once.
kubectl apply -f deploy/k8s/configmap.yaml -f deploy/k8s/pvc-models.yaml -f deploy/k8s/job-preload.yaml
kubectl -n voicebox wait --for=condition=complete job/voicebox-preload --timeout=30m
```

A catalog with cloned voices is larger than a ConfigMap allows: put the
bundles on a second RWX PVC (a Job with `kubectl cp`, or your bucket CSI
driver) and mount it at `/seed` instead.

## 2. GPU nodes

| Cloud | Node pool | Autoscaling |
|-------|-----------|-------------|
| GKE | `--accelerator type=nvidia-l4,count=1 --machine-type g2-standard-8`; the driver installs automatically (`gpu-driver-version=default`) | node auto-provisioning or a min 0 / max N pool |
| EKS | Karpenter `NodePool` with `karpenter.k8s.aws/instance-gpu-manufacturer: nvidia`, or a managed node group on the NVIDIA AL2023 AMI; either way install the NVIDIA device plugin DaemonSet (the AMI ships the driver, not the plugin) | Karpenter consolidates to zero; a node group needs the cluster autoscaler |
| AKS | `az aks nodepool add --node-vm-size Standard_NC8as_T4_v3 --enable-cluster-autoscaler --min-count 0 --max-count 4` | cluster autoscaler |

Keep the `nodeSelector` line for your cloud in `deployment.yaml` and delete
the others; `tolerations` covers the usual `nvidia.com/gpu` taint.

## 3. Deploy and scale

```bash
# TLS: cert-manager with the ClusterIssuer the Ingress names (edit the email first).
kubectl apply -f deploy/k8s/clusterissuer.yaml
kubectl apply -f deploy/k8s/deployment.yaml -f deploy/k8s/service.yaml -f deploy/k8s/ingress.yaml -f deploy/k8s/pdb.yaml
kubectl -n voicebox rollout status deploy/voicebox
# Metrics + autoscaling (Prometheus Operator and KEDA installed):
kubectl apply -f deploy/k8s/servicemonitor.yaml -f deploy/k8s/keda-scaledobject.yaml
```

Without KEDA, an HPA on CPU is a poor proxy for GPU work; prefer the
Prometheus trigger, or the `http` trigger of the KEDA HTTP add-on with
`targetPendingRequests: 2`.

## 4. Verify

```bash
kubectl -n voicebox port-forward svc/voicebox 17493:80 &
scripts/fleet-check.sh http://127.0.0.1:17493 "$CLIENT_KEY" --rounds 6
```

Live transcription uses a WebSocket; `ingress.yaml` already sets the 900 s
read and send timeouts ingress-nginx needs, and a cloud-managed ingress
needs its backend timeout raised the same way.

Rolling updates keep traffic on ready pods only: the new pod becomes ready
after its models load and its seed is applied, the old pod drains for up to
90 s. `pdb.yaml` keeps one pod available when the cluster autoscaler drains
a node. Nothing in a replica needs a backup.
