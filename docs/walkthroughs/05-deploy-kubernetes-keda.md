# 5. Deploy to Kubernetes with KEDA (GKE, or EKS)

**Goal.** The `deploy/k8s/` recipe running on a cluster with a GPU node
pool: the Deployment on an L4 node, ingress-nginx with a Let's Encrypt
certificate, kube-prometheus-stack scraping the replicas, and KEDA scaling
the Deployment on `voicebox_queue_pending_jobs`.

**Time.** About 2 hours: cluster 10–15 min, GPU node and driver 5–10 min,
add-ons 10 min, certificate 2 min, models 10–30 min. **Money.** The cluster
fee, one CPU node for the add-ons, and a GPU node while they exist; the
teardown removes everything. **You need.** [Document 2](02-cloud-prep.md)
done, `kubectl` (present), `helm` (step 1), and for GKE the `gcloud` setup
from [document 3](03-deploy-gcp-cloud-run.md) step 1 (the L4 quota there is
a Cloud Run quota; GKE needs the Compute Engine `NVIDIA_L4_GPUS` quota in
the region), or for EKS the `aws` setup from [document 4](04-deploy-aws-ecs.md)
step 1 plus `eksctl`. **Not run yet.** This document has not been executed
on a cluster; the GKE and EKS facts carry their source and the rest says
"confirm at run time" where it matters.

GKE is the primary path; the EKS differences are in step 2 and the teardown.
Everything reads `~/.voicebox-cloud.env`; working copies of the manifests go
to `~/voicebox-cloud/k8s/`.

## 1. Tools

```bash
brew install helm
helm version --short
# GKE only:
gcloud components install gke-gcloud-auth-plugin
# EKS only:
brew install eksctl
```

**Check:** `helm version` prints `v3...`. `kubectl` is already installed
(`kubectl version --client`).

## 2. A cluster with a GPU node pool

### GKE

A zonal cluster keeps the bill to one CPU node; pick a zone that offers L4:

```bash
source ~/.voicebox-cloud.env
gcloud services enable container.googleapis.com
gcloud compute accelerator-types list --filter="name=nvidia-l4 AND zone:$REGION" --format='value(zone)'
ZONE=$REGION-a
gcloud container clusters create voicebox --zone $ZONE --num-nodes 1 --machine-type e2-standard-4 --release-channel regular
gcloud container node-pools create gpu --cluster voicebox --zone $ZONE \
  --machine-type g2-standard-8 --accelerator type=nvidia-l4,count=1,gpu-driver-version=default \
  --enable-autoscaling --min-nodes 0 --max-nodes 3 --num-nodes 1
gcloud container clusters get-credentials voicebox --zone $ZONE
echo "export ZONE=$ZONE" >> ~/.voicebox-cloud.env
until kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-l4 -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}' 2>/dev/null | grep -q 1; do printf .; sleep 15; done; echo
kubectl get nodes -L cloud.google.com/gke-accelerator
```

**Check:** two nodes, the `gpu` pool's node labelled `nvidia-l4`, and the
loop ended, which means the node reports `nvidia.com/gpu: 1`.
`gpu-driver-version=default` makes GKE install the driver and the device
plugin itself; the `nvidia.com/gpu` taint it adds is tolerated by any pod
that requests a GPU, which the recipe's Deployment does
([GKE GPU docs](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/gpus)).

**If not:** `ZONE_RESOURCE_POOL_EXHAUSTED` or a quota error on the node
pool: pick another zone from the list, or request `NVIDIA_L4_GPUS` in IAM &
Admin → Quotas (often 0 on a new project; days). A node that never reports
the GPU: `kubectl -n kube-system get pods -l k8s-app=nvidia-driver-installer`
shows the driver DaemonSet; give it five minutes.

### EKS

`eksctl` picks the NVIDIA-enabled AL2023 AMI for a GPU instance type, but
that AMI ships the driver and container toolkit only; the Kubernetes
device plugin has to be installed separately
([EKS accelerated AMI docs](https://docs.aws.amazon.com/eks/latest/userguide/ml-eks-optimized-ami.html)).

```bash
source ~/.voicebox-cloud.env
eksctl create cluster --name voicebox --region $AWS_REGION --nodegroup-name cpu --node-type m6i.large --nodes 1 --nodes-min 1 --nodes-max 1
eksctl create nodegroup --cluster voicebox --region $AWS_REGION --name gpu --node-type g6.xlarge --node-ami-family AmazonLinux2023 \
  --nodes 1 --nodes-min 1 --nodes-max 3 --node-labels role=gpu
PLUGIN_VERSION=v0.18.0   # the current release at github.com/NVIDIA/k8s-device-plugin/releases; confirm at run time
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/$PLUGIN_VERSION/deployments/static/nvidia-device-plugin.yml
until kubectl get nodes -l role=gpu -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}' 2>/dev/null | grep -q 1; do printf .; sleep 15; done; echo
```

**Check:** the loop ends (the g6 node reports `nvidia.com/gpu: 1`). In
`deployment.yaml` you will use the `nodeSelector` line `role: gpu` instead
of the GKE label (step 7). `--nodes-min 1` keeps it simple; scaling the
node group to zero needs the cluster autoscaler or Karpenter, which the
recipe's README mentions and this document does not install.

## 3. Add-ons with Helm

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo add jetstack https://charts.jetstack.io
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx -n ingress-nginx --create-namespace --wait
helm upgrade --install cert-manager jetstack/cert-manager -n cert-manager --create-namespace --set crds.enabled=true --wait
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace --wait \
  --set grafana.enabled=false --set alertmanager.enabled=false
helm upgrade --install keda kedacore/keda -n keda --create-namespace --wait
helm list -A --output json | jq -r '.[]|"\(.namespace) \(.name) \(.status)"'
kubectl -n monitoring get svc prometheus-kube-prometheus-prometheus -o jsonpath='{.metadata.name}:{.spec.ports[0].port}{"\n"}'
```

**Check:** four releases `deployed`, and the last line prints
`prometheus-kube-prometheus-prometheus:9090`, the address
`keda-scaledobject.yaml` points at. The release name `prometheus` matters
twice: it is in that Service name, and the recipe's ServiceMonitor carries
the label `release: prometheus` that this chart's Prometheus selects.

**If not:** a `--wait` that times out on GKE is usually the ingress
controller waiting for its external IP (a few minutes); re-run the same
command. On EKS ingress-nginx gets a classic ELB, which is fine here.

## 4. DNS: a name for the ingress

```bash
INGRESS=$(kubectl -n ingress-nginx get svc ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}{.status.loadBalancer.ingress[0].hostname}')
echo "$INGRESS"
```

**Check:** an IP (GKE) or an ELB hostname (EKS). Pick the host name:

- Your own domain: create an A record (GKE) or CNAME (EKS) for
  `voice.YOURDOMAIN` pointing at `$INGRESS`, then `HOST=voice.YOURDOMAIN`.
- No domain: `HOST=voice.$INGRESS.nip.io` on GKE resolves to that IP with no
  setup, and Let's Encrypt's HTTP-01 challenge works with it. On EKS the
  ELB has changing IPs, so nip.io only works for a short test with
  `HOST=voice.$(dig +short $INGRESS | head -1).nip.io`.

```bash
HOST=voice.$INGRESS.nip.io        # or your record
printf 'export HOST=%s\nexport URL=https://%s\n' "$HOST" "$HOST" >> ~/.voicebox-cloud.env
dig +short $HOST | head -1
```

**Check:** the IP of the ingress. Let's Encrypt must be able to reach
`http://$HOST/.well-known/acme-challenge/` from the internet, which it can
on both platforms once DNS resolves.

## 5. The certificate issuer

```bash
source ~/.voicebox-cloud.env
mkdir -p ~/voicebox-cloud/k8s
sed "s/CHANGE-ME@example.com/YOUR-EMAIL@example.com/" deploy/k8s/clusterissuer.yaml > ~/voicebox-cloud/k8s/clusterissuer.yaml
kubectl apply -f ~/voicebox-cloud/k8s/clusterissuer.yaml
kubectl get clusterissuer letsencrypt -o jsonpath='{.status.conditions[0].type}={.status.conditions[0].status}{"\n"}'
```

**Check:** `Ready=True`. (Use a real address: Let's Encrypt sends expiry
warnings there.)

## 6. Inputs: namespace, secret, seed, settings, models

```bash
source ~/.voicebox-cloud.env
kubectl apply -f deploy/k8s/namespace.yaml
kubectl -n voicebox create secret generic voicebox-keys \
  --from-literal=VOICEBOX_API_KEY="$ADMIN_KEY" \
  --from-literal=VOICEBOX_MEDIA_TOKEN_SECRET="$MEDIA_SECRET" \
  --from-file=api_keys.json=./secrets/api_keys.json
kubectl -n voicebox create configmap voicebox-seed --from-file=./seed
sed "/^  FORWARDED_ALLOW_IPS/a\\
  VOICEBOX_ALLOWED_HOSTS: \"$HOST\"
" deploy/k8s/configmap.yaml > ~/voicebox-cloud/k8s/configmap.yaml
kubectl apply -f ~/voicebox-cloud/k8s/configmap.yaml
kubectl -n voicebox get configmap voicebox-config -o jsonpath='{.data.VOICEBOX_ALLOWED_HOSTS} {.data.VOICEBOX_REALTIME_MAX_SESSION_S}{"\n"}'
```

**Check:** `voice.... 840`. The seed ConfigMap holds the two bundles
(under 1 MiB together; a bigger catalog goes on a volume, as the recipe's
README says).

Models, one of two ways:

- **Option A, for this test: each pod downloads at boot.** No shared
  volume, no Job; a new pod takes a minute or two longer to become ready
  (about 2 GB for the readiness set). Applied in step 7 with two patches.
- **Option B, the recipe: a ReadWriteMany volume filled once by the Job.**
  Needs an RWX StorageClass: on GKE the Filestore CSI driver
  (`gcloud container clusters update voicebox --zone $ZONE --update-addons=GcpFilestoreCsiDriver=ENABLED`,
  class `standard-rwx`; a Filestore instance has a large minimum size, so
  check its price first), on EKS the EFS CSI add-on and an `efs-sc` class.
  Then:

  ```bash
  sed -e "s/CHANGE-ME-rwx/standard-rwx/" deploy/k8s/pvc-models.yaml > ~/voicebox-cloud/k8s/pvc-models.yaml
  sed -e "s#ghcr.io/OWNER/voicebox:VERSION-cpu#ghcr.io/andrew-zhao12/voicebox:main-cpu#" deploy/k8s/job-preload.yaml > ~/voicebox-cloud/k8s/job-preload.yaml
  kubectl apply -f ~/voicebox-cloud/k8s/pvc-models.yaml -f ~/voicebox-cloud/k8s/job-preload.yaml
  kubectl -n voicebox wait --for=condition=complete job/voicebox-preload --timeout=30m
  kubectl -n voicebox logs job/voicebox-preload | grep 'ready:'
  ```

  **Check:** one `ready: NAME` line per model. **If not:** a PVC stuck in
  `Pending` means the class does not exist or is not RWX; an `OOMKilled`
  Job needs bigger limits in the copy for the larger engines.

## 7. Deploy

```bash
source ~/.voicebox-cloud.env
sed -e "s#ghcr.io/OWNER/voicebox:VERSION-cu128#ghcr.io/andrew-zhao12/voicebox:main-cu128#" deploy/k8s/deployment.yaml > ~/voicebox-cloud/k8s/deployment.yaml
# EKS only: swap the nodeSelector line
# sed -i '' -e 's/^        cloud.google.com\/gke-accelerator: nvidia-l4.*/        role: gpu/' ~/voicebox-cloud/k8s/deployment.yaml
sed -e "s/voice.example.com/$HOST/g" deploy/k8s/ingress.yaml > ~/voicebox-cloud/k8s/ingress.yaml
grep -nE 'OWNER|VERSION|example.com' ~/voicebox-cloud/k8s/*.yaml
kubectl apply -f ~/voicebox-cloud/k8s/deployment.yaml -f deploy/k8s/service.yaml -f ~/voicebox-cloud/k8s/ingress.yaml -f deploy/k8s/pdb.yaml
```

Option A only, right after the apply (the pod is Pending on the missing
PVC until the first patch lands):

```bash
kubectl -n voicebox patch deploy voicebox --type json \
  -p '[{"op":"replace","path":"/spec/template/spec/volumes/0","value":{"name":"models","emptyDir":{"sizeLimit":"40Gi"}}}]'
kubectl -n voicebox patch configmap voicebox-config --type merge -p '{"data":{"HF_HUB_OFFLINE":"0"}}'
kubectl -n voicebox rollout restart deploy/voicebox
```

Then, for both options:

```bash
kubectl -n voicebox rollout status deploy/voicebox --timeout=15m
kubectl -n voicebox get certificate voicebox-tls -o jsonpath='{.status.conditions[0].type}={.status.conditions[0].status}{"\n"}'
curl -s $URL/health/ready | jq
```

**Check:** `deployment "voicebox" successfully rolled out`, the certificate
`Ready=True`, and the readiness body over HTTPS with `"ready": true`,
`models.ready` containing `kokoro` and `whisper-turbo`, and `startup.done`
containing `seed_profiles` and `gpu`.

**If not:**

- `kubectl -n voicebox describe pod -l app=voicebox | tail -20`:
  `Insufficient nvidia.com/gpu` means the driver or device plugin is not
  ready yet (step 2), `ImagePullBackOff` means the package went private.
- `kubectl -n voicebox logs deploy/voicebox` for the container's own lines.
- Readiness from inside the cluster while DNS or the certificate is not
  ready: `kubectl -n voicebox port-forward svc/voicebox 17493:80 &` then
  `curl -s http://127.0.0.1:17493/health/ready | jq`.
- The certificate stays `False`: `kubectl -n voicebox describe challenge`
  shows why Let's Encrypt could not reach `http://$HOST/...` (DNS not
  propagated, or the host does not point at the ingress).

## 8. Metrics and the scaler

```bash
kubectl apply -f deploy/k8s/servicemonitor.yaml -f deploy/k8s/keda-scaledobject.yaml
sleep 60
kubectl -n voicebox get scaledobject voicebox
kubectl -n voicebox get hpa
kubectl -n monitoring port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 >/dev/null 2>&1 &
sleep 3; curl -s 'http://127.0.0.1:9090/api/v1/query?query=sum(voicebox_queue_pending_jobs)' | jq -c '.data.result[0].value'
kill %1
```

**Check:** the ScaledObject shows `READY True` and `ACTIVE False` (no
queue), the HPA `keda-hpa-voicebox` reads `0/2 (avg)` (the HPA target
column shows the value once Prometheus has scraped for a minute), and the
query returns `["<timestamp>","0"]`. KEDA's `ignoreNullValues` default means
an empty query result is not an error, so a wrong `serverAddress` shows as
`READY False` with an event, not as a crash
([KEDA Prometheus scaler](https://keda.sh/docs/latest/scalers/prometheus/)).

**If not:** `kubectl -n voicebox describe scaledobject voicebox` prints the
scaler's error; the usual ones are the Prometheus Service name (step 3) and
the ServiceMonitor not being selected (`kubectl -n voicebox get
servicemonitor voicebox -o jsonpath='{.metadata.labels}'` must show
`release: prometheus`).

## 9. First contact

```bash
source ~/.voicebox-cloud.env
curl -s -H "Authorization: Bearer $CLIENT_KEY" $URL/auth/whoami | jq -c .
scripts/fleet-check.sh "$URL" "$CLIENT_KEY" --rounds 2
curl -s -o /dev/null -w '%{http_code}\n' -H 'Host: evil.example' -H "Authorization: Bearer $CLIENT_KEY" $URL/v1/models
```

**Check:** `{"key_id":"myapp",...}`, `fleet check passed: 2 round(s)
against https://voice...`, then `404` (ingress-nginx has no rule for that
host, so the request never reaches the container; the allowlist inside
answers `400` for anything that does). The full battery is
[document 6](06-test-on-the-cloud.md).

## Teardown

```bash
source ~/.voicebox-cloud.env
kubectl delete ns voicebox
helm uninstall keda -n keda; helm uninstall prometheus -n monitoring; helm uninstall cert-manager -n cert-manager; helm uninstall ingress-nginx -n ingress-nginx
# GKE:
gcloud container clusters delete voicebox --zone $ZONE --quiet
gcloud compute forwarding-rules list --format='value(name)'; gcloud compute disks list --format='value(name)'
# EKS:
eksctl delete cluster --name voicebox --region $AWS_REGION
aws elb describe-load-balancers --query 'LoadBalancerDescriptions[].LoadBalancerName' --output text
```

**Check:** the cluster is gone and the two listing commands print nothing
(a forwarding rule or classic ELB left behind belongs to an ingress that
was deleted after its controller; delete it by hand). If you created a
Filestore or EFS instance for Option B, delete it too; it is billed on its
own.

## Done when

- [ ] The GPU node reports `nvidia.com/gpu: 1` and the Deployment rolled out on it.
- [ ] `curl $URL/health/ready` over HTTPS answers 200 with `startup.done` ⊇ `seed_profiles, gpu`.
- [ ] `fleet-check.sh` passes; the ScaledObject is `READY True`.
- [ ] `~/.voicebox-cloud.env` holds `HOST`, `URL` and `ZONE` for the teardown.
