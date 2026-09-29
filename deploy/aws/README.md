# Voicebox on AWS (ECS on EC2 with GPUs, or Fargate for CPU-only)

AWS has no serverless GPU containers, so the recipe is ECS tasks on a GPU
Auto Scaling group (g6.xlarge: one L4 24 GB, 4 vCPU, 16 GiB; g5.xlarge for
an A10G; g6e.xlarge for an L40S 48 GB) behind an Application Load Balancer,
with ECS Service Auto Scaling on requests per target and a capacity provider
that adds instances as tasks need placement. For Kokoro-only workloads the
same task definition runs on Fargate without a GPU.

## 1. Inputs on EFS

The task definition mounts one EFS file system with three directories:
`/models` (filled once), `/seed` (the exported catalog) and `/secrets`
(`api_keys.json`, which holds SHA-256 digests only). Fill it from any
instance or a one-off task in the same VPC:

```bash
sudo mount -t efs -o tls fs-EFS:/ /mnt/efs
VOICEBOX_MODELS_DIR=/mnt/efs/models python -m backend.preload kokoro whisper-turbo
python -m backend.voices export /mnt/efs/seed
python -m backend.keys create --id myapp --role client --data-dir /mnt/efs/secrets   # prints the client key once
aws secretsmanager create-secret --name voicebox/admin-key --secret-string "$(openssl rand -base64 32)"
aws secretsmanager create-secret --name voicebox/media-token-secret --secret-string "$(openssl rand -base64 32)"
```

EFS is fine for a few gigabytes of model files read once per boot; for
faster loads keep the models baked in the image (`VOICEBOX_BAKE_MODELS`)
or on an instance-local NVMe warmed by the user data script.

## 2. Cluster, GPU instances, capacity provider

```bash
export REGION=us-east-1
aws ecs create-cluster --cluster-name voicebox
AMI=$(aws ssm get-parameters --names /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended --region $REGION --query 'Parameters[0].Value' --output text | python3 -c 'import json,sys; print(json.load(sys.stdin)["image_id"])')
# Launch template: the GPU-optimized AMI, ECS_ENABLE_GPU_SUPPORT=true and the cluster name in user data.
aws ec2 create-launch-template --launch-template-name voicebox-gpu --launch-template-data "{
  \"ImageId\": \"$AMI\", \"InstanceType\": \"g6.xlarge\",
  \"IamInstanceProfile\": {\"Name\": \"ecsInstanceRole\"},
  \"UserData\": \"$(printf '#!/bin/bash\necho ECS_CLUSTER=voicebox >> /etc/ecs/ecs.config\necho ECS_ENABLE_GPU_SUPPORT=true >> /etc/ecs/ecs.config\n' | base64 -w0)\"
}"
aws autoscaling create-auto-scaling-group --auto-scaling-group-name voicebox-gpu \
  --launch-template LaunchTemplateName=voicebox-gpu --min-size 1 --max-size 4 --desired-capacity 1 \
  --vpc-zone-identifier "subnet-A,subnet-B" --new-instances-protected-from-scale-in
ASG_ARN=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names voicebox-gpu --query 'AutoScalingGroups[0].AutoScalingGroupARN' --output text)
aws ecs create-capacity-provider --name voicebox-gpu --auto-scaling-group-provider "autoScalingGroupArn=$ASG_ARN,managedScaling={status=ENABLED,targetCapacity=100},managedTerminationProtection=ENABLED"
aws ecs put-cluster-capacity-providers --cluster voicebox --capacity-providers voicebox-gpu --default-capacity-provider-strategy capacityProvider=voicebox-gpu,weight=1
```

Managed scaling adds a g6 instance when a task cannot be placed and
removes idle ones; a new GPU instance takes a few minutes to join, which is
why the service keeps at least one task.

## 3. Load balancer and service

```bash
# ALB with an HTTPS listener (ACM certificate) and a target group that checks readiness.
aws elbv2 create-target-group --name voicebox-tg --protocol HTTP --port 17493 --vpc-id vpc-X --target-type ip \
  --health-check-path /health/ready --health-check-interval-seconds 10 --healthy-threshold-count 2 --unhealthy-threshold-count 3
aws elbv2 modify-target-group-attributes --target-group-arn TG_ARN --attributes Key=deregistration_delay.timeout_seconds,Value=45
aws elbv2 modify-load-balancer-attributes --load-balancer-arn ALB_ARN --attributes Key=idle_timeout.timeout_seconds,Value=300

aws ecs register-task-definition --cli-input-json file://deploy/aws/task-definition.json
aws ecs create-service --cluster voicebox --service-name voicebox --task-definition voicebox --desired-count 1 \
  --capacity-provider-strategy capacityProvider=voicebox-gpu,weight=1 \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-A,subnet-B],securityGroups=[sg-tasks]}" \
  --load-balancers targetGroupArn=TG_ARN,containerName=voicebox,containerPort=17493 \
  --health-check-grace-period-seconds 300 \
  --deployment-configuration minimumHealthyPercent=100,maximumPercent=200
```

The ALB idle timeout of 300 s covers a request that waits in a replica's
queue before its first byte; once audio streams, the connection is never
idle. `stopTimeout: 90` in the task definition gives the drain its time.
When ECS stops a task it deregisters the target and sends SIGTERM at the
same moment; `VOICEBOX_SHUTDOWN_DELAY_S=15` keeps the task serving (with
readiness at 503) until the ALB has stopped routing to it, so scale-in
drops no requests.

## 4. Autoscaling on requests per target

```bash
aws application-autoscaling register-scalable-target --service-namespace ecs --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/voicebox/voicebox --min-capacity 1 --max-capacity 4
# ResourceLabel in scaling-policy.json is "<ALB ARN suffix>/<target group ARN suffix>".
aws application-autoscaling put-scaling-policy --service-namespace ecs --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/voicebox/voicebox --policy-name voicebox-requests --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://deploy/aws/scaling-policy.json
```

Two requests per target per minute is a starting point for sentence-length
requests; use the load test to set the value so a replica never holds more
than its `VOICEBOX_MAX_QUEUE_DEPTH`. A custom CloudWatch metric from the
CloudWatch agent or ADOT sidecar scraping port 9464
(`voicebox_queue_pending_jobs`) gives a more direct signal.

WebSockets: the ALB proxies them; its idle timeout (300 s here) closes a
session that sends nothing for that long, which the route's own 60 s idle
timeout already ends earlier.

## 5. CPU-only variant (Fargate)

Kokoro, LuxTTS and Whisper turbo run on CPU: use the `-cpu` image, set
`"requiresCompatibilities": ["FARGATE"]`, `"cpu": "4096"`, `"memory":
"16384"`, drop `resourceRequirements` and `NVIDIA_DRIVER_CAPABILITIES`, set
`VOICEBOX_REQUIRE_GPU=0`, and create the service with `--launch-type
FARGATE`. Fargate tasks start in about a minute, so `min-capacity 1` still
applies. Kubernetes users: `deploy/k8s/` with Karpenter is the EKS route.

## 6. Verify

```bash
scripts/fleet-check.sh https://voice.example.com "$CLIENT_KEY" --rounds 6
```
