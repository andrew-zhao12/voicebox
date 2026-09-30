# 4. Deploy to AWS: ECS on g6 instances behind an ALB

**Goal.** The `deploy/aws/` recipe running: an ECS service of GPU tasks on
an Auto Scaling group of g6.xlarge instances (one L4 each), an Application
Load Balancer in front, EFS holding the models, seed and keys file, and
target-tracking autoscaling on requests per target.

**Time.** About 3 hours the first time; the recipe's README assumes IAM
roles, networking, EFS and the load balancer already exist, so this
document creates them. **Money.** A g6.xlarge, the ALB, EFS and a small S3
bucket are billed while they exist; the teardown at the end removes all of
them. **You need.** [Document 2](02-cloud-prep.md) done, the `aws` CLI
logged in to an account you may create IAM roles in, a default VPC in the
region (every account has one unless it was deleted), and the G-instance
quota from step 1. **Not run yet.** This document has not been executed on
an account; the commands were checked against the installed CLI's own help,
and the AWS facts carry their source.

Everything reads `~/.voicebox-cloud.env`; start each session with
`source ~/.voicebox-cloud.env`. Working copies of the recipe files go to
`~/voicebox-cloud/` so the repository keeps its placeholders.

## 1. Log in, pick the region, check the GPU quota

```bash
aws sso login                     # or `aws configure`, whichever your account uses
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
printf 'export AWS_REGION=%s\nexport AWS_DEFAULT_REGION=%s\nexport ACCOUNT=%s\n' "$AWS_REGION" "$AWS_REGION" "$ACCOUNT" >> ~/.voicebox-cloud.env
mkdir -p ~/voicebox-cloud
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.{name:QuotaName,value:Value}'
```

**Check:** the quota is named `Running On-Demand G and VT instances` and
its value is at least 8 (a g6.xlarge uses 4 vCPUs of it; the scale-out test
in [document 6](06-test-on-the-cloud.md) needs two instances). Confirm the
quota code in the Service Quotas console if the name differs.

**If not:** `aws service-quotas request-service-quota-increase
--service-code ec2 --quota-code L-DB2E81BA --desired-value 16`, then wait;
new accounts often start at 0 and an increase can take hours to days. Do
the rest of this document only once it is granted, because the Auto
Scaling group in step 7 otherwise fails silently to launch anything.

## 2. Networking: the default VPC, two subnets, three security groups

The ALB accepts the internet (port 80 only from your own IP until HTTPS is
set up in step 12), the tasks accept the ALB, EFS accepts the tasks.

```bash
source ~/.voicebox-cloud.env
VPC_ID=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
read -r SUBNET_A SUBNET_B <<< "$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID Name=default-for-az,Values=true --query 'Subnets[0:2].SubnetId' --output text)"
SG_ALB=$(aws ec2 create-security-group --group-name voicebox-alb --description "Voicebox ALB" --vpc-id $VPC_ID --query GroupId --output text)
SG_TASKS=$(aws ec2 create-security-group --group-name voicebox-tasks --description "Voicebox tasks" --vpc-id $VPC_ID --query GroupId --output text)
SG_EFS=$(aws ec2 create-security-group --group-name voicebox-efs --description "Voicebox EFS" --vpc-id $VPC_ID --query GroupId --output text)
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id $SG_ALB --protocol tcp --port 80 --cidr $MYIP/32
aws ec2 authorize-security-group-ingress --group-id $SG_TASKS --protocol tcp --port 17493 --source-group $SG_ALB
aws ec2 authorize-security-group-ingress --group-id $SG_EFS --protocol tcp --port 2049 --source-group $SG_TASKS
printf 'export VPC_ID=%s\nexport SUBNET_A=%s\nexport SUBNET_B=%s\nexport SG_ALB=%s\nexport SG_TASKS=%s\nexport SG_EFS=%s\n' "$VPC_ID" "$SUBNET_A" "$SUBNET_B" "$SG_ALB" "$SG_TASKS" "$SG_EFS" >> ~/.voicebox-cloud.env
echo "$VPC_ID $SUBNET_A $SUBNET_B $SG_ALB $SG_TASKS $SG_EFS"
```

**Check:** six ids print, none of them `None`. **If not:** no default VPC
(`VPC_ID` is `None`): create one with `aws ec2 create-default-vpc`, or use
your own VPC's two public subnets (they need `MapPublicIpOnLaunch`, or a NAT
gateway, so the instance can pull the image and reach SSM).

## 3. IAM: the instance role, the task execution role, the task role

The names are the ones the recipe's task definition and launch template
reference. `ecsInstanceRole` and `ecsTaskExecutionRole` may already exist
in your account; then only the `attach`/`put` lines matter.

```bash
EC2_TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
ECS_TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam create-role --role-name ecsInstanceRole --assume-role-policy-document "$EC2_TRUST"
aws iam attach-role-policy --role-name ecsInstanceRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role
aws iam attach-role-policy --role-name ecsInstanceRole --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam attach-role-policy --role-name ecsInstanceRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
aws iam create-instance-profile --instance-profile-name ecsInstanceRole
aws iam add-role-to-instance-profile --instance-profile-name ecsInstanceRole --role-name ecsInstanceRole
aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document "$ECS_TRUST"
aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam put-role-policy --role-name ecsTaskExecutionRole --policy-name voicebox-secrets-and-logs --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"secretsmanager:GetSecretValue","Resource":"arn:aws:secretsmanager:*:*:secret:voicebox/*"},{"Effect":"Allow","Action":"logs:CreateLogGroup","Resource":"*"}]}'
aws iam create-role --role-name voicebox-task --assume-role-policy-document "$ECS_TRUST"
aws iam get-role --role-name ecsInstanceRole --query 'Role.Arn' --output text
aws iam list-attached-role-policies --role-name ecsInstanceRole --query 'AttachedPolicies[].PolicyName' --output text
```

**Check:** the role ARN prints and the attached policies include
`AmazonEC2ContainerServiceforEC2Role` and `AmazonSSMManagedInstanceCore`
(SSM is how step 8 runs commands on the instance without SSH; S3 read is
how the instance fetches your seed and keys). **If not:** `EntityAlreadyExists`
on a `create-role` is fine; `AccessDenied` means your login may not create
IAM roles, and an administrator has to run this step.

## 4. EFS: one file system, a mount target per subnet

```bash
source ~/.voicebox-cloud.env
FS_ID=$(aws efs create-file-system --encrypted --performance-mode generalPurpose --throughput-mode elastic --tags Key=Name,Value=voicebox --query FileSystemId --output text)
echo "export FS_ID=$FS_ID" >> ~/.voicebox-cloud.env
until [ "$(aws efs describe-file-systems --file-system-id $FS_ID --query 'FileSystems[0].LifeCycleState' --output text)" = available ]; do sleep 5; done
for s in $SUBNET_A $SUBNET_B; do aws efs create-mount-target --file-system-id $FS_ID --subnet-id $s --security-groups $SG_EFS --query MountTargetId --output text; done
until [ "$(aws efs describe-mount-targets --file-system-id $FS_ID --query 'MountTargets[?LifeCycleState!=`available`] | length(@)')" = 0 ]; do sleep 10; done
aws efs describe-mount-targets --file-system-id $FS_ID --query 'MountTargets[].[SubnetId,LifeCycleState]' --output text
```

**Check:** two lines, both `available`. The directories `/models`, `/seed`
and `/secrets` the task definition mounts are created in step 8, from the
instance.

## 5. Secrets Manager and the inputs bucket

```bash
source ~/.voicebox-cloud.env
aws secretsmanager create-secret --name voicebox/admin-key --secret-string "$ADMIN_KEY" --query ARN --output text
aws secretsmanager create-secret --name voicebox/media-token-secret --secret-string "$MEDIA_SECRET" --query ARN --output text
BUCKET=voicebox-inputs-$ACCOUNT
aws s3 mb s3://$BUCKET
aws s3 cp ./seed s3://$BUCKET/seed/ --recursive
aws s3 cp ./secrets/api_keys.json s3://$BUCKET/secrets/api_keys.json
echo "export BUCKET=$BUCKET" >> ~/.voicebox-cloud.env
aws s3 ls s3://$BUCKET --recursive
```

**Check:** two secret ARNs, and the listing shows the two bundles and
`secrets/api_keys.json`. The task definition names the secrets by ARN
without the random suffix; Secrets Manager resolves that partial ARN as
long as the name is unique, which it is.

## 6. The cluster, the GPU AMI, the launch template

```bash
source ~/.voicebox-cloud.env
aws ecs create-cluster --cluster-name voicebox --query 'cluster.status' --output text
AMI=$(aws ssm get-parameters --names /aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended --query 'Parameters[0].Value' --output text | python3 -c 'import json,sys; print(json.load(sys.stdin)["image_id"])')
echo "$AMI"
```

**Check:** `ACTIVE`, then an `ami-...` id. The parameter is the ECS
GPU-optimized AMI ([ECS GPU
docs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html)
document the `amazon-linux-2` path; Amazon Linux 2 is past its end of
life, so this document tries the `amazon-linux-2023` parameter first,
**confirm at run time**). **If not:** an empty `AMI` means the AL2023
parameter does not exist in your region yet; use
`/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended` and
`yum` instead of `dnf` in step 8.

The launch template: the AMI, a g6.xlarge, the instance profile, the user
data that joins the cluster with GPU support, and a 100 GB root disk (the
default 30 GB does not hold the 8 GB image plus its layers comfortably):

```bash
USERDATA=$(printf '#!/bin/bash\necho ECS_CLUSTER=voicebox >> /etc/ecs/ecs.config\necho ECS_ENABLE_GPU_SUPPORT=true >> /etc/ecs/ecs.config\n' | base64 | tr -d '\n')
cat > ~/voicebox-cloud/launch-template.json <<JSON
{"ImageId":"$AMI","InstanceType":"g6.xlarge","IamInstanceProfile":{"Name":"ecsInstanceRole"},"UserData":"$USERDATA",
 "BlockDeviceMappings":[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]}
JSON
aws ec2 create-launch-template --launch-template-name voicebox-gpu --launch-template-data file://$HOME/voicebox-cloud/launch-template.json --query 'LaunchTemplate.LaunchTemplateId' --output text
```

**Check:** an `lt-...` id. **If not:** `InvalidParameterValue` on
`ImageId` means the AMI id belongs to another region than `AWS_REGION`.

## 7. The Auto Scaling group and the capacity provider

```bash
source ~/.voicebox-cloud.env
aws autoscaling create-auto-scaling-group --auto-scaling-group-name voicebox-gpu \
  --launch-template LaunchTemplateName=voicebox-gpu --min-size 1 --max-size 4 --desired-capacity 1 \
  --vpc-zone-identifier "$SUBNET_A,$SUBNET_B" --new-instances-protected-from-scale-in
ASG_ARN=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names voicebox-gpu --query 'AutoScalingGroups[0].AutoScalingGroupARN' --output text)
aws ecs create-capacity-provider --name voicebox-gpu --auto-scaling-group-provider "autoScalingGroupArn=$ASG_ARN,managedScaling={status=ENABLED,targetCapacity=100},managedTerminationProtection=ENABLED" --query 'capacityProvider.status' --output text
aws ecs put-cluster-capacity-providers --cluster voicebox --capacity-providers voicebox-gpu --default-capacity-provider-strategy capacityProvider=voicebox-gpu,weight=1 --query 'cluster.capacityProviders' --output text
until [ "$(aws ecs list-container-instances --cluster voicebox --query 'length(containerInstanceArns)')" = 1 ]; do printf .; sleep 15; done; echo
CI_ARN=$(aws ecs list-container-instances --cluster voicebox --query 'containerInstanceArns[0]' --output text)
aws ecs describe-container-instances --cluster voicebox --container-instances $CI_ARN \
  --query 'containerInstances[0].{status:status,instance:ec2InstanceId,memoryMiB:registeredResources[?name==`MEMORY`].integerValue|[0],gpus:registeredResources[?name==`GPU`].stringSetValue|[0]}'
INSTANCE_ID=$(aws ecs describe-container-instances --cluster voicebox --container-instances $CI_ARN --query 'containerInstances[0].ec2InstanceId' --output text)
echo "export INSTANCE_ID=$INSTANCE_ID" >> ~/.voicebox-cloud.env
```

**Check:** the instance joins within about five minutes with `status:
ACTIVE`, `memoryMiB` around 15000 (a g6.xlarge has 16 GiB; the task
definition asks for 14336, which is why the recipe's original 28672 could
never be placed) and `gpus: ["0"]`.

**If not:** no instance after ten minutes: `aws autoscaling
describe-scaling-activities --auto-scaling-group-name voicebox-gpu
--max-items 3` shows the reason (`VcpuLimitExceeded` is the quota from
step 1; `Unsupported` means g6 is not offered in that subnet's zone, so
pick two other subnets). An instance that is running but never joins the
cluster has no route to the ECS endpoint (a subnet without public IPs or a
NAT) or wrong user data (`aws ssm start-session` and read
`/etc/ecs/ecs.config`).

## 8. Fill EFS from the instance, with the CUDA image

The instance has the GPU driver, Docker with GPU support, the AWS CLI and an
SSM agent, so it can mount EFS, fetch the seed and keys from S3 and run the
preload inside `main-cu128`, which also warms the image for the first task.
This runs the script through SSM Run Command; no SSH, no plugin. The
preload downloads on the order of 15 GB and loads each model once, so give
it up to an hour.

```bash
source ~/.voicebox-cloud.env
cat > ~/voicebox-cloud/fill-efs.sh <<SCRIPT
set -euxo pipefail
dnf install -y amazon-efs-utils
mkdir -p /mnt/efs && mount -t efs -o tls $FS_ID:/ /mnt/efs
mkdir -p /mnt/efs/models /mnt/efs/seed /mnt/efs/secrets
aws s3 sync s3://$BUCKET/seed /mnt/efs/seed
aws s3 cp s3://$BUCKET/secrets/api_keys.json /mnt/efs/secrets/api_keys.json
chown -R 999:999 /mnt/efs/models /mnt/efs/seed /mnt/efs/secrets
docker run --rm --gpus all -v /mnt/efs/models:/models -e VOICEBOX_MODELS_DIR=/models -e NUMBA_CACHE_DIR=/tmp/numba_cache \\
  ghcr.io/andrew-zhao12/voicebox:main-cu128 python -m backend.preload kokoro whisper-turbo qwen-tts-1.7B qwen-custom-voice-1.7B luxtts chatterbox-turbo tada-1b
ls -la /mnt/efs/models /mnt/efs/seed /mnt/efs/secrets
SCRIPT
CMD_ID=$(aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters "$(jq -n --arg s "$(cat ~/voicebox-cloud/fill-efs.sh)" '{commands: [$s], executionTimeout: ["5400"]}')" \
  --query 'Command.CommandId' --output text)
until [ "$(aws ssm get-command-invocation --command-id $CMD_ID --instance-id $INSTANCE_ID --query Status --output text)" != InProgress ]; do printf .; sleep 30; done; echo
aws ssm get-command-invocation --command-id $CMD_ID --instance-id $INSTANCE_ID --query '{status:Status,out:StandardOutputContent,err:StandardErrorContent}' --output json | jq -r '.status, (.out|.[-1500:]), (.err|.[-1500:])'
```

**Check:** status `Success`, one `ready: NAME` line per model in the
output, and the final listing shows `models--...` directories owned by
`999` (the image's `voicebox` user), the two bundles in `seed` and
`api_keys.json` in `secrets`.

**If not:** `Pending` for minutes means the SSM agent has not registered
(the instance role lacks `AmazonSSMManagedInstanceCore`, or no route to
SSM); `dnf: command not found` means an AL2 AMI (`yum`); `docker: Error
response from daemon: could not select device driver` means the AMI is
not the GPU one; an interrupted download can simply be re-run (the cache
is resumable). To debug interactively install the Session Manager plugin
(`brew install --cask session-manager-plugin`) and `aws ssm start-session
--target $INSTANCE_ID`. **Confirm at run time:** the `amazon-efs-utils`
package name and `dnf` are the AL2023 conventions.

## 9. The load balancer, target group and listener

```bash
source ~/.voicebox-cloud.env
ALB_ARN=$(aws elbv2 create-load-balancer --name voicebox-alb --subnets $SUBNET_A $SUBNET_B --security-groups $SG_ALB --query 'LoadBalancers[0].LoadBalancerArn' --output text)
ALB_DNS=$(aws elbv2 describe-load-balancers --load-balancer-arns $ALB_ARN --query 'LoadBalancers[0].DNSName' --output text)
TG_ARN=$(aws elbv2 create-target-group --name voicebox-tg --protocol HTTP --port 17493 --vpc-id $VPC_ID --target-type ip \
  --health-check-path /health/ready --health-check-interval-seconds 10 --healthy-threshold-count 2 --unhealthy-threshold-count 3 \
  --query 'TargetGroups[0].TargetGroupArn' --output text)
aws elbv2 modify-target-group-attributes --target-group-arn $TG_ARN --attributes Key=deregistration_delay.timeout_seconds,Value=45 --query 'Attributes[?Key==`deregistration_delay.timeout_seconds`].Value' --output text
aws elbv2 modify-load-balancer-attributes --load-balancer-arn $ALB_ARN --attributes Key=idle_timeout.timeout_seconds,Value=300 --query 'Attributes[?Key==`idle_timeout.timeout_seconds`].Value' --output text
aws elbv2 create-listener --load-balancer-arn $ALB_ARN --protocol HTTP --port 80 --default-actions Type=forward,TargetGroupArn=$TG_ARN --query 'Listeners[0].ListenerArn' --output text
printf 'export ALB_ARN=%s\nexport ALB_DNS=%s\nexport TG_ARN=%s\nexport URL=http://%s\n' "$ALB_ARN" "$ALB_DNS" "$TG_ARN" "$ALB_DNS" >> ~/.voicebox-cloud.env
echo "$ALB_DNS"
```

**Check:** `45`, `300`, a listener ARN and the `voicebox-alb-....elb.amazonaws.com`
name. The idle timeout of 300 s covers a request that waits in a queue
before its first byte; the default is 60 s ([ALB
docs](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/application-load-balancers.html)).
Port 80 is open to your IP only (step 2); step 12 adds HTTPS for everyone
else.

## 10. The task definition and the service

```bash
source ~/.voicebox-cloud.env
sed -e "s/ACCOUNT/$ACCOUNT/g" -e "s/REGION/$AWS_REGION/g" -e "s/fs-EFS/$FS_ID/g" \
    -e "s#ghcr.io/OWNER/voicebox:VERSION-cu128#ghcr.io/andrew-zhao12/voicebox:main-cu128#" deploy/aws/task-definition.json \
  | jq --arg h "$ALB_DNS" '.containerDefinitions[0].environment += [{"name":"VOICEBOX_ALLOWED_HOSTS","value":$h}]' > ~/voicebox-cloud/task-definition.json
grep -cE 'ACCOUNT|REGION|fs-EFS|OWNER' ~/voicebox-cloud/task-definition.json
aws ecs register-task-definition --cli-input-json file://$HOME/voicebox-cloud/task-definition.json --query 'taskDefinition.{family:family,revision:revision,memory:memory}'
aws ecs create-service --cluster voicebox --service-name voicebox --task-definition voicebox --desired-count 1 \
  --capacity-provider-strategy capacityProvider=voicebox-gpu,weight=1 \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_A,$SUBNET_B],securityGroups=[$SG_TASKS]}" \
  --load-balancers targetGroupArn=$TG_ARN,containerName=voicebox,containerPort=17493 \
  --health-check-grace-period-seconds 300 \
  --deployment-configuration minimumHealthyPercent=100,maximumPercent=200 \
  --query 'service.status' --output text
aws ecs wait services-stable --cluster voicebox --services voicebox
aws elbv2 describe-target-health --target-group-arn $TG_ARN --query 'TargetHealthDescriptions[].TargetHealth.State' --output text
curl -s $URL/health/ready | jq
```

**Check:** the `grep` prints `0`; the registration prints `revision: 1`
and `memory: "14336"`; the wait returns within ten minutes (the image is
already on the instance from step 8, so the task starts as soon as the
models load); the target is `healthy`; the readiness body has
`"ready": true`, `models.ready` with `kokoro` and `whisper-turbo`, and
`startup.done` with `seed_profiles` and `gpu`.

**If not:**

- `aws logs tail /ecs/voicebox --since 20m` shows the container's lines.
- The task stops with `ResourceInitializationError ... efs`: the mount
  targets are not `available`, or `SG_EFS` does not allow `SG_TASKS`.
- `CannotPullContainerError`: the instance cannot reach ghcr.io (no public
  IP or NAT).
- The target stays `unhealthy` past the 300 s grace: readiness never
  reached 200; the body from inside the instance tells why
  (`curl -s http://TASK_IP:17493/health/ready` through SSM), usually
  `models.failed` (EFS not filled) or `startup.failed: ["gpu"]`.
- `unable to place a task because no container instance met all of its
  requirements`: memory or GPU; compare step 7's registered resources with
  the task definition.

## 11. Autoscaling on requests per target

`ResourceLabel` is the ALB's ARN suffix joined with the target group's:

```bash
source ~/.voicebox-cloud.env
LABEL="${ALB_ARN##*:loadbalancer/}/${TG_ARN##*:}"
echo "$LABEL"
sed "s#app/voicebox-alb/ALB-ID/targetgroup/voicebox-tg/TG-ID#$LABEL#" deploy/aws/scaling-policy.json > ~/voicebox-cloud/scaling-policy.json
aws application-autoscaling register-scalable-target --service-namespace ecs --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/voicebox/voicebox --min-capacity 1 --max-capacity 4
aws application-autoscaling put-scaling-policy --service-namespace ecs --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/voicebox/voicebox --policy-name voicebox-requests --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://$HOME/voicebox-cloud/scaling-policy.json --query 'Alarms[].AlarmName' --output text
```

**Check:** the label looks like
`app/voicebox-alb/0123abcd/targetgroup/voicebox-tg/4567efgh`, and two
alarm names print (`...AlarmHigh...` and `...AlarmLow...`). The policy
adds a task when the ALB sees more than 2 requests per target per minute
and needs a second instance for it, which the capacity provider launches;
scale-in is suspended while a deployment is in progress ([target tracking
docs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-autoscaling-targettracking.html)).

## 12. First contact, then HTTPS if the service outlives the test

```bash
source ~/.voicebox-cloud.env
curl -s -H "Authorization: Bearer $CLIENT_KEY" $URL/auth/whoami | jq -c .
scripts/fleet-check.sh "$URL" "$CLIENT_KEY" --rounds 2
curl -s -o /dev/null -w '%{http_code}\n' -H 'Host: evil.example' -H "Authorization: Bearer $CLIENT_KEY" $URL/v1/models
```

**Check:** `{"key_id":"myapp",...}`, `fleet check passed: 2 round(s)
against http://voicebox-alb-...`, then `400` for the foreign `Host`. The
full battery is [document 6](06-test-on-the-cloud.md).

Keys travel in the `Authorization` header, so plain HTTP is only
acceptable from your own IP for a test. For anything else: request an ACM
certificate for a name you control, validate it through DNS, add an HTTPS
listener and open port 443:

```bash
CERT_ARN=$(aws acm request-certificate --domain-name voice.example.com --validation-method DNS --query CertificateArn --output text)
aws acm describe-certificate --certificate-arn $CERT_ARN --query 'Certificate.DomainValidationOptions[0].ResourceRecord'   # create this CNAME at your DNS provider
aws acm wait certificate-validated --certificate-arn $CERT_ARN
aws elbv2 create-listener --load-balancer-arn $ALB_ARN --protocol HTTPS --port 443 --certificates CertificateArn=$CERT_ARN \
  --ssl-policy ELBSecurityPolicy-TLS13-1-2-2021-06 --default-actions Type=forward,TargetGroupArn=$TG_ARN
aws ec2 authorize-security-group-ingress --group-id $SG_ALB --protocol tcp --port 443 --cidr 0.0.0.0/0
# then a CNAME or alias record voice.example.com -> $ALB_DNS, and VOICEBOX_ALLOWED_HOSTS=voice.example.com in a new task-definition revision
```

## Teardown

In dependency order; each block waits for the previous one to settle.

```bash
source ~/.voicebox-cloud.env
aws application-autoscaling deregister-scalable-target --service-namespace ecs --scalable-dimension ecs:service:DesiredCount --resource-id service/voicebox/voicebox
aws ecs update-service --cluster voicebox --service voicebox --desired-count 0 --query 'service.desiredCount'
aws ecs wait services-stable --cluster voicebox --services voicebox
aws ecs delete-service --cluster voicebox --service voicebox --force --query 'service.status' --output text
for l in $(aws elbv2 describe-listeners --load-balancer-arn $ALB_ARN --query 'Listeners[].ListenerArn' --output text); do aws elbv2 delete-listener --listener-arn $l; done
aws elbv2 delete-load-balancer --load-balancer-arn $ALB_ARN
aws elbv2 delete-target-group --target-group-arn $TG_ARN
aws autoscaling update-auto-scaling-group --auto-scaling-group-name voicebox-gpu --min-size 0 --desired-capacity 0
aws autoscaling delete-auto-scaling-group --auto-scaling-group-name voicebox-gpu --force-delete
aws ecs put-cluster-capacity-providers --cluster voicebox --capacity-providers --default-capacity-provider-strategy
aws ecs delete-capacity-provider --capacity-provider voicebox-gpu --query 'capacityProvider.status' --output text
aws ecs delete-cluster --cluster voicebox --query 'cluster.status' --output text
aws ec2 delete-launch-template --launch-template-name voicebox-gpu --query 'LaunchTemplate.LaunchTemplateName' --output text
for mt in $(aws efs describe-mount-targets --file-system-id $FS_ID --query 'MountTargets[].MountTargetId' --output text); do aws efs delete-mount-target --mount-target-id $mt; done
sleep 90; aws efs delete-file-system --file-system-id $FS_ID
aws secretsmanager delete-secret --secret-id voicebox/admin-key --force-delete-without-recovery --query Name --output text
aws secretsmanager delete-secret --secret-id voicebox/media-token-secret --force-delete-without-recovery --query Name --output text
aws s3 rb s3://$BUCKET --force
aws logs delete-log-group --log-group-name /ecs/voicebox
sleep 60; for sg in $SG_EFS $SG_TASKS $SG_ALB; do aws ec2 delete-security-group --group-id $sg; done
aws ec2 describe-instances --filters Name=instance-state-name,Values=running Name=tag:aws:autoscaling:groupName,Values=voicebox-gpu --query 'Reservations[].Instances[].InstanceId' --output text
```

**Check:** the last command prints nothing, and the billing console shows
EC2, ELB, EFS and S3 usage ending today. `DependencyViolation` on a
security group means an ENI still references it (a stopping task or the
load balancer); wait a minute and retry that line. The IAM roles cost
nothing and other services may use the two AWS-standard names, so they stay.

## Done when

- [ ] `curl $URL/health/ready` answers 200 with `startup.done` ⊇ `seed_profiles, gpu` and the target is `healthy`.
- [ ] `fleet-check.sh` passes over the ALB with `CLIENT_KEY`; a foreign `Host` is refused.
- [ ] The scaling policy exists with its two alarms.
- [ ] `~/.voicebox-cloud.env` holds `URL`, `ALB_ARN`, `TG_ARN`, `FS_ID`, `INSTANCE_ID`, `BUCKET` and the security-group ids for the teardown.
