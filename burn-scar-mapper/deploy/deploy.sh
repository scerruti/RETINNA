#!/usr/bin/env bash
# Build the image, push it to ECR, and create/update the CloudFormation stack.
#
#   deploy/deploy.sh models/phase3_nir_swir_20261124_032554.pt            # first deploy / update
#   THRESHOLD=0.9 ALARM_EMAIL=you@example.com deploy/deploy.sh models/...  # with settings
#
# Needs: Docker running, AWS CLI v2 configured (aws configure / SSO) for the target account.
# Costs (check current pricing): one always-on Fargate task (2 vCPU / 4 GB, x86) plus the
# ECS-managed Application Load Balancer, roughly $90/month together. Park it between interviews:
#   TASK_COUNT=0 deploy/deploy.sh models/...   (stops Fargate charges; the ALB still bills)
# Delete the stack to stop all charges:
#   aws cloudformation delete-stack --stack-name burn-scar-mapper --region us-west-2
#   (empty the results bucket first, or the delete stops at the bucket)
set -euo pipefail

CKPT=${1:?usage: deploy/deploy.sh models/<checkpoint>.pt}
REGION=${AWS_REGION:-us-west-2}
STACK=${STACK:-burn-scar-mapper}
REPO=${REPO:-burn-scar-mapper}
THRESHOLD=${THRESHOLD:-0.5}
NEW_JOBS_PER_HOUR=${NEW_JOBS_PER_HOUR:-10}
ALARM_EMAIL=${ALARM_EMAIL:-}
TASK_COUNT=${TASK_COUNT:-1}

cd "$(dirname "$0")/.."
[[ -f "$CKPT" && "$CKPT" == models/* ]] || { echo "checkpoint must be a file under models/: $CKPT" >&2; exit 1; }
MODEL_FILE=$(basename "$CKPT")

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
TAG="$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD 2>/dev/null || echo local)"
IMAGE="$REGISTRY/$REPO:$TAG"

echo "== ECR repository $REPO ($REGION)"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 ||
  aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null

echo "== Build $IMAGE (linux/amd64, model $MODEL_FILE)"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker build --platform linux/amd64 --build-arg MODEL_FILE="$MODEL_FILE" -t "$IMAGE" .
docker push "$IMAGE"

echo "== Deploy stack $STACK"
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
  --template-file deploy/cloudformation.yaml --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
  --parameter-overrides ImageUri="$IMAGE" Threshold="$THRESHOLD" \
    NewJobsPerHour="$NEW_JOBS_PER_HOUR" AlarmEmail="$ALARM_EMAIL" TaskCount="$TASK_COUNT"

aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[].[OutputKey,OutputValue]" --output table
