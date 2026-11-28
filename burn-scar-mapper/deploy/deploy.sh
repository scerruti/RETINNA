#!/usr/bin/env bash
# Build the image, push it to ECR, and create/update the CloudFormation stack.
#
#   deploy/deploy.sh models/phase3_nir_swir_20261124_032554.pt            # first deploy / update
#   THRESHOLD=0.9 ALARM_EMAIL=you@example.com deploy/deploy.sh models/...  # with settings
#
# Needs: Docker running, AWS CLI v2 configured (aws configure / SSO) for the target account.
#
# Costs (check current pricing): nothing runs while idle. Each page view is one short API
# Lambda call; each NEW fire is one worker run (10 GB for a few minutes, a few cents). Fixed
# costs are S3 storage for results and ~1 GB of image in ECR (well under $1/month).
# Set ALARM_EMAIL to also get a monthly budget alert (MONTHLY_BUDGET_USD, default 10).
#
# Tear down (stops all charges):
#   aws s3 rm s3://<ResultsBucket> --recursive   # bucket name is in the stack outputs
#   aws cloudformation delete-stack --stack-name burn-scar-mapper --region us-west-2
#   aws ecr delete-repository --repository-name burn-scar-mapper --force --region us-west-2
set -euo pipefail

CKPT=${1:?usage: deploy/deploy.sh models/<checkpoint>.pt}
REGION=${AWS_REGION:-us-west-2}
STACK=${STACK:-burn-scar-mapper}
REPO=${REPO:-burn-scar-mapper}
THRESHOLD=${THRESHOLD:-0.5}
NEW_JOBS_PER_HOUR=${NEW_JOBS_PER_HOUR:-10}
WORKER_MEMORY_MB=${WORKER_MEMORY_MB:-10240}
WORKER_CONCURRENCY=${WORKER_CONCURRENCY:-0}
ALARM_EMAIL=${ALARM_EMAIL:-}
MONTHLY_BUDGET_USD=${MONTHLY_BUDGET_USD:-10}

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

echo "== Build $IMAGE (linux/arm64, model $MODEL_FILE)"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
# --provenance=false: Lambda rejects the multi-manifest index that attestations produce.
docker buildx build --platform linux/arm64 --provenance=false --build-arg MODEL_FILE="$MODEL_FILE" \
  -t "$IMAGE" --push .

echo "== Deploy stack $STACK"
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
  --template-file deploy/cloudformation.yaml --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
  --parameter-overrides ImageUri="$IMAGE" Threshold="$THRESHOLD" NewJobsPerHour="$NEW_JOBS_PER_HOUR" \
    WorkerMemoryMB="$WORKER_MEMORY_MB" WorkerConcurrency="$WORKER_CONCURRENCY" \
    AlarmEmail="$ALARM_EMAIL" MonthlyBudgetUSD="$MONTHLY_BUDGET_USD"

aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[].[OutputKey,OutputValue]" --output table
