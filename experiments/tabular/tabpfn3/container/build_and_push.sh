#!/usr/bin/env bash
# Build the TabPFN-3 SageMaker inference image with Finch and push to ECR.
# Usage: ./build_and_push.sh [image_name] [region] [device]
#   device: gpu (default) | cpu
set -euo pipefail

IMAGE_NAME="${1:-tabpfn3-sagemaker}"
REGION="${2:-us-west-2}"
DEVICE="${3:-gpu}"
# Tag is :latest for gpu, :cpu for cpu so both can coexist in the same repo.
if [ "$DEVICE" = "cpu" ]; then
    TAG="${TAG:-cpu}"
else
    TAG="${TAG:-latest}"
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:${TAG}"
DLC_ACCOUNT="763104351884"

cd "$(dirname "$0")"

if ! finch vm status >/dev/null 2>&1; then
    echo "Starting Finch VM..."
    finch vm start || finch vm init
fi

echo "Logging into AWS PyTorch DLC ECR (${DLC_ACCOUNT}, ${REGION})..."
aws ecr get-login-password --region "${REGION}" \
    | finch login --username AWS --password-stdin "${DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

echo "Ensuring ECR repo ${IMAGE_NAME} exists..."
aws ecr describe-repositories --region "${REGION}" --repository-names "${IMAGE_NAME}" >/dev/null 2>&1 \
    || aws ecr create-repository --region "${REGION}" --repository-name "${IMAGE_NAME}" >/dev/null

echo "Logging into account ECR (${ACCOUNT_ID})..."
aws ecr get-login-password --region "${REGION}" \
    | finch login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "Building ${ECR_URI} (device=${DEVICE}) for region ${REGION}..."
finch build --platform linux/amd64 \
    --build-arg REGION="${REGION}" \
    --build-arg DEVICE="${DEVICE}" \
    -t "${ECR_URI}" .

echo "Pushing ${ECR_URI}..."
finch push "${ECR_URI}"

echo ""
echo "ECR_IMAGE_URI=${ECR_URI}"
