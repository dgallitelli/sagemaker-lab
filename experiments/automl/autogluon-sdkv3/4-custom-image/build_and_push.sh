#!/bin/bash
#
# Build custom AutoGluon Docker images and push them to Amazon ECR.
#
# Usage:
#   ./build_and_push.sh <sub-experiment> [repo-name] [image-tag]
#
# Examples:
#   ./build_and_push.sh ag-dlc-upgrade
#   ./build_and_push.sh pytorch-dlc autogluon-custom ag150-pytorch-dlc
#
# Arguments:
#   sub-experiment  Required. Directory name: "ag-dlc-upgrade" or "pytorch-dlc"
#   repo-name       Optional. ECR repository name (default: autogluon-custom)
#   image-tag       Optional. Image tag prefix (default: derived from sub-experiment)

set -euo pipefail

# ---------------------------------------------------------------------------
# Container runtime: prefer docker, fall back to finch
# Override by setting CONTAINER_RUNTIME=finch (or docker)
# ---------------------------------------------------------------------------
if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
    RUNTIME="$CONTAINER_RUNTIME"
elif command -v docker &>/dev/null; then
    RUNTIME="docker"
elif command -v finch &>/dev/null; then
    RUNTIME="finch"
else
    echo "ERROR: Neither docker nor finch found. Install one and retry."
    exit 1
fi
echo "Using container runtime: $RUNTIME"

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
SUB_EXPERIMENT="${1:?Usage: $0 <ag-dlc-upgrade|pytorch-dlc> [repo-name] [image-tag]}"
REPO_NAME="${2:-autogluon-custom}"

# Derive default tag from sub-experiment name
case "$SUB_EXPERIMENT" in
    ag-dlc-upgrade)  DEFAULT_TAG="ag150-dlc-upgrade" ;;
    pytorch-dlc)     DEFAULT_TAG="ag150-pytorch-dlc" ;;
    *)               DEFAULT_TAG="$SUB_EXPERIMENT" ;;
esac
IMAGE_TAG="${3:-$DEFAULT_TAG}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/$SUB_EXPERIMENT/docker"

if [[ ! -d "$DOCKER_DIR" ]]; then
    echo "ERROR: Docker directory not found: $DOCKER_DIR"
    exit 1
fi

# ---------------------------------------------------------------------------
# AWS configuration
# ---------------------------------------------------------------------------
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region || echo "us-east-1")

if [[ -z "$ACCOUNT_ID" ]]; then
    echo "ERROR: Could not determine AWS account ID. Check your credentials."
    exit 1
fi

ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
FULL_TRAINING="$ECR_URI/$REPO_NAME:$IMAGE_TAG-training"
FULL_INFERENCE="$ECR_URI/$REPO_NAME:$IMAGE_TAG-inference"

echo "Account:    $ACCOUNT_ID"
echo "Region:     $REGION"
echo "Repository: $REPO_NAME"
echo "Tag prefix: $IMAGE_TAG"
echo "Training:   $FULL_TRAINING"
echo "Inference:  $FULL_INFERENCE"

# ---------------------------------------------------------------------------
# Resolve base images using the SageMaker SDK
# ---------------------------------------------------------------------------
resolve_base_image() {
    local framework="$1" scope="$2" extra_args="${3:-}"
    python3 -c "
from sagemaker import image_uris
print(image_uris.retrieve(
    '$framework', region='$REGION', $extra_args
    image_scope='$scope', instance_type='ml.m5.2xlarge'))
"
}

case "$SUB_EXPERIMENT" in
    ag-dlc-upgrade)
        TRAINING_BASE=$(resolve_base_image autogluon training "version='1.5', py_version='py312',")
        INFERENCE_BASE=$(resolve_base_image autogluon inference "version='1.5', py_version='py312',")
        ;;
    pytorch-dlc)
        TRAINING_BASE=$(resolve_base_image pytorch training "version='2.6', py_version='py312',")
        INFERENCE_BASE=$(resolve_base_image pytorch inference "version='2.6', py_version='py312',")
        ;;
    *)
        echo "ERROR: Unknown sub-experiment '$SUB_EXPERIMENT'. Expected 'ag-dlc-upgrade' or 'pytorch-dlc'."
        exit 1
        ;;
esac

DLC_ACCOUNT=$(echo "$TRAINING_BASE" | cut -d. -f1)
echo "Training base:  $TRAINING_BASE"
echo "Inference base: $INFERENCE_BASE"

# ---------------------------------------------------------------------------
# ECR setup
# ---------------------------------------------------------------------------
# Create repository if it doesn't exist
aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION" >/dev/null

# Login to DLC ECR (to pull base images)
aws ecr get-login-password --region "$REGION" \
    | $RUNTIME login --username AWS --password-stdin "$DLC_ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

# Login to your account ECR (to push)
aws ecr get-login-password --region "$REGION" \
    | $RUNTIME login --username AWS --password-stdin "$ECR_URI"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo ""
echo "=== Building training image ==="
$RUNTIME build \
    --build-arg BASE_IMAGE="$TRAINING_BASE" \
    -t "$FULL_TRAINING" \
    -f "$DOCKER_DIR/Dockerfile.training" \
    "$DOCKER_DIR"

echo ""
echo "=== Building inference image ==="
$RUNTIME build \
    --build-arg BASE_IMAGE="$INFERENCE_BASE" \
    -t "$FULL_INFERENCE" \
    -f "$DOCKER_DIR/Dockerfile.inference" \
    "$DOCKER_DIR"

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
echo ""
echo "=== Smoke test ==="
$RUNTIME run --rm --entrypoint python "$FULL_TRAINING" \
    -c "import autogluon.tabular; print('Training image - AG version:', autogluon.tabular.__version__)"
$RUNTIME run --rm --entrypoint python "$FULL_INFERENCE" \
    -c "import autogluon.tabular; print('Inference image - AG version:', autogluon.tabular.__version__)"

# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------
echo ""
echo "=== Pushing to ECR ==="
$RUNTIME push "$FULL_TRAINING"
$RUNTIME push "$FULL_INFERENCE"

echo ""
echo "=== Done ==="
echo "Training:  $FULL_TRAINING"
echo "Inference: $FULL_INFERENCE"
