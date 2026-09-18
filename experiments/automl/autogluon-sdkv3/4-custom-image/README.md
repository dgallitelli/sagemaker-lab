# Custom Docker Images for AutoGluon on SageMaker

Build custom Docker images for AutoGluon on SageMaker. Useful when you need custom dependencies, private packages, or a specific AutoGluon version pinned in your image.

Two base image strategies are provided:

| Sub-experiment | Base Image | When to Use |
|---|---|---|
| **ag-dlc-upgrade** | AutoGluon DLC v1.5 | Minimal changes — add custom dependencies on top of the DLC. Inherits all DLC optimizations. |
| **pytorch-dlc** | PyTorch DLC v2.6 | Full control — install AG from scratch on a well-maintained PyTorch base. |

Both sub-experiments use the **tabular classification** (Adult Census) use case and follow the same structure:

```
<sub-experiment>/
  docker/
    Dockerfile.training       # Custom training image
    Dockerfile.inference      # Custom inference image
    requirements.txt          # AutoGluon version pin
  0-build-push/
    build_and_push.ipynb      # Build images with finch, push to ECR
  1-training/
    launch_training.ipynb     # SageMaker Training job with custom image
    train.py                  # Training script
    config.yaml               # AutoGluon config
  2-inference/
    deploy.ipynb              # Deploy real-time endpoint
    serve.py                  # Inference handler
  3-pipeline/
    pipeline.ipynb            # SageMaker Pipeline
    evaluate.py               # Evaluation script
```

## Prerequisites

1. **Processed data from tabular-classification**: Run `1-tabular-classification/0-data-prep/` first to upload and preprocess the Adult Census dataset. Both sub-experiments reuse that processed data in S3.

2. **Finch installed**: This project uses [Finch](https://github.com/runfinch/finch) (not Docker) for container builds:
   ```bash
   brew install --cask finch
   finch vm start
   ```

3. **Python environment**: Use the project's `.venv`:
   ```bash
   cd sagemaker-autogluon-sdkv3
   source .venv/bin/activate
   pip install "sagemaker>=3.0,<4.0" boto3 pandas jupyter
   ```

4. **AWS credentials** configured with permissions for ECR and SageMaker.

## Which Base Image to Choose?

**ag-dlc-upgrade** if you:
- Want the smallest possible diff from the official DLC
- Need DLC-specific optimizations (SageMaker training toolkit, pre-configured TorchServe)
- Just need to add custom dependencies on top of the DLC

**pytorch-dlc** if you:
- Want to control the full dependency stack
- Need a specific PyTorch version not available in the AG DLC
- Plan to add other ML libraries alongside AutoGluon

## Workflow

1. **Build and push** (`0-build-push/build_and_push.ipynb`):
   - Resolves the base DLC URI via `image_uris.retrieve()`
   - Builds training and inference images with `finch build`
   - Runs a local smoke test to verify the AutoGluon version
   - Pushes both images to your ECR repository

2. **Train** (`1-training/launch_training.ipynb`):
   - Same `ModelTrainer` pattern as the 1-tabular-classification experiment
   - Uses your custom ECR image instead of the managed DLC

3. **Deploy** (`2-inference/deploy.ipynb`):
   - Deploys a real-time endpoint with your custom inference image
   - Same v3 resource API: `Model.create()` -> `EndpointConfig.create()` -> `Endpoint.create()`

4. **Pipeline** (`3-pipeline/pipeline.ipynb`):
   - End-to-end pipeline using custom images for training and evaluation
   - Preprocessing still uses the sklearn DLC (no custom image needed)

## Dockerfile Strategy

Both sub-experiments use a `BASE_IMAGE` build arg:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
```

The notebook resolves the DLC URI at build time via `image_uris.retrieve()` and passes it as `--build-arg BASE_IMAGE=<uri>`. This avoids hardcoding AWS DLC account IDs in Dockerfiles.

## Cleanup

Delete ECR images and SageMaker resources when done:

```bash
# Delete ECR images
aws ecr batch-delete-image --repository-name autogluon-custom --image-ids imageTag=ag150-dlc-upgrade-training
aws ecr batch-delete-image --repository-name autogluon-custom --image-ids imageTag=ag150-dlc-upgrade-inference
aws ecr batch-delete-image --repository-name autogluon-custom --image-ids imageTag=ag150-pytorch-dlc-training
aws ecr batch-delete-image --repository-name autogluon-custom --image-ids imageTag=ag150-pytorch-dlc-inference

# Delete ECR repository (if empty)
aws ecr delete-repository --repository-name autogluon-custom
```

Also delete any SageMaker endpoints, endpoint configs, and models created during testing.
