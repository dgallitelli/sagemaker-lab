# DLC base image account is constant across commercial regions; only the
# regional ECR endpoint changes. Override at build time:
#   finch build --build-arg REGION=us-east-1 --build-arg DEVICE=gpu ...
#   finch build --build-arg REGION=us-east-1 --build-arg DEVICE=cpu ...
ARG REGION=us-west-2
ARG DEVICE=gpu
ARG BASE_TAG_GPU=2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker
ARG BASE_TAG_CPU=2.6.0-cpu-py312-ubuntu22.04-sagemaker

FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker AS gpu
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker AS cpu
FROM ${DEVICE}

# Bake heavy deps into the image. Do NOT ship them in source_dir/requirements.txt
# — the SageMaker inference toolkit pip-installs that file on every cold start.
# Pin tabpfn explicitly so silent jumps to 8.1+ (which could change checkpoint
# loading) don't sneak in on rebuilds.
RUN pip install --no-cache-dir \
    'tabpfn==8.0.2' \
    pandas \
    scikit-learn \
    pyarrow

# Do NOT COPY inference.py here. The SageMaker Python SDK injects it into
# model.tar.gz under code/, and the inference toolkit imports it from
# /opt/ml/model/code/ at runtime — the only path it adds to PYTHONPATH.
# Setting SAGEMAKER_PROGRAM / SAGEMAKER_SUBMIT_DIRECTORY here is a no-op
# because the SDK overrides them at deploy time.
