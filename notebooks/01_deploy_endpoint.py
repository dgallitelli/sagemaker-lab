"""Deploy a TabPFN-3 SageMaker real-time endpoint.

Usage:
    python notebooks/01_deploy_endpoint.py \
        --image-uri <ecr-uri> \
        --model-data s3://.../model.tar.gz \
        --instance-type ml.g6e.xlarge

The image must already be built+pushed (container/build_and_push.sh) and the
model.tar.gz must already be uploaded (scripts/download_and_package_weights.py).

Uses PyTorchModel with entry_point + source_dir so the SDK repacks model.tar.gz
to embed src/inference.py at code/inference.py — the only path the SageMaker
inference toolkit imports from at runtime (/opt/ml/model/code/).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--model-data", required=True, help="s3://.../model.tar.gz")
    parser.add_argument("--instance-type", default="ml.g6e.xlarge")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--role", default=None,
                        help="SageMaker execution role ARN; default = SageMaker session default")
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument("--source-dir", default=None,
                        help="Path to src/ holding inference.py (default: <repo>/src)")
    parser.add_argument("--skip-smoke-test", action="store_true")
    args = parser.parse_args()

    sm_session = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role(sm_session)
    endpoint_name = args.endpoint_name or (
        f"tabpfn3-{args.instance_type.replace('.', '-')}-"
        f"{datetime.utcnow():%Y%m%d-%H%M%S}"
    )
    source_dir = args.source_dir or str(Path(__file__).resolve().parent.parent / "src")

    print(f"Region:        {args.region}")
    print(f"Image URI:     {args.image_uri}")
    print(f"Model data:    {args.model_data}")
    print(f"Instance:      {args.instance_type}")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Source dir:    {source_dir}")

    model = PyTorchModel(
        image_uri=args.image_uri,
        model_data=args.model_data,
        role=role,
        sagemaker_session=sm_session,
        entry_point="inference.py",
        source_dir=source_dir,
        env={
            "TABPFN_MODEL_CACHE_DIR": "/opt/ml/model/tabpfn_cache",
            # Survive the first cold prediction (TabPFN JITs kernels on first fit).
            "SAGEMAKER_MODEL_SERVER_TIMEOUT": "120",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            "TS_DEFAULT_RESPONSE_TIMEOUT": "120",
        },
    )

    print("\nDeploying endpoint (this can take 5-10 minutes)...")
    t0 = time.time()
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        container_startup_health_check_timeout=600,
    )
    print(f"Endpoint InService after {time.time() - t0:.1f}s.")

    if args.skip_smoke_test:
        print(f"\nENDPOINT_NAME={endpoint_name}")
        return 0

    print("\nRunning smoke test (sklearn breast_cancer, classification)...")
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    payload = {
        "task": "classification",
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_test": X_test.tolist(),
        "return_probabilities": True,
    }
    response = predictor.predict(payload)
    preds = np.asarray(response["predictions"])
    accuracy = float((preds == y_test).mean())
    meta = response["metadata"]
    print(f"  accuracy:        {accuracy:.4f}")
    print(f"  fit_seconds:     {meta['fit_seconds']:.3f}")
    print(f"  predict_seconds: {meta['predict_seconds']:.3f}")
    print(f"  device:          {meta['device']}")
    print(f"  gpu_memory:      {meta.get('gpu_memory')}")
    print(f"\nENDPOINT_NAME={endpoint_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
