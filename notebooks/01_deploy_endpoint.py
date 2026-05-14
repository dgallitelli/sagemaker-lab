"""Deploy a TabPFN-3 SageMaker endpoint (real-time OR async).

Usage:
    # Real-time endpoint (sync InvokeEndpoint, 6 MB payload cap, 60s timeout)
    python notebooks/01_deploy_endpoint.py \\
        --image-uri <ecr-uri> \\
        --model-data s3://.../model.tar.gz \\
        --instance-type ml.g5.xlarge

    # Async endpoint (S3-staged payload up to 1 GB, 60 min timeout, scale-to-zero)
    python notebooks/01_deploy_endpoint.py \\
        --image-uri <ecr-uri> \\
        --model-data s3://.../model.tar.gz \\
        --instance-type ml.g5.xlarge \\
        --mode async \\
        --async-output-bucket <bucket>

Note: an async endpoint can ONLY be invoked via InvokeEndpointAsync; sync
InvokeEndpoint calls are rejected by the SageMaker frontend (validation
error, container never sees the request). For "small=sync, large=async"
deploy two endpoints sharing the same image+model artefact.

The image must already be built+pushed (container/build_and_push.sh) and the
model.tar.gz must already be uploaded (scripts/download_and_package_weights.py).
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import boto3
import numpy as np
import sagemaker
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SMOKE_PAYLOAD_SAMPLER = {
    "test_size": 0.2, "random_state": 42,
}


def _smoke_payload() -> tuple[dict, np.ndarray]:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SMOKE_PAYLOAD_SAMPLER["test_size"],
        random_state=SMOKE_PAYLOAD_SAMPLER["random_state"], stratify=y,
    )
    payload = {
        "task": "classification",
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_test": X_test.tolist(),
        "return_probabilities": True,
    }
    return payload, y_test


def _print_smoke_results(response: dict, y_test: np.ndarray) -> None:
    preds = np.asarray(response["predictions"])
    accuracy = float((preds == y_test).mean())
    meta = response["metadata"]
    print(f"  accuracy:        {accuracy:.4f}")
    print(f"  fit_seconds:     {meta['fit_seconds']:.3f}")
    print(f"  predict_seconds: {meta['predict_seconds']:.3f}")
    print(f"  device:          {meta['device']}")
    if meta.get("gpu_memory"):
        print(f"  gpu_memory:      {meta['gpu_memory']}")
    if meta.get("cpu_memory"):
        print(f"  cpu_memory:      {meta['cpu_memory']}")
    print(f"  checkpoint:      {meta.get('model_checkpoint')}")


def _smoke_realtime(predictor) -> None:
    print("\nRunning smoke test (sklearn breast_cancer, classification)...")
    payload, y_test = _smoke_payload()
    response = predictor.predict(payload)
    _print_smoke_results(response, y_test)


def _register_scale_to_zero(endpoint_name: str, region: str, *,
                            min_capacity: int = 0, max_capacity: int = 2,
                            scale_down_after_s: int = 900) -> None:
    """Attach autoscaling to an async endpoint so it scales to zero when idle.

    Two policies wired up:
    1. Target-tracking on ApproximateBacklogSizePerInstance to scale up under
       load (target = 1; one queued item per instance triggers scale-up).
    2. Step-scaling on the HasBacklogWithoutCapacity CloudWatch alarm so the
       endpoint wakes from MinCapacity=0 the moment a request is queued.
       Without this, MinCapacity=0 endpoints would never wake on their own.
    """
    aas = boto3.client("application-autoscaling", region_name=region)
    cw = boto3.client("cloudwatch", region_name=region)
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    scalable_dim = "sagemaker:variant:DesiredInstanceCount"
    namespace = "sagemaker"

    print(f"\nRegistering autoscaling: min={min_capacity} max={max_capacity}")
    aas.register_scalable_target(
        ServiceNamespace=namespace,
        ResourceId=resource_id,
        ScalableDimension=scalable_dim,
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity,
    )

    # 1. Target-tracking on backlog size — handles steady-state load.
    aas.put_scaling_policy(
        PolicyName=f"{endpoint_name}-backlog-target",
        ServiceNamespace=namespace,
        ResourceId=resource_id,
        ScalableDimension=scalable_dim,
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 1.0,
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Statistic": "Average",
            },
            "ScaleInCooldown": scale_down_after_s,
            "ScaleOutCooldown": 60,
        },
    )
    print("  + target-tracking on ApproximateBacklogSizePerInstance (target=1)")

    # 2. Step-scaling on HasBacklogWithoutCapacity — wakes from zero.
    step_policy_name = f"{endpoint_name}-wake-from-zero"
    step_resp = aas.put_scaling_policy(
        PolicyName=step_policy_name,
        ServiceNamespace=namespace,
        ResourceId=resource_id,
        ScalableDimension=scalable_dim,
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity",
            "Cooldown": 60,
            "MetricAggregationType": "Maximum",
            "StepAdjustments": [
                {"MetricIntervalLowerBound": 0, "ScalingAdjustment": 1},
            ],
        },
    )
    cw.put_metric_alarm(
        AlarmName=f"{endpoint_name}-HasBacklogWithoutCapacity",
        MetricName="HasBacklogWithoutCapacity",
        Namespace="AWS/SageMaker",
        Statistic="Average",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        EvaluationPeriods=2,
        DatapointsToAlarm=2,
        Threshold=1,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="missing",
        Period=60,
        AlarmActions=[step_resp["PolicyARN"]],
    )
    print("  + step-scaling alarm HasBacklogWithoutCapacity → +1 instance")
    print(f"  scale-down idle window: {scale_down_after_s}s")


def _smoke_async(endpoint_name: str, region: str, output_bucket: str,
                 input_prefix: str = "tabpfn3/async/inputs") -> None:
    """Upload payload to S3, call InvokeEndpointAsync, poll the result URI."""
    print("\nRunning async smoke test (sklearn breast_cancer, classification)...")
    payload, y_test = _smoke_payload()

    s3 = boto3.client("s3", region_name=region)
    rt = boto3.client("sagemaker-runtime", region_name=region)
    key = f"{input_prefix.strip('/')}/smoke-{uuid.uuid4().hex[:8]}.json"
    s3.put_object(Bucket=output_bucket, Key=key, Body=json.dumps(payload).encode(),
                  ContentType="application/json")
    input_uri = f"s3://{output_bucket}/{key}"
    print(f"  input staged at: {input_uri}")

    t0 = time.perf_counter()
    resp = rt.invoke_endpoint_async(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        InputLocation=input_uri,
    )
    output_uri = resp["OutputLocation"]
    failure_uri = resp.get("FailureLocation")
    print(f"  output expected: {output_uri}")
    print("  polling for result...")

    out_bucket = output_uri.split("/", 3)[2]
    out_key = output_uri.split("/", 3)[3]
    fail_bucket = failure_uri.split("/", 3)[2] if failure_uri else None
    fail_key = failure_uri.split("/", 3)[3] if failure_uri else None

    deadline = time.time() + 600  # 10 min cap on the smoke test
    while time.time() < deadline:
        try:
            obj = s3.get_object(Bucket=out_bucket, Key=out_key)
            body = json.loads(obj["Body"].read())
            elapsed = time.perf_counter() - t0
            print(f"  result after {elapsed:.1f}s")
            _print_smoke_results(body, y_test)
            return
        except s3.exceptions.NoSuchKey:
            pass
        if fail_bucket:
            try:
                err = s3.get_object(Bucket=fail_bucket, Key=fail_key)
                msg = err["Body"].read().decode()
                raise RuntimeError(f"Async invocation failed: {msg[:400]}")
            except s3.exceptions.NoSuchKey:
                pass
        time.sleep(5)
    raise TimeoutError(f"No result at {output_uri} after 10 min")


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--model-data", required=True, help="s3://.../model.tar.gz")
    parser.add_argument("--instance-type", default="ml.g5.xlarge")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--role", default=None,
                        help="SageMaker execution role ARN; default = SageMaker session default")
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument("--source-dir", default=None,
                        help="Path to src/ holding inference.py (default: <repo>/src)")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--mode", choices=["realtime", "async"], default="realtime",
                        help="realtime = sync InvokeEndpoint (6 MB cap, 60s timeout); "
                             "async = InvokeEndpointAsync via S3 (1 GB, 60 min)")
    parser.add_argument("--async-output-bucket", default=None,
                        help="S3 bucket for async output. Required with --mode async.")
    parser.add_argument("--async-output-prefix", default="tabpfn3/async/outputs",
                        help="S3 key prefix for async output objects")
    parser.add_argument("--async-max-concurrent", type=int, default=4,
                        help="MaxConcurrentInvocationsPerInstance for async")
    parser.add_argument("--scale-to-zero", action="store_true",
                        help="(async only) Register autoscaling with MinCapacity=0 "
                             "so the endpoint scales down when idle. Adds a "
                             "step-scaling policy on HasBacklogWithoutCapacity "
                             "to wake from zero on the first queued request.")
    parser.add_argument("--scale-min", type=int, default=0,
                        help="MinCapacity for autoscaling (default 0)")
    parser.add_argument("--scale-max", type=int, default=2,
                        help="MaxCapacity for autoscaling (default 2)")
    parser.add_argument("--scale-down-after", type=int, default=900,
                        help="Seconds of idle before scaling to zero (default 900)")
    args = parser.parse_args()

    if args.mode == "async" and not args.async_output_bucket:
        parser.error("--async-output-bucket is required with --mode async")
    if args.scale_to_zero and args.mode != "async":
        parser.error("--scale-to-zero requires --mode async (realtime endpoints "
                     "do not support MinCapacity=0)")

    sm_session = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role(sm_session)
    suffix = f"{datetime.utcnow():%Y%m%d-%H%M%S}"
    default_name = (f"tabpfn3-{args.mode}-{args.instance_type.replace('.', '-')}-{suffix}"
                    if args.mode == "async"
                    else f"tabpfn3-{args.instance_type.replace('.', '-')}-{suffix}")
    endpoint_name = args.endpoint_name or default_name
    source_dir = args.source_dir or str(Path(__file__).resolve().parent.parent / "src")

    print(f"Mode:          {args.mode}")
    print(f"Region:        {args.region}")
    print(f"Image URI:     {args.image_uri}")
    print(f"Model data:    {args.model_data}")
    print(f"Instance:      {args.instance_type}")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Source dir:    {source_dir}")
    if args.mode == "async":
        print(f"Output bucket: {args.async_output_bucket}")
        print(f"Output prefix: {args.async_output_prefix}")

    model = PyTorchModel(
        image_uri=args.image_uri,
        model_data=args.model_data,
        role=role,
        sagemaker_session=sm_session,
        entry_point="inference.py",
        source_dir=source_dir,
        env={
            "TABPFN_MODEL_CACHE_DIR": "/opt/ml/model/tabpfn_cache",
            # Async endpoints can have very long predict times; use a generous
            # response timeout. Realtime has its own 60s cap from the frontend.
            "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            "TS_DEFAULT_RESPONSE_TIMEOUT": "3600",
            # TorchServe defaults max request/response to 6.5 MB. Async
            # supports up to 1 GB at the SageMaker API level, but the
            # container-side limit is what bites first on 100k+ row payloads.
            # Set both to 1 GB.
            "TS_MAX_REQUEST_SIZE": "1073741824",
            "TS_MAX_RESPONSE_SIZE": "1073741824",
        },
    )

    deploy_kwargs = {
        "initial_instance_count": 1,
        "instance_type": args.instance_type,
        "endpoint_name": endpoint_name,
        "container_startup_health_check_timeout": 600,
    }
    if args.mode == "realtime":
        deploy_kwargs["serializer"] = JSONSerializer()
        deploy_kwargs["deserializer"] = JSONDeserializer()
    else:
        async_cfg = AsyncInferenceConfig(
            output_path=f"s3://{args.async_output_bucket}/{args.async_output_prefix.strip('/')}/",
            max_concurrent_invocations_per_instance=args.async_max_concurrent,
        )
        deploy_kwargs["async_inference_config"] = async_cfg

    print(f"\nDeploying {args.mode} endpoint (this can take 5-10 minutes)...")
    t0 = time.time()
    predictor = model.deploy(**deploy_kwargs)
    print(f"Endpoint InService after {time.time() - t0:.1f}s.")

    if args.scale_to_zero:
        _register_scale_to_zero(
            endpoint_name, args.region,
            min_capacity=args.scale_min,
            max_capacity=args.scale_max,
            scale_down_after_s=args.scale_down_after,
        )

    if args.skip_smoke_test:
        print(f"\nENDPOINT_NAME={endpoint_name}")
        return 0

    if args.mode == "realtime":
        _smoke_realtime(predictor)
    else:
        _smoke_async(endpoint_name, args.region, args.async_output_bucket)

    print(f"\nENDPOINT_NAME={endpoint_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
