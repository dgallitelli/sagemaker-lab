"""Lambda handler: launch a SageMaker batch Transform job and wait for it.

Invoked synchronously by the deploy pipeline's staging Test stage, and on a
schedule (EventBridge rule) in prod. model_name falls back to the MODEL_NAME
env var (set by batch-transform-template.yml) when the caller's event
payload omits it — the staging Test stage does this deliberately, since it
doesn't know the CFN-generated SageMaker Model name ahead of time.
"""
import os
import time

import boto3

sm_client = boto3.client("sagemaker")

POLL_INTERVAL_SECONDS = 30
MAX_POLL_ATTEMPTS = 120  # 60 minutes


REQUIRED_EVENT_KEYS = ["transform_job_name", "input_s3_uri", "content_type", "output_s3_uri", "instance_type", "instance_count"]


def handler(event, context):
    missing = [key for key in REQUIRED_EVENT_KEYS if key not in event]
    if missing:
        raise ValueError(f"Missing required event parameters: {', '.join(missing)}")

    transform_job_name = event["transform_job_name"]
    model_name = event.get("model_name") or os.environ.get("MODEL_NAME")
    if not model_name:
        raise ValueError("model_name must be provided in the event or the MODEL_NAME environment variable must be set")

    sm_client.create_transform_job(
        TransformJobName=transform_job_name,
        ModelName=model_name,
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": event["input_s3_uri"]}},
            "ContentType": event["content_type"],
        },
        TransformOutput={"S3OutputPath": event["output_s3_uri"]},
        TransformResources={
            "InstanceType": event["instance_type"],
            "InstanceCount": event["instance_count"],
        },
    )

    for _ in range(MAX_POLL_ATTEMPTS):
        response = sm_client.describe_transform_job(TransformJobName=transform_job_name)
        status = response["TransformJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    else:
        raise Exception(f"Transform job {transform_job_name} did not finish within the poll window")

    if status != "Completed":
        raise Exception(f"Transform job {transform_job_name} ended with status {status}: {response.get('FailureReason')}")

    return {"status": status, "transform_job_name": transform_job_name}
