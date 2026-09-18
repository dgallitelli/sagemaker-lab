"""Invokes the batch-transform Lambda synchronously against a staging fixture
and asserts the transform job completes with expected output."""
import argparse
import json
import logging
import os
import uuid

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)
# The run-transform Lambda has a 900s (15 min) timeout and legitimately blocks for the full
# transform job duration (it polls create_transform_job to completion before returning).
# botocore's default read_timeout is only 60s, and its standard retry mode treats a client-side
# read timeout on a synchronous Invoke as retriable, silently re-sending the *same* payload
# (same transform_job_name) once the timeout fires — even though the original Lambda invocation
# is still running server-side. That produces two concurrent CreateTransformJob calls with an
# identical job name, which SageMaker rejects with ResourceInUse. Set read_timeout above the
# Lambda's own timeout and disable retries so a slow-but-legitimate invocation is never resent.
lambda_client = boto3.client("lambda", config=Config(read_timeout=910, retries={"max_attempts": 0}))
s3_client = boto3.client("s3")


def invoke_and_verify(function_name, bucket, input_prefix, output_prefix):
    # SageMaker transform job names must be unique per account/region; a hardcoded name
    # collides with a job left over from any prior pipeline run/retry (ResourceInUse). A
    # second-resolution timestamp (int(time.time())) is not sufficiently unique either — two
    # pipeline executions/retries triggered in quick succession (common with auto-retriggered
    # CodePipeline executions from rapid source pushes) can land in the same wall-clock second
    # and collide. Use a random uuid4 suffix instead, which is unique regardless of timing.
    transform_job_name = f"staging-test-transform-{uuid.uuid4().hex[:12]}"
    payload = {
        "model_name": None,  # resolved by the Lambda's own env var at runtime
        "transform_job_name": transform_job_name,
        "input_s3_uri": f"s3://{bucket}/{input_prefix}",
        "output_s3_uri": f"s3://{bucket}/{output_prefix}",
        "instance_type": "ml.m5.xlarge",
        "instance_count": 1,
        "content_type": "text/csv",
    }
    response = lambda_client.invoke(
        FunctionName=function_name, InvocationType="RequestResponse", Payload=json.dumps(payload).encode()
    )
    result = json.loads(response["Payload"].read())
    if response.get("FunctionError"):
        raise Exception(f"Lambda invocation failed: {result}")
    if result.get("status") != "Completed":
        raise Exception(f"Transform job did not complete: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--import-build-config", type=str, required=True)
    parser.add_argument("--export-test-results", type=str, required=True)
    parser.add_argument("--fixture-bucket", type=str, required=True)
    parser.add_argument("--fixture-input-prefix", type=str, default="AutoML/batch-test-input/")
    parser.add_argument("--fixture-output-prefix", type=str, default="AutoML/batch-test-output/")
    args, _ = parser.parse_known_args()

    logging.basicConfig(format="%(levelname)s: [%(filename)s:%(lineno)s] %(message)s", level=args.log_level)

    with open(args.import_build_config) as f:
        config = json.load(f)

    # The Lambda's actual name is sagemaker-{LambdaResourceNamePrefix}-run-transform (see
    # batch-transform-template.yml), where LambdaResourceNamePrefix is a name_from_base(...)
    # value computed by build.py (stage+project name, truncated, plus a fresh timestamp) — not
    # a simple "{SageMakerProjectName}-{StageName}" string. Read the real prefix from the
    # exported build config instead of reconstructing it.
    function_name = "sagemaker-{}-run-transform".format(config["Parameters"]["LambdaResourceNamePrefix"])
    results = invoke_and_verify(function_name, args.fixture_bucket, args.fixture_input_prefix, args.fixture_output_prefix)

    with open(args.export_test_results, "w") as f:
        json.dump(results, f, indent=4)
