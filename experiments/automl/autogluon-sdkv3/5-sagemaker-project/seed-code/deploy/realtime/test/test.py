"""Smoke test for the staging endpoint: ensure InService, then invoke."""
import argparse
import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")
runtime_client = boto3.client("sagemaker-runtime")


def invoke_endpoint(endpoint_name):
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Accept="application/json",
        Body="age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country\n39,State-gov,77516,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States\n",
    )
    return {"endpoint_name": endpoint_name, "success": True, "body": response["Body"].read().decode("utf-8")}


def test_endpoint(endpoint_name):
    try:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        if status != "InService":
            raise Exception(f"SageMaker endpoint: {endpoint_name} status: {status} not InService")
        return invoke_endpoint(endpoint_name)
    except ClientError as e:
        raise Exception(e.response["Error"]["Message"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--import-build-config", type=str, required=True)
    parser.add_argument("--export-test-results", type=str, required=True)
    args, _ = parser.parse_known_args()

    logging.basicConfig(format="%(levelname)s: [%(filename)s:%(lineno)s] %(message)s", level=args.log_level)

    with open(args.import_build_config) as f:
        config = json.load(f)

    endpoint_name = "{}-{}".format(config["Parameters"]["SageMakerProjectName"], config["Parameters"]["StageName"])
    results = test_endpoint(endpoint_name)

    with open(args.export_test_results, "w") as f:
        json.dump(results, f, indent=4)
