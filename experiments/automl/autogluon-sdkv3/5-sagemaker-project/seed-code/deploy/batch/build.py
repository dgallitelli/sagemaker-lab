"""Reads the latest approved model package and renders staging/prod CFN configs
for the batch-transform deploy pipeline. Adapted from build.py in
5-sagemaker-project/seed-code/deploy/realtime/, extended with the
LambdaCodeS3Bucket/LambdaCodeS3Key parameters batch-transform-template.yml needs."""
import argparse
import json
import logging
import os

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.utils import name_from_base

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


def get_approved_package(model_package_group_name):
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]
        while len(approved_packages) == 0 and "NextToken" in response:
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])
        if len(approved_packages) == 0:
            raise Exception(f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}")
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return model_package_arn
    except ClientError as e:
        raise Exception(e.response["Error"]["Message"])


def get_pipeline_custom_tags(args, new_tags):
    try:
        project_arn = sm_client.describe_project(ProjectName=args.sagemaker_project_name)["ProjectArn"]
        for tag in sm_client.list_tags(ResourceArn=project_arn)["Tags"]:
            new_tags[tag["Key"]] = tag["Value"]
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error getting project tags: {e}")
    return new_tags


def extend_config(args, model_package_arn, stage_config):
    if "Parameters" not in stage_config or "StageName" not in stage_config["Parameters"]:
        raise Exception("Configuration file must include StageName parameter")
    if "Tags" not in stage_config:
        stage_config["Tags"] = {}
    stage_name = stage_config["Parameters"]["StageName"]
    new_params = {
        "SageMakerProjectName": args.sagemaker_project_name,
        "ModelPackageName": model_package_arn,
        "ModelExecutionRoleArn": args.model_execution_role,
        "DataCaptureUploadPath": "s3://" + args.s3_bucket + "/datacapture-" + stage_name,
        "LambdaCodeS3Bucket": args.s3_bucket,
        "LambdaCodeS3Key": f"AutoML/lambda/{stage_name}/run_transform.zip",
        # name_from_base appends a fresh timestamp on every call (no way to seed it
        # deterministically), so this prefix differs on each build.py execution. That's
        # accepted: it keeps the IAM RoleName/Lambda FunctionName/EventBridge Rule Name
        # under their respective 64-char limits regardless of SageMakerProjectName length,
        # at the cost of CloudFormation replacing (not updating in place) those three
        # resources on every deploy. stage_name is prepended (not appended) to the base
        # string because staging and prod are built back-to-back within the same script
        # run and can land in the same millisecond timestamp; name_from_base truncates the
        # base to only `max_length - len(timestamp) - 1` chars (7, given max_length=31), so
        # putting stage_name first guarantees it survives truncation and differentiates the
        # two stages even when args.sagemaker_project_name alone would consume that entire
        # budget. Without this, staging and prod would collide on the account/region-scoped
        # RunTransformLambdaRole/RunTransformLambda/ScheduleRule resource names.
        "LambdaResourceNamePrefix": name_from_base(f"{stage_name}-{args.sagemaker_project_name}", max_length=31),
    }
    new_tags = {
        "sagemaker:deployment-stage": stage_name,
        "sagemaker:project-id": args.sagemaker_project_id,
        "sagemaker:project-name": args.sagemaker_project_name,
    }
    new_tags = get_pipeline_custom_tags(args, new_tags)
    return {
        "Parameters": {**stage_config["Parameters"], **new_params},
        "Tags": {**stage_config.get("Tags", {}), **new_tags},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--model-execution-role", type=str, required=True)
    parser.add_argument("--model-package-group-name", type=str, required=True)
    parser.add_argument("--sagemaker-project-id", type=str, required=True)
    parser.add_argument("--sagemaker-project-name", type=str, required=True)
    parser.add_argument("--s3-bucket", type=str, required=True)
    parser.add_argument("--import-staging-config", type=str, default="staging-config.json")
    parser.add_argument("--import-prod-config", type=str, default="prod-config.json")
    parser.add_argument("--export-staging-config", type=str, default="staging-config-export.json")
    parser.add_argument("--export-prod-config", type=str, default="prod-config-export.json")
    args, _ = parser.parse_known_args()

    logging.basicConfig(format="%(levelname)s: [%(filename)s:%(lineno)s] %(message)s", level=args.log_level)

    model_package_arn = get_approved_package(args.model_package_group_name)

    with open(args.import_staging_config) as f:
        staging_config = extend_config(args, model_package_arn, json.load(f))
    with open(args.export_staging_config, "w") as f:
        json.dump(staging_config, f, indent=4)

    with open(args.import_prod_config) as f:
        prod_config = extend_config(args, model_package_arn, json.load(f))
    with open(args.export_prod_config, "w") as f:
        json.dump(prod_config, f, indent=4)
