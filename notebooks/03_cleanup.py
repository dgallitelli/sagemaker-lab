"""Tear down TabPFN-3 demo resources.

By default, deletes endpoints whose name starts with `tabpfn3-` plus their
endpoint configs and models. With --delete-image / --delete-weights it also
removes the ECR repository and the S3 model artifact.
"""
from __future__ import annotations

import argparse
import sys

import boto3
from botocore.exceptions import ClientError


def delete_endpoints(sm, prefix: str) -> None:
    paginator = sm.get_paginator("list_endpoints")
    for page in paginator.paginate(NameContains=prefix):
        for ep in page["Endpoints"]:
            name = ep["EndpointName"]
            print(f"Deleting endpoint {name}")
            try:
                sm.delete_endpoint(EndpointName=name)
            except ClientError as exc:
                print(f"  warn: {exc}")
    paginator = sm.get_paginator("list_endpoint_configs")
    for page in paginator.paginate(NameContains=prefix):
        for ec in page["EndpointConfigs"]:
            name = ec["EndpointConfigName"]
            print(f"Deleting endpoint config {name}")
            try:
                sm.delete_endpoint_config(EndpointConfigName=name)
            except ClientError as exc:
                print(f"  warn: {exc}")
    paginator = sm.get_paginator("list_models")
    for page in paginator.paginate(NameContains=prefix):
        for m in page["Models"]:
            name = m["ModelName"]
            print(f"Deleting model {name}")
            try:
                sm.delete_model(ModelName=name)
            except ClientError as exc:
                print(f"  warn: {exc}")


def delete_ecr_repo(region: str, repo: str) -> None:
    ecr = boto3.client("ecr", region_name=region)
    print(f"Deleting ECR repository {repo}")
    try:
        ecr.delete_repository(repositoryName=repo, force=True)
    except ClientError as exc:
        print(f"  warn: {exc}")


def delete_s3_object(bucket: str, key: str) -> None:
    s3 = boto3.client("s3")
    print(f"Deleting s3://{bucket}/{key}")
    try:
        s3.delete_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        print(f"  warn: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--prefix", default="tabpfn3",
                        help="Endpoint/config/model name prefix to match")
    parser.add_argument("--delete-image", action="store_true",
                        help="Also delete the ECR repository")
    parser.add_argument("--ecr-repo", default="tabpfn3-sagemaker")
    parser.add_argument("--delete-weights", action="store_true",
                        help="Also delete the S3 model.tar.gz")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--key", default="tabpfn3/model.tar.gz")
    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    delete_endpoints(sm, args.prefix)

    if args.delete_image:
        delete_ecr_repo(args.region, args.ecr_repo)

    if args.delete_weights:
        if not args.bucket:
            print("--delete-weights requires --bucket", file=sys.stderr)
            return 2
        delete_s3_object(args.bucket, args.key)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
