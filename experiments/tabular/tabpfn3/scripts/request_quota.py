#!/usr/bin/env python3
"""Check and (if needed) request SageMaker endpoint quotas for TabPFN demo.

Targets:
    ml.g6e.xlarge
    ml.g7e.xlarge
    ml.p5.xlarge

Usage:
    python scripts/request_quota.py --region us-west-2 [--desired 1] [--auto-request]

The script always prints the current quota. With --auto-request it submits a
Service Quotas increase request when the current value is below --desired,
and politely no-ops if a request is already pending.
"""
from __future__ import annotations

import argparse
import sys

import boto3
from botocore.exceptions import ClientError

INSTANCE_TYPES = ["ml.g6e.xlarge", "ml.g7e.xlarge", "ml.p5.xlarge"]
SERVICE_CODE = "sagemaker"


def find_quota(client, instance_type: str) -> dict | None:
    """Look up the 'X for endpoint usage' quota for a given instance type."""
    needle = f"{instance_type} for endpoint usage"
    paginator = client.get_paginator("list_service_quotas")
    for page in paginator.paginate(ServiceCode=SERVICE_CODE):
        for q in page["Quotas"]:
            if q["QuotaName"].lower() == needle.lower():
                return q
    return None


def pending_request(client, quota_code: str) -> dict | None:
    paginator = client.get_paginator("list_requested_service_quota_change_history_by_quota")
    for page in paginator.paginate(ServiceCode=SERVICE_CODE, QuotaCode=quota_code):
        for req in page["RequestedQuotas"]:
            if req["Status"] in ("PENDING", "CASE_OPENED"):
                return req
    return None


def handle_instance(client, instance_type: str, desired: float, auto_request: bool) -> None:
    print(f"\n=== {instance_type} ===")
    quota = find_quota(client, instance_type)
    if quota is None:
        print(f"  Quota entry not found in service '{SERVICE_CODE}'. "
              "Either the quota is not yet exposed via Service Quotas in this region, "
              "or the instance type is unavailable here.")
        return

    quota_code = quota["QuotaCode"]
    current = quota["Value"]
    print(f"  QuotaCode: {quota_code}")
    print(f"  Current:   {current}")

    if current >= desired:
        print(f"  OK: current quota ({current}) >= desired ({desired}).")
        return

    pending = pending_request(client, quota_code)
    if pending is not None:
        print(f"  Pending request already exists (status={pending['Status']}, "
              f"requested={pending['DesiredValue']}). No new request submitted.")
        return

    if not auto_request:
        print(f"  Would request increase to {desired}. Re-run with --auto-request to submit.")
        return

    try:
        resp = client.request_service_quota_increase(
            ServiceCode=SERVICE_CODE,
            QuotaCode=quota_code,
            DesiredValue=desired,
        )
        case_id = resp["RequestedQuota"].get("CaseId", "<no case id>")
        print(f"  Submitted increase request to {desired}. CaseId={case_id}")
    except ClientError as exc:
        print(f"  ERROR submitting request: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--desired", type=float, default=1.0,
                        help="Desired quota value (default: 1)")
    parser.add_argument("--auto-request", action="store_true",
                        help="Submit increase request when current < desired")
    args = parser.parse_args()

    client = boto3.client("service-quotas", region_name=args.region)
    print(f"Region: {args.region}  Desired: {args.desired}  "
          f"AutoRequest: {args.auto_request}")

    for itype in INSTANCE_TYPES:
        handle_instance(client, itype, args.desired, args.auto_request)

    return 0


if __name__ == "__main__":
    sys.exit(main())
