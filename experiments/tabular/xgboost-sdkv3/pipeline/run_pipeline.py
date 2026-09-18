#!/usr/bin/env python3
"""CLI Runner for Gas Lift Optimization Pipeline

Commands:
    create   - Create or update the pipeline definition
    run      - Start a new pipeline execution
    describe - Show execution status
    delete   - Delete the pipeline

Usage:
    python run_pipeline.py create --region us-east-1
    python run_pipeline.py run --input-data-uri s3://bucket/data/ --wait
    python run_pipeline.py describe --execution-arn arn:aws:...
    python run_pipeline.py delete
"""

import argparse
import sys
import time
from pipeline import get_pipeline
from sagemaker.core.helper.session_helper import get_execution_role


def create_pipeline(args):
    """Create or update the pipeline definition."""
    print(f"Creating pipeline in {args.region}...")

    role = args.role or get_execution_role()
    pipeline = get_pipeline(
        region=args.region,
        role=role,
        pipeline_name=args.pipeline_name,
    )

    response = pipeline.upsert(role_arn=role)
    print(f"Pipeline ARN: {response['PipelineArn']}")
    print("Pipeline created successfully.")
    return response


def run_pipeline(args):
    """Start a new pipeline execution."""
    print(f"Starting pipeline execution...")

    role = args.role or get_execution_role()
    pipeline = get_pipeline(
        region=args.region,
        role=role,
        pipeline_name=args.pipeline_name,
    )

    # Build execution parameters
    parameters = {}
    if args.input_data_uri:
        parameters["InputDataUri"] = args.input_data_uri
    if args.r2_threshold is not None:
        parameters["R2Threshold"] = args.r2_threshold
    if args.processing_instance_type:
        parameters["ProcessingInstanceType"] = args.processing_instance_type
    if args.training_instance_type:
        parameters["TrainingInstanceType"] = args.training_instance_type
    if args.max_depth is not None:
        parameters["MaxDepth"] = args.max_depth
    if args.n_estimators is not None:
        parameters["NEstimators"] = args.n_estimators
    if args.approval_status:
        parameters["ModelApprovalStatus"] = args.approval_status

    execution = pipeline.start(parameters=parameters if parameters else None)
    print(f"Execution ARN: {execution.arn}")

    if args.wait:
        print("Waiting for execution to complete...")
        execution.wait()
        status = execution.describe()["PipelineExecutionStatus"]
        print(f"Final status: {status}")

        if status == "Failed":
            print("Execution failed. Check CloudWatch logs for details.")
            sys.exit(1)
    else:
        print("Execution started. Use 'describe' command to check status.")

    return execution


def describe_execution(args):
    """Show execution status."""
    from sagemaker.mlops.workflow.pipeline import Pipeline, PipelineExecution
    from sagemaker.core.workflow.pipeline_context import PipelineSession

    pipeline_session = PipelineSession()

    if args.execution_arn:
        # Describe specific execution
        execution = PipelineExecution(
            sagemaker_session=pipeline_session,
            arn=args.execution_arn,
        )
        status = execution.describe()
        print(f"Status: {status['PipelineExecutionStatus']}")
        print(f"Start time: {status.get('CreationTime', 'N/A')}")

        # List steps
        steps = execution.list_steps()
        print("\nSteps:")
        for step in steps["PipelineExecutionSteps"]:
            print(f"  {step['StepName']}: {step['StepStatus']}")
    else:
        # List recent executions
        role = args.role or get_execution_role()
        pipeline = get_pipeline(
            region=args.region,
            role=role,
            pipeline_name=args.pipeline_name,
        )

        executions = pipeline.list_executions()
        print(f"Recent executions for {args.pipeline_name}:")
        for i, ex in enumerate(executions.get("PipelineExecutionSummaries", [])[:5]):
            print(f"  {i+1}. {ex['PipelineExecutionArn']}")
            print(f"     Status: {ex['PipelineExecutionStatus']}")
            print(f"     Started: {ex.get('StartTime', 'N/A')}")
            print()


def delete_pipeline(args):
    """Delete the pipeline."""
    print(f"Deleting pipeline {args.pipeline_name}...")

    role = args.role or get_execution_role()
    pipeline = get_pipeline(
        region=args.region,
        role=role,
        pipeline_name=args.pipeline_name,
    )

    pipeline.delete()
    print("Pipeline deleted.")


def main():
    parser = argparse.ArgumentParser(
        description="Gas Lift Optimization Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global arguments
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--role", type=str, default=None, help="IAM role ARN")
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default="GasLiftOptimizationPipeline",
        help="Pipeline name",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create command
    subparsers.add_parser("create", help="Create or update pipeline")

    # Run command
    run_parser = subparsers.add_parser("run", help="Start pipeline execution")
    run_parser.add_argument("--input-data-uri", type=str, help="S3 URI for input data")
    run_parser.add_argument("--r2-threshold", type=float, help="R2 threshold for quality gate")
    run_parser.add_argument("--processing-instance-type", type=str, help="Processing instance")
    run_parser.add_argument("--training-instance-type", type=str, help="Training instance")
    run_parser.add_argument("--max-depth", type=int, help="XGBoost max_depth")
    run_parser.add_argument("--n-estimators", type=int, help="XGBoost n_estimators")
    run_parser.add_argument("--approval-status", type=str, help="Model approval status")
    run_parser.add_argument("--wait", action="store_true", help="Wait for completion")

    # Describe command
    describe_parser = subparsers.add_parser("describe", help="Show execution status")
    describe_parser.add_argument("--execution-arn", type=str, help="Specific execution ARN")

    # Delete command
    subparsers.add_parser("delete", help="Delete the pipeline")

    args = parser.parse_args()

    if args.command == "create":
        create_pipeline(args)
    elif args.command == "run":
        run_pipeline(args)
    elif args.command == "describe":
        describe_execution(args)
    elif args.command == "delete":
        delete_pipeline(args)


if __name__ == "__main__":
    main()
