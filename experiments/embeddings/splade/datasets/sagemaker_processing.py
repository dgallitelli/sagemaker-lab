"""
Launch SageMaker Processing jobs for dataset preparation.

Runs prepare_{esci,fiqa,nfcorpus}.py on a SageMaker instance, outputs to S3.
The S3 output feeds directly into the training job.

Usage:
  # ESCI full dataset on ml.m5.4xlarge:
  python datasets/sagemaker_processing.py --dataset esci

  # ESCI 200K subsample:
  python datasets/sagemaker_processing.py --dataset esci --max-pairs 200000

  # ESCI distributed across 5 instances:
  python datasets/sagemaker_processing.py --dataset esci --distributed --instance-count 5

  # FiQA (small, ml.m5.xlarge is enough):
  python datasets/sagemaker_processing.py --dataset fiqa --instance-type ml.m5.xlarge

  # Dry run:
  python datasets/sagemaker_processing.py --dataset esci --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

# Map dataset name to prepare script
DATASET_SCRIPTS = {
    "esci": "prepare_esci.py",
    "fiqa": "prepare_fiqa.py",
    "nfcorpus": "prepare_nfcorpus.py",
}

# Default instance types per dataset
DEFAULT_INSTANCES = {
    "esci": "ml.m5.4xlarge",       # 16 vCPU, 64 GB — handles 1.4M pairs + BM25
    "fiqa": "ml.m5.xlarge",        # 4 vCPU, 16 GB — 57K corpus is small
    "nfcorpus": "ml.m5.xlarge",    # 4 vCPU, 16 GB — 3.6K corpus is tiny
}


def run_processing_job(args: argparse.Namespace) -> str:
    """
    Submit a SageMaker Processing job. Returns the S3 output path.
    """
    from sagemaker.core.processing import FrameworkProcessor, ProcessingOutput
    from sagemaker.core.shapes.shapes import ProcessingS3Output
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.core.image_uris import retrieve

    session = Session()
    region = session.boto_region_name
    bucket = session.default_bucket()

    # Resolve IAM role
    role = args.role or _get_sagemaker_role()

    # Use sklearn container (numpy + pip install support)
    image_uri = retrieve(
        "sklearn", region, version="1.2-1",
        instance_type=args.instance_type,
    )
    logger.info(f"Region: {region} | Role: {role}")
    logger.info(f"Container: {image_uri}")
    logger.info(f"Instance: {args.instance_type}")

    # S3 output path — include suffix to avoid collisions
    suffix = ""
    if args.max_pairs:
        suffix += f"-{args.max_pairs // 1000}k"
    if getattr(args, "distributed", False):
        suffix += "-distributed"
    s3_prefix = f"splade-data/{args.dataset}{suffix}"
    s3_output = f"s3://{bucket}/{s3_prefix}"
    logger.info(f"S3 output: {s3_output}")

    # Select script: distributed wrapper for multi-instance ESCI
    if args.distributed:
        script_name = "prepare_esci_distributed.py"
    else:
        script_name = DATASET_SCRIPTS[args.dataset]

    # Build command-line arguments for the prepare script
    script_args = ["--output-dir", "/opt/ml/processing/output"]
    if args.max_pairs:
        script_args.extend(["--max-pairs", str(args.max_pairs)])
    if args.skip_bm25:
        script_args.append("--skip-bm25")

    instance_count = args.instance_count if args.distributed else 1

    if args.dry_run:
        logger.info(f"DRY RUN — would run {script_name} with args: {script_args}")
        logger.info(f"Instances: {instance_count}x {args.instance_type}")
        logger.info(f"Output would go to: {s3_output}")
        return s3_output

    processor = FrameworkProcessor(
        role=role,
        image_uri=image_uri,
        command=["python3"],
        instance_count=instance_count,
        instance_type=args.instance_type,
        volume_size_in_gb=50,
        max_runtime_in_seconds=args.max_runtime,
        base_job_name=f"splade-prep-{args.dataset}",
        sagemaker_session=session,
    )

    logger.info(f"Submitting processing job for {args.dataset} ({instance_count} instances)...")
    t0 = time.time()

    processor.run(
        code=str(HERE / script_name),
        source_dir=str(HERE),         # uploads common.py alongside the script
        requirements=str(HERE / "requirements.txt"),
        arguments=script_args,
        outputs=[
            ProcessingOutput(
                output_name="data",
                s3_output=ProcessingS3Output(
                    s3_uri=s3_output,
                    local_path="/opt/ml/processing/output",
                    s3_upload_mode="EndOfJob",
                ),
            ),
        ],
        wait=True,
        logs=True,
    )

    elapsed = time.time() - t0
    logger.info(f"Processing job complete in {elapsed / 60:.1f} minutes")
    logger.info(f"Output: {s3_output}")

    return s3_output


def _get_sagemaker_role() -> str:
    import boto3
    from sagemaker.core.helper.session_helper import get_execution_role

    iam = boto3.client("iam")
    try:
        roles = iam.list_roles()["Roles"]
        sm_roles = [
            r for r in roles
            if "SageMaker" in r["RoleName"]
            and "Execution" in r["RoleName"]
            and "service-role" in r["Arn"]
        ]
        if sm_roles:
            return sm_roles[0]["Arn"]
    except Exception:
        pass

    return get_execution_role()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset preparation on SageMaker Processing")
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASET_SCRIPTS.keys()),
        help="Which dataset to prepare",
    )
    parser.add_argument(
        "--instance-type", default=None,
        help="Processing instance type (defaults per dataset)",
    )
    parser.add_argument("--max-pairs", type=int, default=None, help="Max training pairs (ESCI only)")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 baseline (too slow for full ESCI corpus)")
    parser.add_argument("--distributed", action="store_true", help="Use distributed multi-instance processing (ESCI only)")
    parser.add_argument("--instance-count", type=int, default=5, help="Number of instances for distributed mode (default: 5)")
    parser.add_argument("--max-runtime", type=int, default=7200, help="Max runtime in seconds (default: 7200 = 2h)")
    parser.add_argument("--role", help="IAM role ARN (defaults to SageMaker execution role)")
    parser.add_argument("--dry-run", action="store_true", help="Print config but don't submit")
    args = parser.parse_args()

    if args.distributed and args.dataset != "esci":
        parser.error("--distributed is only supported for ESCI dataset")

    if args.instance_type is None:
        args.instance_type = DEFAULT_INSTANCES[args.dataset]

    return args


def main() -> None:
    args = parse_args()
    s3_path = run_processing_job(args)
    print(f"\nData ready at: {s3_path}")
    print(f"Use with training: python src/sagemaker_launcher.py --s3-bucket <inferred> --data-prefix splade-data/{args.dataset}")


if __name__ == "__main__":
    main()
