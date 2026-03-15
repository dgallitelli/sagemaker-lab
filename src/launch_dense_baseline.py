"""
Launch dense baseline evaluation on SageMaker.

Runs BGE-large-en-v1.5 on all three datasets (FiQA, NFCorpus, ESCI-200k)
in a single training job with multiple input channels.

Usage:
    python src/launch_dense_baseline.py
    python src/launch_dense_baseline.py --instance-type ml.g5.4xlarge
"""

import argparse
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-type", default="ml.g5.12xlarge")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import boto3
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.core.image_uris import retrieve
    from sagemaker.train.configs import (
        Compute,
        InputData,
        OutputDataConfig,
        SourceCode,
        StoppingCondition,
    )
    from sagemaker.train.model_trainer import ModelTrainer

    session = Session()

    # Find a proper SageMaker execution role (SSO roles can't be assumed by SageMaker)
    iam = boto3.client("iam")
    roles = iam.list_roles()["Roles"]
    sm_roles = [
        r for r in roles
        if "SageMaker" in r["RoleName"]
        and "Execution" in r["RoleName"]
        and "service-role" in r["Arn"]
    ]
    if not sm_roles:
        raise RuntimeError("No SageMaker execution role found in account")
    role = sm_roles[0]["Arn"]
    bucket = session.default_bucket()
    region = session.boto_region_name

    logger.info(f"Region: {region} | Role: {role}")

    # PyTorch DLC — same as SPLADE training
    image_uri = retrieve(
        framework="pytorch",
        region=region,
        version="2.6.0",
        py_version="py312",
        instance_type=args.instance_type,
        image_scope="training",
    )
    logger.info(f"Image: {image_uri}")

    # Input channels — all three datasets from S3
    s3_base = f"s3://{bucket}/splade-data"
    input_data = [
        InputData(channel_name="fiqa", data_source=f"{s3_base}/fiqa/"),
        InputData(channel_name="nfcorpus", data_source=f"{s3_base}/nfcorpus/"),
        InputData(channel_name="esci", data_source=f"{s3_base}/esci-200k/"),
    ]

    job_name = f"dense-baseline-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.info(f"Job name: {job_name}")
    logger.info(f"Instance: {args.instance_type}")
    logger.info(f"Channels: fiqa, nfcorpus, esci (esci-200k)")

    if args.dry_run:
        logger.info("DRY RUN — would submit job. Exiting.")
        return

    # SDK v3.4.1 bug workaround: patch get_training_code_hash
    import sagemaker.core.workflow.utilities as _wf
    _orig = _wf.get_training_code_hash
    def _patched(entry_point, source_dir, dependencies):
        if dependencies is None:
            dependencies = []
        elif isinstance(dependencies, str):
            dependencies = [dependencies] if dependencies else []
        return _orig(entry_point, source_dir, dependencies)
    _wf.get_training_code_hash = _patched

    trainer = ModelTrainer(
        training_image=image_uri,
        source_code=SourceCode(
            source_dir="src",
            entry_script="dense_baseline.py",
            requirements="requirements_dense.txt",
        ),
        compute=Compute(
            instance_type=args.instance_type,
            instance_count=1,
        ),
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{bucket}/dense-baseline-output/",
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=14400),
        base_job_name="dense-baseline",
        role=role,
        sagemaker_session=session,
    )

    logger.info("Submitting SageMaker training job...")
    trainer.train(
        input_data_config=input_data,
        wait=False,
        logs=False,
    )
    logger.info(f"Job submitted: {job_name}")
    logger.info("Monitor with: aws sagemaker describe-training-job --training-job-name <name>")


if __name__ == "__main__":
    main()
