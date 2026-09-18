"""
Launch standalone SPLADE evaluation on SageMaker.

Takes a model artifact S3 URI and test data S3 URI, runs eval_only.py.

Usage:
  python src/launch_eval_only.py \
    --model-artifact s3://your-bucket/splade-training-output/job-name/output/model.tar.gz \
    --data-prefix splade-data/esci-200k \
    --instance-type ml.g5.2xlarge
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Launch standalone SPLADE evaluation")
    parser.add_argument("--model-artifact", required=True, help="S3 URI to model.tar.gz")
    parser.add_argument("--data-prefix", required=True, help="S3 prefix for test data (e.g., splade-data/esci-200k)")
    parser.add_argument("--s3-bucket", help="S3 bucket (defaults to SageMaker default)")
    parser.add_argument("--instance-type", default="ml.g5.2xlarge")
    parser.add_argument("--role", help="IAM role ARN")
    parser.add_argument("--base-job-name", default="splade-eval")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from sagemaker.train.model_trainer import ModelTrainer
    from sagemaker.train.configs import InputData, Compute, SourceCode, OutputDataConfig, StoppingCondition
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.core.image_uris import retrieve

    # SDK v3.4.1 bug workaround
    import sagemaker.core.workflow.utilities as _wf
    _orig = _wf.get_training_code_hash
    def _patched(entry_point, source_dir, dependencies):
        if dependencies is None:
            dependencies = []
        elif isinstance(dependencies, str):
            dependencies = [dependencies] if dependencies else []
        return _orig(entry_point, source_dir, dependencies)
    _wf.get_training_code_hash = _patched

    session = Session()
    region = session.boto_region_name
    s3_bucket = args.s3_bucket or session.default_bucket()

    # Resolve role
    if args.role:
        role = args.role
    else:
        from sagemaker_launcher import _get_sagemaker_role
        role = _get_sagemaker_role()

    logger.info(f"Region: {region} | Role: {role}")

    try:
        image_uri = retrieve(
            framework="pytorch",
            region=region,
            version="2.6.0",
            py_version="py312",
            instance_type=args.instance_type,
            image_scope="training",
        )
    except Exception:
        image_uri = (
            f"763104351884.dkr.ecr.{region}.amazonaws.com/"
            "pytorch-training:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker"
        )

    s3_training_uri = f"s3://{s3_bucket}/{args.data_prefix.strip('/')}"
    s3_output_uri = f"s3://{s3_bucket}/splade-training-output"

    logger.info(f"Model artifact: {args.model_artifact}")
    logger.info(f"Training data: {s3_training_uri}")
    logger.info(f"Image: {image_uri}")

    if args.dry_run:
        logger.info("DRY RUN — exiting.")
        return

    trainer = ModelTrainer(
        role=role,
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=str(HERE),
            entry_script="eval_only.py",
            requirements="requirements.txt",
        ),
        compute=Compute(
            instance_type=args.instance_type,
            instance_count=1,
            volume_size_in_gb=100,
        ),
        output_data_config=OutputDataConfig(
            s3_output_path=s3_output_uri,
        ),
        hyperparameters={},
        stopping_condition=StoppingCondition(max_runtime_in_seconds=14400),  # 4 hours
        base_job_name=args.base_job_name,
        sagemaker_session=session,
    )

    logger.info("Submitting evaluation job...")
    t0 = time.time()

    trainer.train(
        input_data_config=[
            InputData(
                channel_name="training",
                data_source=s3_training_uri,
            ),
            InputData(
                channel_name="model",
                data_source=args.model_artifact,
            ),
        ],
        wait=True,
        logs=True,
    )

    elapsed = (time.time() - t0) / 60
    logger.info(f"Evaluation complete in {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
