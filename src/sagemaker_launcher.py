"""
SageMaker training job launcher for SPLADE fine-tuning.

Usage:
  # Run locally (ESCI, full dataset):
  python src/sagemaker_launcher.py --local --data-dir data/esci --no-truncate

  # Run locally (FiQA, truncated for fast iteration):
  python src/sagemaker_launcher.py --local --data-dir data/fiqa

  # Run full SageMaker job:
  python src/sagemaker_launcher.py --s3-bucket my-bucket --data-prefix data/esci

  # Dry run (print config, don't submit):
  python src/sagemaker_launcher.py --s3-bucket my-bucket --dry-run
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


def load_config() -> dict:
    with open(HERE / "config.yaml") as f:
        return yaml.safe_load(f)


def flatten_hyperparameters(config: dict) -> dict:
    """Flatten nested config.yaml into a flat dict for SageMaker."""
    hp = {}
    for section, values in config.items():
        if isinstance(values, dict):
            hp.update(values)
    # SageMaker accepts only str/int/float/bool values
    return {k: v for k, v in hp.items() if not isinstance(v, list)}


# ---------------------------------------------------------------------------
# Local mode
# ---------------------------------------------------------------------------

def run_local(args: argparse.Namespace, config: dict) -> None:
    """
    Run the training script locally without SageMaker.
    Works without AWS credentials. Truncates dataset for fast iteration.
    """
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.local_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate required data files
    required = ["train.jsonl", "test.jsonl", "corpus.jsonl"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        logger.error(f"Missing data files in {data_dir}: {missing}")
        logger.error("Run prepare_dataset.py first to generate the data files.")
        sys.exit(1)

    local_cfg = config.get("local", {})
    if args.no_truncate:
        max_train = "999999"
        max_test = "999999"
        max_corpus = "999999"
    else:
        max_train = str(local_cfg.get("max_train_queries", 500))
        max_test = str(local_cfg.get("max_test_queries", 100))
        max_corpus = str(local_cfg.get("max_corpus_products", 5000))

    env = {
        **os.environ,
        "SM_MODEL_DIR": str(output_dir),
        "SM_CHANNEL_TRAINING": str(data_dir),
        "SM_HPS_PATH": "/dev/null",  # use config.yaml defaults
        "LOCAL_MODE": "1",
        "LOCAL_MAX_TRAIN": max_train,
        "LOCAL_MAX_TEST": max_test,
        "LOCAL_MAX_CORPUS": max_corpus,
        "PYTHONPATH": str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    # Override batch size for CPU/single-GPU local runs
    if not _has_gpu():
        env["SM_HP_BATCH_SIZE"] = str(local_cfg.get("batch_size", 8))
        logger.info("No GPU detected — using smaller batch size for local mode")

    cmd = [sys.executable, str(HERE / "train.py")]
    logger.info(f"Running local training: {' '.join(cmd)}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(
        f"Dataset limits: train={env['LOCAL_MAX_TRAIN']}, "
        f"test={env['LOCAL_MAX_TEST']}, corpus={env['LOCAL_MAX_CORPUS']}"
    )

    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(HERE))
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"Local training failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    logger.info(f"Local training complete in {elapsed / 60:.1f} minutes")
    logger.info(f"Model saved to: {output_dir}")

    # Print eval metrics if saved
    metrics_path = output_dir / "eval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print("\n=== Final Metrics ===")
        print(json.dumps(metrics.get("final", {}), indent=2))


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available() or torch.backends.mps.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# SageMaker mode
# ---------------------------------------------------------------------------

def _get_sagemaker_role() -> str:
    """
    Resolve a SageMaker execution role.
    Prefers an explicit SageMaker Execution role in the account; falls back to
    get_execution_role() which works inside SageMaker notebooks/jobs.
    """
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


def run_sagemaker(args: argparse.Namespace, config: dict) -> None:
    """Submit a SageMaker training job using SDK v3."""
    from sagemaker.train.model_trainer import ModelTrainer
    from sagemaker.train.configs import InputData, Compute, SourceCode, OutputDataConfig
    from sagemaker.core.helper.session_helper import Session, get_execution_role
    from sagemaker.core.image_uris import retrieve

    session = Session()
    region = session.boto_region_name
    role = args.role or _get_sagemaker_role()
    s3_bucket = args.s3_bucket
    data_prefix = args.data_prefix.strip("/")

    logger.info(f"Region: {region} | Role: {role}")
    logger.info(f"S3 bucket: s3://{s3_bucket}/{data_prefix}")

    # ── Upload training data if not already there ────────────────────────────
    if not args.skip_upload:
        _upload_data_to_s3(args.data_dir, s3_bucket, data_prefix, region)

    s3_training_uri = f"s3://{s3_bucket}/{data_prefix}"
    s3_output_uri = f"s3://{s3_bucket}/splade-training-output"

    # ── PyTorch 2.2 GPU DLC ──────────────────────────────────────────────────
    try:
        image_uri = retrieve(
            framework="pytorch",
            region=region,
            version="2.2.0",
            py_version="py310",
            instance_type="ml.g5.12xlarge",
            image_scope="training",
        )
    except Exception:
        # Fallback to known-good URI if retrieve() fails
        image_uri = (
            f"763104351884.dkr.ecr.{region}.amazonaws.com/"
            "pytorch-training:2.2.0-gpu-py310-cu118-ubuntu20.04-sagemaker"
        )
    logger.info(f"Training image: {image_uri}")

    hp = flatten_hyperparameters(config)
    logger.info(f"Hyperparameters: {hp}")

    # ── Metric definitions for CloudWatch ────────────────────────────────────
    # Parses JSON-formatted metric lines from stdout
    metric_definitions = [
        {"Name": "final/ndcg@10",    "Regex": r'"metric_name": "final/ndcg@10",\s*"value": ([0-9.]+)'},
        {"Name": "final/recall@100", "Regex": r'"metric_name": "final/recall@100",\s*"value": ([0-9.]+)'},
        {"Name": "final/mrr@10",     "Regex": r'"metric_name": "final/mrr@10",\s*"value": ([0-9.]+)'},
        {"Name": "train/loss",       "Regex": r'"metric_name": "phase1_easy/train_loss",\s*"value": ([0-9.]+)'},
    ]

    if args.dry_run:
        logger.info("DRY RUN — would submit job with above config. Exiting.")
        return

    # ── Submit training job ──────────────────────────────────────────────────
    # SDK v3.4.1 bug: requirements passed as str causes get_training_code_hash() to fail.
    # Workaround: patch the utility before importing sagemaker (applied below).
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
        role=role,
        training_image=image_uri,
        source_code=SourceCode(
            source_dir=str(HERE),
            entry_script="train.py",       # v3 uses entry_script, not entry_point
            requirements="requirements.txt",
        ),
        compute=Compute(
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            volume_size_in_gb=100,
        ),
        output_data_config=OutputDataConfig(
            s3_output_path=s3_output_uri
        ),
        hyperparameters=hp,
        metric_definitions=metric_definitions,
        base_job_name="splade-esci",
        sagemaker_session=session,
    )

    logger.info("Submitting SageMaker training job...")
    t0 = time.time()

    trainer.train(
        input_data_config=[
            InputData(
                channel_name="training",
                data_source=s3_training_uri,
            )
        ],
        wait=True,   # wait and logs go on train(), not ModelTrainer()
        logs=True,
    )

    elapsed_min = (time.time() - t0) / 60
    _print_cost_estimate(elapsed_min, spot=True)


def _upload_data_to_s3(data_dir: str, bucket: str, prefix: str, region: str) -> None:
    """Upload JSONL data files to S3."""
    import boto3

    data_path = Path(data_dir)
    s3 = boto3.client("s3", region_name=region)

    files = ["train.jsonl", "test.jsonl", "corpus.jsonl", "bm25_baseline_results.json"]
    for fname in files:
        local_path = data_path / fname
        if not local_path.exists():
            if fname == "bm25_baseline_results.json":
                continue  # optional
            logger.error(f"Required file not found: {local_path}")
            sys.exit(1)
        s3_key = f"{prefix}/{fname}"
        logger.info(f"Uploading {local_path} → s3://{bucket}/{s3_key}")
        s3.upload_file(str(local_path), bucket, s3_key)


def _print_cost_estimate(duration_minutes: float, spot: bool = True) -> None:
    """Print rough cost estimate for ml.g5.12xlarge."""
    # ml.g5.12xlarge on-demand: ~$5.672/hr; spot: ~60% discount
    on_demand_hourly = 5.672
    rate = on_demand_hourly * (0.40 if spot else 1.0)
    cost = rate * (duration_minutes / 60)
    mode = "spot" if spot else "on-demand"
    print(f"\nEstimated cost ({mode}): ${cost:.2f} USD ({duration_minutes:.0f} min at ${rate:.3f}/hr)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch SPLADE training job")

    parser.add_argument("--local", action="store_true", help="Run locally without SageMaker")
    parser.add_argument("--no-truncate", action="store_true", help="Use full dataset in local mode (no truncation)")
    parser.add_argument("--data-dir", default="data/esci", help="Local data directory (e.g., data/esci, data/fiqa)")
    parser.add_argument("--local-output", default="./local_model_output", help="Local output dir")

    # SageMaker args
    parser.add_argument("--s3-bucket", help="S3 bucket for data and model artifacts")
    parser.add_argument("--data-prefix", default="splade-esci/data", help="S3 prefix for training data")
    parser.add_argument("--role", help="IAM role ARN (defaults to execution role)")
    parser.add_argument("--skip-upload", action="store_true", help="Skip data upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print config but don't submit job")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    if args.local:
        run_local(args, config)
    else:
        if not args.s3_bucket:
            logger.error("--s3-bucket is required for SageMaker mode. Use --local for local testing.")
            sys.exit(1)
        run_sagemaker(args, config)


if __name__ == "__main__":
    main()
