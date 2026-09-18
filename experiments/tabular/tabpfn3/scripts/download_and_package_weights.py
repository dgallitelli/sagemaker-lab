#!/usr/bin/env python3
"""Download TabPFN-3 weights and package them as a SageMaker model.tar.gz.

Run on a developer machine with TABPFN_TOKEN exported. Weights are downloaded
into a controlled cache directory, archived under `tabpfn_cache/`, and
optionally uploaded to S3.

    python scripts/download_and_package_weights.py \
        --bucket my-bucket --prefix tabpfn3 --region us-west-2

The resulting archive layout:

    model.tar.gz
    └── tabpfn_cache/
        ├── ... (TabPFN model files) ...

At inference time, SageMaker extracts this to /opt/ml/model/, and inference.py
points TABPFN_MODEL_CACHE_DIR at /opt/ml/model/tabpfn_cache.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def trigger_download(cache_dir: Path) -> None:
    """Pull every TabPFN checkpoint into cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(cache_dir)
    if "TABPFN_TOKEN" not in os.environ:
        sys.exit("ERROR: TABPFN_TOKEN not set. `source ~/.zshrc` first.")

    print(f"Downloading TabPFN weights into {cache_dir} ...")
    # Verified against tabpfn 8.0.1: tabpfn.model_loading.download_all_models
    # (to: Path) -> None pulls every classifier+regressor variant for all
    # ModelVersions. If a future version renames the symbol we fall back to
    # estimator instantiation, which triggers the same machinery on first .fit().
    try:
        from tabpfn.model_loading import download_all_models  # noqa: WPS433
        download_all_models(to=cache_dir)
    except ImportError:
        print("download_all_models not available; falling back to estimator init.")
        from tabpfn import TabPFNClassifier, TabPFNRegressor  # noqa: WPS433
        TabPFNClassifier(device="cpu")
        TabPFNRegressor(device="cpu")

    # The downloader prints per-version retry errors to stderr but eventually
    # succeeds for variants we care about. Validate by checking that the V3
    # default checkpoints are present and non-trivially sized.
    required = [
        "tabpfn-v3-classifier-v3_default.ckpt",
        "tabpfn-v3-regressor-v3_default.ckpt",
    ]
    missing = []
    for name in required:
        p = cache_dir / name
        if not p.exists() or p.stat().st_size < 50_000_000:
            missing.append(name)
    if missing:
        sys.exit(
            "ERROR: required V3 checkpoints are missing or truncated: "
            f"{missing}. Likely the TabPFN license has not been accepted at "
            "https://ux.priorlabs.ai (Licenses tab) for the TABPFN_TOKEN owner."
        )
    print("Weights downloaded (V3 defaults verified).")


def package(cache_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Packaging {cache_dir} -> {archive_path}")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(cache_dir, arcname="tabpfn_cache")
    size_mb = archive_path.stat().st_size / 1e6
    print(f"Archive size: {size_mb:.1f} MB")


def upload(archive_path: Path, bucket: str, prefix: str, region: str) -> str:
    key = f"{prefix.strip('/')}/model.tar.gz"
    s3_uri = f"s3://{bucket}/{key}"
    _run(["aws", "s3", "cp", str(archive_path), s3_uri, "--region", region])
    return s3_uri


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="Target S3 bucket")
    parser.add_argument("--prefix", default="tabpfn3", help="S3 key prefix")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "tabpfn-package"),
        help="Local cache directory used as the source for the archive",
    )
    parser.add_argument(
        "--archive",
        default=str(Path(__file__).resolve().parent.parent / "model.tar.gz"),
        help="Output archive path",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing cache without re-downloading",
    )
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    archive_path = Path(args.archive).expanduser().resolve()

    if not args.skip_download:
        if cache_dir.exists():
            print(f"Clearing existing cache at {cache_dir}")
            shutil.rmtree(cache_dir)
        trigger_download(cache_dir)
    else:
        if not cache_dir.exists():
            sys.exit(f"--skip-download set but cache dir {cache_dir} missing")

    package(cache_dir, archive_path)

    if args.skip_upload:
        print(f"Local archive ready at {archive_path}")
        return 0

    s3_uri = upload(archive_path, args.bucket, args.prefix, args.region)
    print(f"\nMODEL_DATA={s3_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
