"""P1.2 — probe TabPFN-3 row capacity past pretraining limits.

Run with: AWS_PROFILE=<your-profile> python scripts/probe_row_capacity.py \
    --endpoint-name <endpoint> --region us-east-1
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import boto3
import numpy as np
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from client_helpers import NPZ_CT, encode_npz  # noqa: E402

ENDPOINT = "tabpfn3-ml-g5-xlarge-20260514-025834"
REGION = "us-east-1"
N_FEATURES = 32
N_TEST = 200
SEED = 42

runtime = boto3.client("sagemaker-runtime", region_name=REGION)


def make_data(n_train: int, *, seed: int = SEED):
    """Synthetic classification with 32 features, fixed 200-row test set."""
    X, y = make_classification(
        n_samples=n_train + N_TEST,
        n_features=N_FEATURES,
        n_informative=16,
        n_redundant=8,
        n_classes=2,
        random_state=seed,
    )
    # Deterministic split — test is the last N_TEST rows of every dataset
    # generated with the same seed (because make_classification is deterministic).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def invoke(
    n_train: int,
    *,
    ignore_pretraining_limits: bool = False,
    inference_config: dict | None = None,
    return_probabilities: bool = False,
    timeout_note: str = "",
) -> dict:
    X_train, y_train, X_test, y_test = make_data(n_train)
    body, ct, _ = encode_npz(
        X_train, y_train, X_test,
        task="classification",
        return_probabilities=return_probabilities,
    )
    # Patch meta to add ignore_pretraining_limits / inference_config
    # The encode_npz helper doesn't accept these — repack with full meta.
    import io as _io
    meta = {
        "task": "classification",
        "return_probabilities": return_probabilities,
        "ignore_pretraining_limits": ignore_pretraining_limits,
    }
    if inference_config is not None:
        meta["inference_config"] = inference_config
    buf = _io.BytesIO()
    np.savez_compressed(
        buf,
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train),
        X_test=np.asarray(X_test, dtype=np.float32),
        meta=np.array(json.dumps(meta)),
    )
    body = buf.getvalue()
    wire_mb = len(body) / 1e6

    result: dict = {
        "n_train": n_train,
        "wire_mb": round(wire_mb, 3),
        "ignore_pretraining_limits": ignore_pretraining_limits,
        "inference_config": inference_config,
    }
    t0 = time.perf_counter()
    try:
        resp = runtime.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType=NPZ_CT,
            Accept=NPZ_CT,
            Body=body,
        )
        raw = resp["Body"].read()
        # Decode NPZ response
        with np.load(_io.BytesIO(raw), allow_pickle=False) as ar:
            preds = ar["predictions"]
            md = json.loads(str(ar["meta"].item()))
        elapsed = time.perf_counter() - t0
        acc = float((preds == y_test).mean())
        result.update({
            "status": "ok",
            "elapsed_total_s": round(elapsed, 2),
            "fit_s": round(md.get("fit_seconds", 0), 2),
            "predict_s": round(md.get("predict_seconds", 0), 2),
            "peak_gb": round(md.get("gpu_memory", {}).get("peak_gb", 0), 2),
            "accuracy": round(acc, 4),
        })
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        msg = str(e)
        # Look for TabPFNValidationError signature
        kind = "error"
        if "TabPFNValidationError" in msg or "pretraining" in msg.lower():
            kind = "validation_error"
        elif "OutOfMemory" in msg or "out of memory" in msg.lower() or "CUDA" in msg:
            kind = "oom"
        elif "PayloadTooLarge" in msg or "413" in msg:
            kind = "payload_too_large"
        elif "ModelError" in msg:
            kind = "model_error"
        result.update({
            "status": kind,
            "elapsed_total_s": round(elapsed, 2),
            "error": msg[:500],
        })
    print(json.dumps(result, default=str))
    return result


def main():
    out_path = Path(__file__).resolve().parents[1] / "scripts" / "probe_results.jsonl"
    results = []

    def log(rec):
        results.append(rec)
        with out_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # ---- Step 1: validation cap (ignore_pretraining_limits=False)
    print("=== Step 1: find validation cap ===")
    for n in [5_000, 10_000, 20_000, 30_000, 50_000, 60_000]:
        log({"phase": "step1_validation_cap", **invoke(n, ignore_pretraining_limits=False)})

    # ---- Step 2: past-cap with ignore_pretraining_limits=True
    print("=== Step 2: past cap with ignore_pretraining_limits=True ===")
    for n in [10_000, 20_000, 30_000, 50_000, 60_000, 75_000, 100_000]:
        log({"phase": "step2_ignore_limits", **invoke(n, ignore_pretraining_limits=True)})

    # ---- Step 3: subsample-ensemble
    print("=== Step 3: subsample ensemble ===")
    log({"phase": "step3_subsample", **invoke(
        30_000,
        ignore_pretraining_limits=True,
        inference_config={"SUBSAMPLE_SAMPLES": 10000},
    )})
    # vanilla 30k with ignore for direct comparison (already in step 2 but rerun for fresh peak)
    log({"phase": "step3_vanilla", **invoke(30_000, ignore_pretraining_limits=True)})

    print("=== done ===")
    print(f"Wrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
