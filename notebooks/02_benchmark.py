"""Benchmark TabPFN-3 across three GPU instance types.

Four sections, all toggleable via --skip-* flags:

1. Synthetic row-size sweep: validates the VRAM rule of thumb
       VRAM ~ 2.5 GB + (rows * features * 50 bytes)
   by sweeping --row-sizes with synthetic make_classification data.

2. Real-world datasets: 3 classification (breast_cancer, adult_income,
   credit_g) + 3 regression (california_housing, diabetes, bike_sharing).
   Reports accuracy/F1 (clf) and RMSE/R2 (reg) against an XGBoost baseline.

3. Time-series forecasting: Air Passengers framed as supervised regression
   over lag features. Compared against seasonal-naive, linear regression,
   and XGBoost baselines. Metrics: MAE, RMSE, MAPE.

4. Anomaly detection: creditcardfraud (OpenML 1597). TabPFN runs as binary
   classification with predict_proba; baselines are IsolationForest
   (unsupervised) and XGBoost (supervised). Metrics: ROC-AUC, PR-AUC,
   plus accuracy/F1 at the 0.5 threshold.

Local baselines run once (instance-independent); TabPFN runs on each endpoint.

Usage:
    python notebooks/02_benchmark.py \
        --image-uri <ecr-uri> \
        --model-data s3://.../model.tar.gz \
        --instance-types ml.g6e.xlarge,ml.g7e.xlarge,ml.p5.xlarge

Skip flags: --skip-synthetic, --skip-real-datasets, --skip-timeseries,
            --skip-anomaly
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    make_classification,
)
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor


DEFAULT_SIZES = [1_000, 10_000, 100_000]
N_FEATURES = 32

# TabPFN's documented context cap is ~10K training samples. Cap real-world
# datasets to keep the JSON payload (and inference latency) sane.
MAX_TRAIN = 8_000
MAX_TEST = 2_000


# ---------------------------------------------------------------------------
# Section 1: synthetic VRAM sweep
# ---------------------------------------------------------------------------

def estimate_vram_gb(rows: int, features: int) -> float:
    return 2.5 + (rows * features * 50) / 1e9


def synthetic_payload(rows: int, features: int) -> dict[str, Any]:
    X, y = make_classification(
        n_samples=rows,
        n_features=features,
        n_informative=max(2, features // 2),
        random_state=0,
    )
    split = max(1, int(rows * 0.8))
    return {
        "task": "classification",
        "X_train": X[:split].tolist(),
        "y_train": y[:split].tolist(),
        "X_test": X[split:].tolist(),
        "return_probabilities": False,
    }


def run_synthetic(predictor, rows: int, features: int) -> dict[str, Any]:
    payload = synthetic_payload(rows, features)
    t0 = time.perf_counter()
    response = predictor.predict(payload)
    wall = time.perf_counter() - t0
    meta = response["metadata"]
    n_test = meta["n_test"]
    return {
        "rows": rows,
        "features": features,
        "wall_seconds": wall,
        "fit_seconds": meta["fit_seconds"],
        "predict_seconds": meta["predict_seconds"],
        "throughput_rows_per_sec":
            n_test / meta["predict_seconds"] if meta["predict_seconds"] else None,
        "gpu_peak_gb": meta.get("gpu_memory", {}).get("peak_gb"),
        "vram_estimate_gb": estimate_vram_gb(rows, features),
        "device": meta["device"],
    }


# ---------------------------------------------------------------------------
# Section 2: real-world datasets
# ---------------------------------------------------------------------------

def _encode_mixed(X: pd.DataFrame) -> np.ndarray:
    """Convert a pandas frame with mixed dtypes into a float matrix.

    Categorical columns get OrdinalEncoder treatment; both TabPFN and XGBoost
    receive the same numeric matrix so the comparison is fair.
    """
    df = X.copy()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(object))
    return df.astype(np.float32).fillna(0.0).to_numpy()


def _subsample(X: np.ndarray, y: np.ndarray, n: int, *,
               stratify: np.ndarray | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    if stratify is None:
        idx = rng.choice(len(X), size=n, replace=False)
        return X[idx], y[idx]
    # Stratified subsample
    classes, counts = np.unique(stratify, return_counts=True)
    proportions = counts / counts.sum()
    take = (proportions * n).astype(int)
    take[-1] = n - take[:-1].sum()  # adjust for rounding
    chunks: list[int] = []
    for cls, k in zip(classes, take):
        cls_idx = np.where(stratify == cls)[0]
        chunks.extend(rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False))
    chunks_arr = np.asarray(chunks)
    rng.shuffle(chunks_arr)
    return X[chunks_arr], y[chunks_arr]


def load_datasets() -> list[dict[str, Any]]:
    """Return a list of dataset specs.

    Each spec: {name, task, X, y}. Loaded lazily so a missing OpenML cache
    fails one dataset rather than the whole run.
    """
    specs: list[dict[str, Any]] = []

    # ----- classification -----
    bc = load_breast_cancer(as_frame=True)
    specs.append({"name": "breast_cancer", "task": "classification",
                  "X": bc.frame.drop(columns="target"), "y": bc.frame["target"]})

    try:
        adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        y = (adult.target.astype(str).str.strip() == ">50K").astype(int)
        specs.append({"name": "adult_income", "task": "classification",
                      "X": adult.data, "y": y})
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not load adult_income: {exc}")

    try:
        # OpenML 'credit-g' (German credit, dataset 31). Smaller than 42477
        # but well-known and reliably available; 42477 (give-me-some-credit)
        # is enormous and would just get subsampled down anyway.
        credit = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
        y = (credit.target.astype(str) == "good").astype(int)
        specs.append({"name": "credit_g", "task": "classification",
                      "X": credit.data, "y": y})
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not load credit_g: {exc}")

    # ----- regression -----
    cal = fetch_california_housing(as_frame=True)
    specs.append({"name": "california_housing", "task": "regression",
                  "X": cal.frame.drop(columns="MedHouseVal"),
                  "y": cal.frame["MedHouseVal"]})

    diab = load_diabetes(as_frame=True)
    specs.append({"name": "diabetes", "task": "regression",
                  "X": diab.frame.drop(columns="target"),
                  "y": diab.frame["target"]})

    try:
        bike = fetch_openml(data_id=42712, as_frame=True, parser="auto")
        specs.append({"name": "bike_sharing", "task": "regression",
                      "X": bike.data, "y": bike.target.astype(float)})
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not load bike_sharing: {exc}")

    return specs


def prepare_split(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    X = _encode_mixed(spec["X"])
    y = np.asarray(spec["y"])
    stratify = y if spec["task"] == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=stratify,
    )
    X_train, y_train = _subsample(
        X_train, y_train, MAX_TRAIN, stratify=y_train if stratify is not None else None, seed=1,
    )
    X_test, y_test = _subsample(
        X_test, y_test, MAX_TEST, stratify=y_test if stratify is not None else None, seed=2,
    )
    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train,
        "X_test": X_test.astype(np.float32),
        "y_test": y_test,
    }


def score_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    average = "binary" if len(np.unique(y_true)) == 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average)),
    }


def score_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_xgboost(task: str, split: dict[str, np.ndarray]) -> dict[str, Any]:
    if task == "classification":
        # Re-label to 0..K-1 since XGB expects contiguous classes.
        classes = sorted(np.unique(split["y_train"]).tolist())
        idx = {c: i for i, c in enumerate(classes)}
        y_train = np.asarray([idx[v] for v in split["y_train"]])
        y_test = np.asarray([idx[v] for v in split["y_test"]])
        model = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            tree_method="hist", n_jobs=-1, eval_metric="logloss",
        )
        t0 = time.perf_counter()
        model.fit(split["X_train"], y_train)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        preds = model.predict(split["X_test"])
        predict_s = time.perf_counter() - t1
        return {"metrics": score_classification(y_test, preds),
                "fit_seconds": fit_s, "predict_seconds": predict_s}
    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        tree_method="hist", n_jobs=-1,
    )
    t0 = time.perf_counter()
    model.fit(split["X_train"], split["y_train"])
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    preds = model.predict(split["X_test"])
    predict_s = time.perf_counter() - t1
    return {"metrics": score_regression(split["y_test"], preds),
            "fit_seconds": fit_s, "predict_seconds": predict_s}


def run_tabpfn(predictor, task: str, split: dict[str, np.ndarray]) -> dict[str, Any]:
    payload = {
        "task": task,
        "X_train": split["X_train"].tolist(),
        "y_train": split["y_train"].tolist(),
        "X_test": split["X_test"].tolist(),
        "return_probabilities": False,
    }
    t0 = time.perf_counter()
    response = predictor.predict(payload)
    wall = time.perf_counter() - t0
    preds = np.asarray(response["predictions"])
    metrics = (score_classification(split["y_test"], preds) if task == "classification"
               else score_regression(split["y_test"], preds))
    meta = response["metadata"]
    return {
        "metrics": metrics,
        "wall_seconds": wall,
        "fit_seconds": meta["fit_seconds"],
        "predict_seconds": meta["predict_seconds"],
        "gpu_peak_gb": meta.get("gpu_memory", {}).get("peak_gb"),
    }


# ---------------------------------------------------------------------------
# Section 3: time-series forecasting (tabular lag approach)
# ---------------------------------------------------------------------------

# Univariate forecasting framed as supervised regression: features are the
# previous N_LAGS values, target is the next value. This is exactly what the
# TabPFN-TS approach does internally, so no extension package is needed.

DEFAULT_TS_LAGS = 12   # monthly seasonality for Air Passengers
DEFAULT_TS_PERIOD = 12  # for seasonal naive baseline


def create_lagged_features(
    series: np.ndarray, n_lags: int = DEFAULT_TS_LAGS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a univariate time-series to supervised tabular format."""
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i])
        y.append(series[i])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def load_air_passengers() -> np.ndarray:
    """Classic Air Passengers dataset (144 monthly observations, 1949-1960)."""
    data = [
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
    ]
    return np.asarray(data, dtype=np.float32)


def score_timeseries(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    eps = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape,
    }


def seasonal_naive_forecast(
    series: np.ndarray, n_test: int, period: int = DEFAULT_TS_PERIOD,
) -> np.ndarray:
    """Forecast y[t] = y[t - period] over the test tail."""
    history = series[: len(series) - n_test]
    preds = np.empty(n_test, dtype=np.float32)
    for i in range(n_test):
        preds[i] = history[-period + (i % period)] if period <= len(history) else history[-1]
    return preds


def run_timeseries_baselines(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    full_series: np.ndarray, n_test: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    # Linear regression on lag features
    t0 = time.perf_counter()
    lin = LinearRegression().fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    preds = lin.predict(X_test)
    pred_s = time.perf_counter() - t1
    out["linear"] = {"preds": preds, "fit_seconds": fit_s, "predict_seconds": pred_s}

    # XGBoost on the same lag matrix (consistent with rest of notebook)
    t0 = time.perf_counter()
    xgb = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        tree_method="hist", n_jobs=-1,
    ).fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    preds = xgb.predict(X_test)
    pred_s = time.perf_counter() - t1
    out["xgboost"] = {"preds": preds, "fit_seconds": fit_s, "predict_seconds": pred_s}

    # Seasonal naive (no fit)
    out["seasonal_naive"] = {
        "preds": seasonal_naive_forecast(full_series, n_test),
        "fit_seconds": 0.0, "predict_seconds": 0.0,
    }
    return out


def run_timeseries_tabpfn(
    predictor, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
) -> dict[str, Any]:
    payload = {
        "task": "regression",
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_test": X_test.tolist(),
    }
    t0 = time.perf_counter()
    response = predictor.predict(payload)
    wall = time.perf_counter() - t0
    meta = response["metadata"]
    return {
        "preds": np.asarray(response["predictions"], dtype=np.float32),
        "wall_seconds": wall,
        "fit_seconds": meta["fit_seconds"],
        "predict_seconds": meta["predict_seconds"],
        "gpu_peak_gb": meta.get("gpu_memory", {}).get("peak_gb"),
    }


# ---------------------------------------------------------------------------
# Section 4: anomaly detection (labeled-benchmark framing)
# ---------------------------------------------------------------------------

def load_anomaly_dataset() -> dict[str, Any] | None:
    """OpenML credit-card fraud (id=1597): the canonical imbalanced benchmark.

    Returns None if the fetch fails so the rest of the run continues.
    """
    try:
        ds = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not load creditcardfraud: {exc}")
        return None
    X = _encode_mixed(ds.data)
    y = np.asarray(ds.target).astype(int)
    # Stratified subsample so TabPFN's <=10K train context is preserved while
    # keeping the rare class visible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y,
    )
    X_train, y_train = _subsample(X_train, y_train, MAX_TRAIN, stratify=y_train, seed=11)
    X_test, y_test = _subsample(X_test, y_test, MAX_TEST, stratify=y_test, seed=12)
    return {
        "name": "creditcardfraud",
        "X_train": X_train.astype(np.float32),
        "y_train": y_train,
        "X_test": X_test.astype(np.float32),
        "y_test": y_test,
    }


def score_anomaly(y_true: np.ndarray, scores: np.ndarray,
                  hard_preds: np.ndarray | None = None) -> dict[str, float]:
    out = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }
    if hard_preds is not None:
        out["accuracy"] = float(accuracy_score(y_true, hard_preds))
        out["f1"] = float(f1_score(y_true, hard_preds, average="binary"))
    return out


def run_anomaly_baselines(split: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    # IsolationForest: unsupervised, train only on the negative-class subset
    # (treat majority "normal" as the support set, ignore train labels).
    iso_train = split["X_train"][split["y_train"] == 0]
    t0 = time.perf_counter()
    iso = IsolationForest(
        n_estimators=200, contamination="auto", random_state=0, n_jobs=-1,
    ).fit(iso_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    # higher score = more anomalous; sklearn gives the opposite sign in
    # decision_function, so negate.
    scores = -iso.decision_function(split["X_test"])
    preds = (iso.predict(split["X_test"]) == -1).astype(int)
    pred_s = time.perf_counter() - t1
    out["isolation_forest"] = {
        "metrics": score_anomaly(split["y_test"], scores, preds),
        "fit_seconds": fit_s, "predict_seconds": pred_s,
    }

    # XGBoost supervised baseline (uses labels — best-case competitor).
    t0 = time.perf_counter()
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        tree_method="hist", n_jobs=-1, eval_metric="logloss",
    ).fit(split["X_train"], split["y_train"])
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    scores = xgb.predict_proba(split["X_test"])[:, 1]
    preds = (scores >= 0.5).astype(int)
    pred_s = time.perf_counter() - t1
    out["xgboost"] = {
        "metrics": score_anomaly(split["y_test"], scores, preds),
        "fit_seconds": fit_s, "predict_seconds": pred_s,
    }
    return out


def run_anomaly_tabpfn(predictor, split: dict[str, np.ndarray]) -> dict[str, Any]:
    payload = {
        "task": "classification",
        "X_train": split["X_train"].tolist(),
        "y_train": split["y_train"].tolist(),
        "X_test": split["X_test"].tolist(),
        "return_probabilities": True,
    }
    t0 = time.perf_counter()
    response = predictor.predict(payload)
    wall = time.perf_counter() - t0
    preds = np.asarray(response["predictions"]).astype(int)
    proba = np.asarray(response.get("probabilities") or [])
    if proba.size:
        # probability of the positive (anomaly) class
        scores = proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
    else:
        scores = preds.astype(float)
    metrics = score_anomaly(split["y_test"], scores, preds)
    meta = response["metadata"]
    return {
        "metrics": metrics,
        "wall_seconds": wall,
        "fit_seconds": meta["fit_seconds"],
        "predict_seconds": meta["predict_seconds"],
        "gpu_peak_gb": meta.get("gpu_memory", {}).get("peak_gb"),
    }


# ---------------------------------------------------------------------------
# Deploy / orchestrate
# ---------------------------------------------------------------------------

def _build_model(image_uri: str, model_data: str, role: str,
                 sm_session: sagemaker.Session, source_dir: str) -> PyTorchModel:
    return PyTorchModel(
        image_uri=image_uri,
        model_data=model_data,
        role=role,
        sagemaker_session=sm_session,
        entry_point="inference.py",
        source_dir=source_dir,
        env={
            "TABPFN_MODEL_CACHE_DIR": "/opt/ml/model/tabpfn_cache",
            "SAGEMAKER_MODEL_SERVER_TIMEOUT": "120",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            "TS_DEFAULT_RESPONSE_TIMEOUT": "120",
        },
    )


def deploy(image_uri: str, model_data: str, instance_type: str, role: str,
           sm_session: sagemaker.Session, source_dir: str,
           fallback_types: list[str] | None = None):
    """Deploy on `instance_type`. If we hit InsufficientInstanceCapacity, try
    each instance in `fallback_types` in order. Returns (endpoint_name,
    predictor, instance_type_actually_used)."""
    candidates = [instance_type] + list(fallback_types or [])
    last_err: Exception | None = None
    for itype in candidates:
        endpoint_name = (
            f"tabpfn3-{itype.replace('.', '-')}-"
            f"{datetime.utcnow():%Y%m%d-%H%M%S}"
        )
        print(f"\n--- Deploying {itype} as {endpoint_name} ---")
        model = _build_model(image_uri, model_data, role, sm_session, source_dir)
        try:
            predictor = model.deploy(
                initial_instance_count=1,
                instance_type=itype,
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
                container_startup_health_check_timeout=600,
            )
            return endpoint_name, predictor, itype
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            last_err = exc
            if "InsufficientInstanceCapacity" in msg or "capacity" in msg.lower():
                print(f"  ! capacity unavailable for {itype}; trying next fallback")
                # The endpoint exists in Failed state; clean up before retry.
                try:
                    sm_session.delete_endpoint(endpoint_name)
                    sm_session.delete_endpoint_config(endpoint_name)
                except Exception:
                    pass
                continue
            raise
    raise RuntimeError(
        f"All deploys failed (tried {candidates}). Last error: {last_err}"
    )


def attach_predictor(endpoint_name: str, sm_session: sagemaker.Session) -> Predictor:
    return Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )


def print_synthetic_summary(results: dict[str, list[dict[str, Any]]]) -> None:
    print("\n=== Synthetic VRAM sweep ===")
    print(f"{'instance':<22} {'rows':>8} {'wall_s':>8} {'rows/s':>10} "
          f"{'peak_gb':>8} {'est_gb':>8}")
    for itype, runs in results.items():
        for r in runs:
            if "error" in r:
                print(f"{itype:<22} {r['rows']:>8} ERROR: {r['error']}")
                continue
            tput = r.get("throughput_rows_per_sec") or 0.0
            peak = r.get("gpu_peak_gb") or 0.0
            print(
                f"{itype:<22} {r['rows']:>8} {r['wall_seconds']:>8.2f} "
                f"{tput:>10.1f} {peak:>8.2f} {r['vram_estimate_gb']:>8.2f}"
            )


def print_real_summary(real: dict[str, Any]) -> None:
    print("\n=== Real-world datasets: TabPFN-3 vs XGBoost ===")
    header = (f"{'dataset':<22} {'task':<14} {'instance':<22} "
              f"{'metric_a':>10} {'tabpfn':>8} {'xgb':>8} "
              f"{'metric_b':>10} {'tabpfn':>8} {'xgb':>8} {'tp_wall_s':>10}")
    print(header)
    for ds_name, ds in real.items():
        task = ds["task"]
        xgb = ds["xgboost"]["metrics"]
        if task == "classification":
            ma, mb = "accuracy", "f1"
        else:
            ma, mb = "rmse", "r2"
        for itype, run in ds["tabpfn"].items():
            if "error" in run:
                print(f"{ds_name:<22} {task:<14} {itype:<22} ERROR: {run['error']}")
                continue
            t = run["metrics"]
            print(
                f"{ds_name:<22} {task:<14} {itype:<22} "
                f"{ma:>10} {t[ma]:>8.4f} {xgb[ma]:>8.4f} "
                f"{mb:>10} {t[mb]:>8.4f} {xgb[mb]:>8.4f} "
                f"{run['wall_seconds']:>10.2f}"
            )


def print_timeseries_summary(ts: dict[str, Any]) -> None:
    if not ts:
        return
    print("\n=== Time-series forecasting (Air Passengers, lag-feature regression) ===")
    print(f"{'instance':<22} {'model':<18} {'mae':>8} {'rmse':>8} {'mape%':>8}")
    for itype, run in ts.get("tabpfn", {}).items():
        if "error" in run:
            print(f"{itype:<22} {'tabpfn-3':<18} ERROR: {run['error']}")
            continue
        m = run["metrics"]
        print(f"{itype:<22} {'tabpfn-3':<18} "
              f"{m['mae']:>8.2f} {m['rmse']:>8.2f} {m['mape']:>8.2f}")
    for name, run in ts.get("baselines", {}).items():
        m = run["metrics"]
        print(f"{'(local)':<22} {name:<18} "
              f"{m['mae']:>8.2f} {m['rmse']:>8.2f} {m['mape']:>8.2f}")


def print_anomaly_summary(an: dict[str, Any]) -> None:
    if not an:
        return
    print("\n=== Anomaly detection (creditcardfraud) ===")
    print(f"{'instance':<22} {'model':<18} {'roc_auc':>8} {'pr_auc':>8} "
          f"{'f1':>8} {'acc':>8}")
    for itype, run in an.get("tabpfn", {}).items():
        if "error" in run:
            print(f"{itype:<22} {'tabpfn-3':<18} ERROR: {run['error']}")
            continue
        m = run["metrics"]
        print(f"{itype:<22} {'tabpfn-3':<18} "
              f"{m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} "
              f"{m.get('f1', 0):>8.4f} {m.get('accuracy', 0):>8.4f}")
    for name, run in an.get("baselines", {}).items():
        m = run["metrics"]
        print(f"{'(local)':<22} {name:<18} "
              f"{m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} "
              f"{m.get('f1', 0):>8.4f} {m.get('accuracy', 0):>8.4f}")


def _serialisable(obj: Any) -> Any:
    """Strip numpy arrays from nested dicts so json.dump succeeds."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items() if not k.startswith("_")
                and not isinstance(v, np.ndarray)}
    if isinstance(obj, list):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--model-data", required=True)
    parser.add_argument(
        "--instance-types",
        default="ml.g5.xlarge,ml.g6e.xlarge",
        help="Comma-separated GPU instance types to benchmark in order",
    )
    parser.add_argument(
        "--fallback-instance-types",
        default="ml.g5.xlarge,ml.g5.2xlarge",
        help="Comma-separated GPU instances to try if the primary hits "
             "InsufficientInstanceCapacity. Tried in order.",
    )
    parser.add_argument(
        "--row-sizes",
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="Comma-separated synthetic dataset sizes",
    )
    parser.add_argument("--features", type=int, default=N_FEATURES,
                        help="Synthetic feature count")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-real-datasets", action="store_true")
    parser.add_argument("--skip-timeseries", action="store_true")
    parser.add_argument("--skip-anomaly", action="store_true")
    parser.add_argument("--keep-endpoints", action="store_true")
    parser.add_argument("--role", default=None)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Path to src/ holding inference.py (default: <repo>/src)",
    )
    parser.add_argument(
        "--existing-endpoints",
        default=None,
        help=("Comma-separated <instance_type>=<endpoint_name> pairs. "
              "Skips deploy and reuses these endpoints. "
              "Example: ml.g6e.xlarge=tabpfn3-ml-g6e-xlarge-20260513-140833"),
    )
    args = parser.parse_args()

    sm_session = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role(sm_session)
    source_dir = args.source_dir or str(
        __import__("pathlib").Path(__file__).resolve().parent.parent / "src"
    )

    instance_types = [s.strip() for s in args.instance_types.split(",") if s.strip()]
    fallback_types_global = [
        s.strip() for s in args.fallback_instance_types.split(",") if s.strip()
    ]
    row_sizes = [int(s) for s in args.row_sizes.split(",") if s.strip()]

    existing_endpoints: dict[str, str] = {}
    if args.existing_endpoints:
        for pair in args.existing_endpoints.split(","):
            pair = pair.strip()
            if not pair:
                continue
            itype, name = pair.split("=", 1)
            existing_endpoints[itype.strip()] = name.strip()

    # XGBoost baseline runs once per dataset, not per instance.
    real_datasets: dict[str, Any] = {}
    if not args.skip_real_datasets:
        print("\n=== Loading real-world datasets and computing XGBoost baselines ===")
        for spec in load_datasets():
            print(f"  -> {spec['name']} ({spec['task']})")
            try:
                split = prepare_split(spec)
                xgb = run_xgboost(spec["task"], split)
                real_datasets[spec["name"]] = {
                    "task": spec["task"],
                    "n_train": int(len(split["X_train"])),
                    "n_test": int(len(split["X_test"])),
                    "n_features": int(split["X_train"].shape[1]),
                    "_split": split,
                    "xgboost": xgb,
                    "tabpfn": {},
                }
                print(f"     XGBoost: {xgb['metrics']}")
            except Exception as exc:  # noqa: BLE001
                print(f"     ERROR preparing {spec['name']}: {exc}")

    # Time-series prep + local baselines (instance-independent)
    ts_state: dict[str, Any] = {}
    if not args.skip_timeseries:
        print("\n=== Preparing time-series benchmark ===")
        try:
            series = load_air_passengers()
            X_full, y_full = create_lagged_features(series, n_lags=DEFAULT_TS_LAGS)
            split_at = int(len(X_full) * 0.8)
            X_train, y_train = X_full[:split_at], y_full[:split_at]
            X_test, y_test = X_full[split_at:], y_full[split_at:]
            baselines = run_timeseries_baselines(
                X_train, y_train, X_test, full_series=series, n_test=len(y_test),
            )
            ts_state = {
                "name": "air_passengers",
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "n_lags": int(DEFAULT_TS_LAGS),
                "_X_train": X_train, "_y_train": y_train,
                "_X_test": X_test, "_y_test": y_test,
                "baselines": {
                    name: {
                        "metrics": score_timeseries(y_test, run["preds"]),
                        "fit_seconds": run["fit_seconds"],
                        "predict_seconds": run["predict_seconds"],
                    }
                    for name, run in baselines.items()
                },
                "tabpfn": {},
            }
            for name, run in ts_state["baselines"].items():
                print(f"  baseline {name}: {run['metrics']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR preparing time-series: {exc}")
            ts_state = {}

    # Anomaly prep + local baselines (instance-independent)
    an_state: dict[str, Any] = {}
    if not args.skip_anomaly:
        print("\n=== Preparing anomaly detection benchmark ===")
        an = load_anomaly_dataset()
        if an is not None:
            try:
                split = {k: an[k] for k in ("X_train", "y_train", "X_test", "y_test")}
                baselines = run_anomaly_baselines(split)
                an_state = {
                    "name": an["name"],
                    "n_train": int(len(split["X_train"])),
                    "n_test": int(len(split["X_test"])),
                    "n_features": int(split["X_train"].shape[1]),
                    "_split": split,
                    "baselines": baselines,
                    "tabpfn": {},
                }
                for name, run in baselines.items():
                    print(f"  baseline {name}: {run['metrics']}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR preparing anomaly: {exc}")
                an_state = {}

    synthetic_results: dict[str, list[dict[str, Any]]] = {}

    covered_instances: set[str] = set()
    for itype in instance_types:
        if itype in covered_instances:
            print(f"\n--- Skipping {itype}: already covered via fallback ---")
            continue
        endpoint_name = None
        reused_endpoint = False
        # Fallbacks: skip the instance we're trying, and skip any we've
        # already benchmarked so we don't re-do work.
        per_instance_fallbacks = [
            f for f in fallback_types_global
            if f != itype and f not in covered_instances
        ]
        try:
            if itype in existing_endpoints:
                endpoint_name = existing_endpoints[itype]
                reused_endpoint = True
                print(f"\n--- Reusing existing endpoint {endpoint_name} for {itype} ---")
                predictor = attach_predictor(endpoint_name, sm_session)
                actual_itype = itype
            else:
                endpoint_name, predictor, actual_itype = deploy(
                    args.image_uri, args.model_data, itype, role, sm_session,
                    source_dir, fallback_types=per_instance_fallbacks,
                )
                if actual_itype != itype:
                    print(f"  (fell back from {itype} to {actual_itype})")
                    itype = actual_itype  # downstream rows are tagged with the actual instance
            covered_instances.add(itype)

            if not args.skip_synthetic:
                print(f"\n--- Synthetic sweep on {itype} ---")
                runs: list[dict[str, Any]] = []
                for rows in row_sizes:
                    print(f"  rows={rows} features={args.features} ...", flush=True)
                    try:
                        r = run_synthetic(predictor, rows, args.features)
                    except Exception as exc:  # noqa: BLE001
                        r = {"rows": rows, "features": args.features, "error": str(exc)}
                    print(f"    -> {r}")
                    runs.append(r)
                synthetic_results[itype] = runs

            if not args.skip_real_datasets:
                print(f"\n--- Real-world datasets on {itype} ---")
                for ds_name, ds in real_datasets.items():
                    try:
                        run = run_tabpfn(predictor, ds["task"], ds["_split"])
                        ds["tabpfn"][itype] = run
                        print(f"  {ds_name}: {run['metrics']} "
                              f"(wall={run['wall_seconds']:.2f}s)")
                    except Exception as exc:  # noqa: BLE001
                        ds["tabpfn"][itype] = {"error": str(exc)}
                        print(f"  {ds_name}: ERROR {exc}")

            if ts_state:
                print(f"\n--- Time-series on {itype} ---")
                try:
                    tp = run_timeseries_tabpfn(
                        predictor, ts_state["_X_train"], ts_state["_y_train"],
                        ts_state["_X_test"],
                    )
                    metrics = score_timeseries(ts_state["_y_test"], tp["preds"])
                    ts_state["tabpfn"][itype] = {
                        "metrics": metrics,
                        "wall_seconds": tp["wall_seconds"],
                        "fit_seconds": tp["fit_seconds"],
                        "predict_seconds": tp["predict_seconds"],
                        "gpu_peak_gb": tp["gpu_peak_gb"],
                    }
                    print(f"  tabpfn-3: {metrics} (wall={tp['wall_seconds']:.2f}s)")
                except Exception as exc:  # noqa: BLE001
                    ts_state["tabpfn"][itype] = {"error": str(exc)}
                    print(f"  ERROR: {exc}")

            if an_state:
                print(f"\n--- Anomaly detection on {itype} ---")
                try:
                    run = run_anomaly_tabpfn(predictor, an_state["_split"])
                    an_state["tabpfn"][itype] = run
                    print(f"  tabpfn-3: {run['metrics']} "
                          f"(wall={run['wall_seconds']:.2f}s)")
                except Exception as exc:  # noqa: BLE001
                    an_state["tabpfn"][itype] = {"error": str(exc)}
                    print(f"  ERROR: {exc}")
        finally:
            if endpoint_name and not args.keep_endpoints and not reused_endpoint:
                print(f"  Deleting endpoint {endpoint_name}")
                try:
                    sm_session.delete_endpoint(endpoint_name)
                    sm_session.delete_endpoint_config(endpoint_name)
                except Exception as exc:  # noqa: BLE001
                    print(f"  (cleanup warning) {exc}")

    print_synthetic_summary(synthetic_results)
    print_real_summary(real_datasets)
    print_timeseries_summary(ts_state)
    print_anomaly_summary(an_state)

    output = {
        "synthetic": synthetic_results,
        "real_datasets": _serialisable(real_datasets),
        "timeseries": _serialisable(ts_state),
        "anomaly": _serialisable(an_state),
    }
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
