"""SageMaker inference handler for TabPFN-3.

TabPFN is an in-context tabular foundation model: training and test data are
both supplied at inference time and processed in a single forward pass. There
is no separate training job — the "fit" call simply stores the support set
that conditions the next predict call.

Weights are NOT downloaded at runtime. They are baked into the model.tar.gz
SageMaker extracts to /opt/ml/model/ (see model_fn).

Supported request encodings (chosen by Content-Type, optionally gzipped via
Content-Encoding: gzip):

  application/json
      {
        "task": "classification" | "regression",
        "X_train": [[...]] | {"col": [...]},   # row-list (numeric only) or
        "X_test":  [[...]] | {"col": [...]},   # column-dict (mixed types)
        "y_train": [...],
        "feature_names": [...],                # optional
        "return_probabilities": false,         # classification only
        "ignore_pretraining_limits": false,    # opt-in past V3 caps
        "inference_config": {...}              # e.g. {"SUBSAMPLE_SAMPLES": 10000}
      }

  application/x-npz
      A `numpy.savez`/`np.savez_compressed` archive containing arrays
      `X_train`, `y_train`, `X_test`, and optionally a 0-d `meta` array
      whose item is a JSON string with task / return_probabilities /
      feature_names / ignore_pretraining_limits / inference_config. The binary
      path is ~4× smaller than JSON for float32.

Response is always JSON unless Accept: application/x-npz is set, in which
case predictions/probabilities are returned as a numpy archive.
"""
from __future__ import annotations

# IMPORTANT: set TABPFN_* env vars BEFORE importing tabpfn. The library uses
# pydantic-settings, which captures env at import time — assigning os.environ
# afterwards has no effect.
import os
_MODEL_DIR_HINT = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
_TABPFN_CACHE_DIR = os.path.join(_MODEL_DIR_HINT, "tabpfn_cache")
if os.path.isdir(_TABPFN_CACHE_DIR):
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", _TABPFN_CACHE_DIR)
# Belt-and-braces: pin the model version even if the user passes model_path.
os.environ.setdefault("TABPFN_MODEL_VERSION", "v3")
# CPU instances can serve TabPFN too — the library raises by default for
# CPU + >1000 samples; this env var disarms that. Quality/latency on CPU is
# the user's problem (see /tmp/cpu_smoke_test.py).
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")

import gzip
import io
import json
import logging
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

JSON_CT = "application/json"
JSON_GZIP_CT = "application/x-json-gzip"
NPZ_CT = "application/x-npz"

# V3 default checkpoint filenames — must match what's in tabpfn_cache/. The
# library exposes the same constants via ModelSource.get_classifier_v3() but
# we hard-code the strings here to fail loudly if someone re-packages weights
# with a different name.
V3_CLASSIFIER_CKPT = "tabpfn-v3-classifier-v3_default.ckpt"
V3_REGRESSOR_CKPT = "tabpfn-v3-regressor-v3_default.ckpt"


def log_gpu_memory(stage: str = "") -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    logger.info(
        "[%s] GPU memory: allocated=%.2fGB reserved=%.2fGB peak=%.2fGB",
        stage, allocated, reserved, peak,
    )
    return {"allocated_gb": allocated, "reserved_gb": reserved, "peak_gb": peak}


def log_cpu_memory(stage: str = "") -> dict[str, float]:
    """Process resident-set-size (RSS) in GB. Linux ru_maxrss is in KB."""
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    logger.info("[%s] CPU peak RSS: %.2fGB", stage, rss_gb)
    return {"peak_rss_gb": rss_gb}


def model_fn(model_dir: str) -> dict[str, Any]:
    """Initialise classifier and regressor with explicit V3 checkpoint paths.

    SageMaker extracts `model.tar.gz` here. The archive must contain a
    `tabpfn_cache/` directory with the V3 default checkpoints. We pass an
    absolute `model_path` to TabPFN so resolution can't drift to V2/V2.5
    and so the library never falls back to downloading from HuggingFace
    (which would fail in an air-gapped SageMaker container).
    """
    cache_dir = Path(model_dir) / "tabpfn_cache"
    clf_ckpt = cache_dir / V3_CLASSIFIER_CKPT
    reg_ckpt = cache_dir / V3_REGRESSOR_CKPT

    for label, p in (("classifier", clf_ckpt), ("regressor", reg_ckpt)):
        if not p.is_file():
            raise FileNotFoundError(
                f"V3 {label} checkpoint missing at {p}. The model.tar.gz must "
                f"contain `tabpfn_cache/{p.name}`. Refusing to start to avoid "
                "an unintended HuggingFace download."
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading TabPFN V3 on device=%s", device)
    logger.info("classifier weights: %s", clf_ckpt)
    logger.info("regressor weights:  %s", reg_ckpt)

    log_gpu_memory("pre-init")
    classifier = TabPFNClassifier(model_path=str(clf_ckpt), device=device)
    regressor = TabPFNRegressor(model_path=str(reg_ckpt), device=device)
    log_gpu_memory("post-init")

    return {
        "classifier": classifier,
        "regressor": regressor,
        "device": device,
        "classifier_ckpt": str(clf_ckpt),
        "regressor_ckpt": str(reg_ckpt),
    }


def _maybe_gunzip(body: bytes, content_type: str) -> tuple[bytes, str]:
    """Transparently gunzip the body if needed.

    SageMaker's PyTorch toolkit pre-decodes the body to a UTF-8 string when
    Content-Type starts with `application/json` or `text/...`. That destroys
    gzipped JSON payloads. To send gzipped JSON, the client must use a non-text
    content type — we use `application/x-json-gzip`. NPZ already arrives as
    bytes because `application/x-npz` is not a recognised text type."""
    if content_type == JSON_GZIP_CT:
        return gzip.decompress(body), JSON_CT
    # Fallback: sniff gzip magic on bytes payloads (e.g. raw gzipped NPZ).
    if isinstance(body, (bytes, bytearray)) and len(body) >= 2 and body[:2] == b"\x1f\x8b":
        return gzip.decompress(body), content_type
    return body, content_type


def _validate_payload(task: str, payload_keys) -> None:
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    for k in ("X_train", "y_train", "X_test"):
        if k not in payload_keys:
            raise ValueError(f"payload must include X_train, y_train, X_test (missing {k})")


def _coerce_x(raw: Any):
    """Convert payload feature data into a TabPFN-compatible array or DataFrame.

    Two wire formats supported:

    1. Row-oriented list-of-lists (existing pure-numeric JSON path).
       Coerced to float32. Mixed types become string arrays — DON'T send mixed
       data this way unless every cell is the same type.
    2. Column-oriented dict {feature_name: [values, ...]}.
       Returned as a pandas DataFrame so TabPFN sees per-column dtypes; this
       is the right path for mixed numeric+categorical data.
    """
    if isinstance(raw, dict):
        import pandas as pd
        return pd.DataFrame(raw)
    arr = np.asarray(raw)
    if arr.dtype.kind in "fiu":
        return arr.astype(np.float32, copy=False)
    return arr


def input_fn(request_body: str | bytes, content_type: str = JSON_CT) -> dict[str, Any]:
    # SageMaker passes str for text content types and bytes for binary; normalise.
    raw = request_body.encode() if isinstance(request_body, str) else request_body
    raw, content_type = _maybe_gunzip(raw, content_type)

    if content_type == JSON_CT:
        payload = json.loads(raw)
        task = payload.get("task", "classification")
        _validate_payload(task, payload.keys())
        return {
            "task": task,
            "X_train": _coerce_x(payload["X_train"]),
            "y_train": np.asarray(payload["y_train"]),
            "X_test": _coerce_x(payload["X_test"]),
            "feature_names": payload.get("feature_names"),
            "return_probabilities": bool(payload.get("return_probabilities", False)),
            "ignore_pretraining_limits": bool(payload.get("ignore_pretraining_limits", False)),
            "inference_config": payload.get("inference_config"),
        }

    if content_type == NPZ_CT:
        # allow_pickle=True is required for object-dtype arrays. Endpoint is
        # private (in-VPC), payloads are SigV4-signed, so the additional pickle
        # surface is acceptable here. Don't change for public endpoints.
        with np.load(io.BytesIO(raw), allow_pickle=True) as archive:
            keys = set(archive.files)
            meta = {}
            if "meta" in keys:
                meta = json.loads(str(archive["meta"].item()))
                keys.discard("meta")
            task = meta.get("task", "classification")
            _validate_payload(task, keys)
            return {
                "task": task,
                "X_train": _coerce_x(archive["X_train"]),
                "y_train": np.asarray(archive["y_train"]),
                "X_test": _coerce_x(archive["X_test"]),
                "feature_names": meta.get("feature_names"),
                "return_probabilities": bool(meta.get("return_probabilities", False)),
                "ignore_pretraining_limits": bool(meta.get("ignore_pretraining_limits", False)),
                "inference_config": meta.get("inference_config"),
            }

    raise ValueError(f"Unsupported content type: {content_type!r}")


def _maybe_per_request_model(
    base_model, task: str, models: dict[str, Any], data: dict[str, Any],
):
    """Return either the cached model or a freshly-built one with overrides.

    Building a fresh estimator every request is cheap (no weights load, just a
    Python wrapper); the actual checkpoint is shared via the underlying
    PyTorch module cache. We rebuild only when the request asks for overrides.
    """
    overrides = {}
    if data.get("ignore_pretraining_limits"):
        overrides["ignore_pretraining_limits"] = True
    if data.get("inference_config"):
        overrides["inference_config"] = data["inference_config"]
    if not overrides:
        return base_model

    ckpt_path = models["classifier_ckpt"] if task == "classification" else models["regressor_ckpt"]
    cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
    return cls(model_path=ckpt_path, device=models["device"], **overrides)


def predict_fn(data: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    task = data["task"]
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    return_probabilities = data["return_probabilities"]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_gpu_memory("pre-fit")

    base_model = models["classifier"] if task == "classification" else models["regressor"]
    model = _maybe_per_request_model(base_model, task, models, data)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    log_gpu_memory("post-fit")

    t1 = time.perf_counter()
    predictions = model.predict(X_test)
    predict_seconds = time.perf_counter() - t1

    probabilities = None
    if task == "classification" and return_probabilities:
        t2 = time.perf_counter()
        probabilities = model.predict_proba(X_test)
        predict_seconds += time.perf_counter() - t2

    gpu_memory = log_gpu_memory("post-predict")
    cpu_memory = log_cpu_memory("post-predict")

    ckpt_path = (
        models.get("classifier_ckpt") if task == "classification"
        else models.get("regressor_ckpt")
    )

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "metadata": {
            "task": task,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "device": models["device"],
            "gpu_memory": gpu_memory,
            "cpu_memory": cpu_memory,
            "model_checkpoint": ckpt_path,
            "ignore_pretraining_limits": bool(data.get("ignore_pretraining_limits")),
        },
    }


def output_fn(prediction: dict[str, Any], accept: str = JSON_CT) -> tuple[bytes | str, str]:
    accept = (accept or JSON_CT).lower()
    if accept == NPZ_CT:
        buf = io.BytesIO()
        kwargs: dict[str, Any] = {
            "predictions": np.asarray(prediction["predictions"]),
        }
        if prediction.get("probabilities") is not None:
            kwargs["probabilities"] = np.asarray(prediction["probabilities"])
        kwargs["meta"] = np.array(json.dumps(prediction["metadata"]))
        np.savez_compressed(buf, **kwargs)
        return buf.getvalue(), NPZ_CT

    body = {
        "predictions": np.asarray(prediction["predictions"]).tolist(),
        "probabilities": (
            np.asarray(prediction["probabilities"]).tolist()
            if prediction.get("probabilities") is not None
            else None
        ),
        "metadata": prediction["metadata"],
    }
    return json.dumps(body), JSON_CT
