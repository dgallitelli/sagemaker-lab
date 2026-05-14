"""Client-side helpers for invoking the TabPFN-3 endpoint.

Four encoding paths, in increasing payload efficiency:

  1. JSON                          (~16 bytes/cell)
  2. JSON + gzip                   (~3-6 bytes/cell on float data)
  3. application/x-npz             (~4 bytes/cell raw, ~2-3 with savez_compressed)
  4. application/x-npz + gzip      (gzip rarely helps after savez_compressed)

The SageMaker boto3 client doesn't expose a Content-Encoding header — the
container sniffs the body for gzip magic bytes (\\x1f\\x8b) and decompresses
transparently. So the client just gzips when it wants to and lets the
container figure it out.
"""
from __future__ import annotations

import gzip
import io
import json
from typing import Any

import numpy as np

JSON_CT = "application/json"
JSON_GZIP_CT = "application/x-json-gzip"
NPZ_CT = "application/x-npz"

# Valid InferenceConfig field names, snapshotted from tabpfn==8.0.2's
# `tabpfn/inference_config.py`. The server-side InferenceConfig is a
# dataclass with `extra="forbid"` semantics — any unknown key triggers a
# Pydantic ValidationError that surfaces as a 500 from the model server.
# Use validate_inference_config() client-side to fail fast and locally.
# When upgrading tabpfn, regenerate via:
#   python -c "from tabpfn.inference_config import InferenceConfig
#              import dataclasses
#              for f in dataclasses.fields(InferenceConfig):
#                  if not f.name.startswith('_'): print(repr(f.name) + ',')"
_INFERENCE_CONFIG_FIELDS = frozenset({
    "PREPROCESS_TRANSFORMS",
    "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES",
    "MIN_UNIQUE_FOR_NUMERICAL_FEATURES",
    "MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE",
    "OUTLIER_REMOVAL_STD",
    "FEATURE_SHIFT_METHOD",
    "CLASS_SHIFT_METHOD",
    "FINGERPRINT_FEATURE",
    "POLYNOMIAL_FEATURES",
    "SUBSAMPLE_SAMPLES",
    "ENABLE_GPU_PREPROCESSING",
    "FEATURE_SUBSAMPLING_METHOD",
    "FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT",
    "FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT",
    "REGRESSION_Y_PREPROCESS_TRANSFORMS",
    "USE_SKLEARN_16_DECIMAL_PRECISION",
    "MAX_NUMBER_OF_CLASSES",
    "MAX_NUMBER_OF_FEATURES",
    "MAX_NUMBER_OF_SAMPLES",
    "FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM",
})


def validate_inference_config(cfg: dict[str, Any] | None) -> None:
    """Reject typo'd InferenceConfig keys before sending to the endpoint.

    The server-side InferenceConfig has `extra="forbid"`, so an unknown key
    crashes the model server with a 500. This local check catches typos
    early with a clean Python TypeError.

    Raises:
        TypeError: if cfg is not a dict (when not None).
        ValueError: if cfg contains keys outside the canonical
            InferenceConfig field set, with a `did_you_mean` suggestion.
    """
    if cfg is None:
        return
    if not isinstance(cfg, dict):
        raise TypeError(f"inference_config must be a dict or None, got {type(cfg).__name__}")
    unknown = set(cfg) - _INFERENCE_CONFIG_FIELDS
    if not unknown:
        return

    import difflib
    msgs = []
    for key in sorted(unknown):
        suggestion = difflib.get_close_matches(key, _INFERENCE_CONFIG_FIELDS, n=1, cutoff=0.6)
        if suggestion:
            msgs.append(f"{key!r} (did you mean {suggestion[0]!r}?)")
        else:
            msgs.append(repr(key))
    raise ValueError(
        f"inference_config contains unknown keys: {', '.join(msgs)}. "
        f"Valid keys: {sorted(_INFERENCE_CONFIG_FIELDS)}"
    )


def encode_json(payload: dict[str, Any], gzip_body: bool = False) -> tuple[bytes, str, str | None]:
    """Returns (body, content_type, content_encoding).

    When gzip_body=True we return Content-Type=application/x-json-gzip so the
    SageMaker toolkit doesn't pre-decode the bytes to UTF-8 (which corrupts the
    gzip stream). The container's input_fn handles the gunzip explicitly.

    Validates `payload["inference_config"]` client-side, so typo'd keys raise
    a local ValueError instead of a 500 from the model server.
    """
    validate_inference_config(payload.get("inference_config"))
    raw = json.dumps(payload).encode()
    if gzip_body:
        return gzip.compress(raw), JSON_GZIP_CT, "gzip"
    return raw, JSON_CT, None


def encode_npz(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    task: str = "classification",
    feature_names: list[str] | None = None,
    return_probabilities: bool = False,
    ignore_pretraining_limits: bool = False,
    inference_config: dict[str, Any] | None = None,
    gzip_body: bool = False,
) -> tuple[bytes, str, str | None]:
    """Pack pure-numeric arrays into a numpy .npz archive. Use this for large
    numeric payloads — ~5-9× smaller wire than JSON for float32 data.

    For mixed numeric+text data, use `encode_npz_mixed` instead.
    """
    validate_inference_config(inference_config)
    meta: dict[str, Any] = {"task": task, "return_probabilities": return_probabilities}
    if feature_names is not None:
        meta["feature_names"] = feature_names
    if ignore_pretraining_limits:
        meta["ignore_pretraining_limits"] = True
    if inference_config is not None:
        meta["inference_config"] = inference_config
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train),
        X_test=np.asarray(X_test, dtype=np.float32),
        meta=np.array(json.dumps(meta)),
    )
    raw = buf.getvalue()
    if gzip_body:
        return gzip.compress(raw), NPZ_CT, "gzip"
    return raw, NPZ_CT, None


def encode_npz_mixed(
    X_train,
    y_train,
    X_test,
    *,
    task: str = "classification",
    feature_names: list[str] | None = None,
    return_probabilities: bool = False,
    ignore_pretraining_limits: bool = False,
    inference_config: dict[str, Any] | None = None,
) -> tuple[bytes, str, str | None]:
    """Pack mixed numeric+categorical data into an NPZ archive.

    Accepts pandas DataFrames, 2-D object ndarrays, or anything DataFrame() can
    consume. Stored as a 2-D object ndarray so per-column dtypes survive the
    round-trip — TabPFN's preprocessing pipeline reconstructs them on the
    server side.

    Inference: ~10× smaller wire than column-dict JSON for the same data.
    """
    validate_inference_config(inference_config)
    import pandas as pd

    def _to_object_2d(x):
        if isinstance(x, pd.DataFrame):
            return x.to_numpy(dtype=object)
        arr = np.asarray(x, dtype=object)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {arr.shape}")
        return arr

    X_train_arr = _to_object_2d(X_train)
    X_test_arr = _to_object_2d(X_test)

    meta: dict[str, Any] = {"task": task, "return_probabilities": return_probabilities}
    if feature_names is not None:
        meta["feature_names"] = feature_names
    elif isinstance(X_train, pd.DataFrame):
        meta["feature_names"] = list(X_train.columns)
    if ignore_pretraining_limits:
        meta["ignore_pretraining_limits"] = True
    if inference_config is not None:
        meta["inference_config"] = inference_config

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        X_train=X_train_arr,
        y_train=np.asarray(y_train),
        X_test=X_test_arr,
        meta=np.array(json.dumps(meta)),
    )
    return buf.getvalue(), NPZ_CT, None


def decode_response(body: bytes, content_type: str) -> dict[str, Any]:
    """Decode whatever the endpoint returned (JSON or .npz)."""
    if content_type.startswith(NPZ_CT):
        with np.load(io.BytesIO(body), allow_pickle=False) as archive:
            out: dict[str, Any] = {
                "predictions": archive["predictions"],
                "probabilities": archive["probabilities"] if "probabilities" in archive.files else None,
                "metadata": json.loads(str(archive["meta"].item())) if "meta" in archive.files else {},
            }
            return out
    return json.loads(body)


def invoke(
    runtime_client,
    endpoint_name: str,
    *,
    body: bytes,
    content_type: str,
    accept: str = JSON_CT,
) -> dict[str, Any]:
    """Thin wrapper around boto3 sagemaker-runtime invoke_endpoint."""
    resp = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Accept=accept,
        Body=body,
    )
    raw = resp["Body"].read()
    return decode_response(raw, resp.get("ContentType", JSON_CT))
