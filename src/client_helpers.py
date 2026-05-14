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


def encode_json(payload: dict[str, Any], gzip_body: bool = False) -> tuple[bytes, str, str | None]:
    """Returns (body, content_type, content_encoding).

    When gzip_body=True we return Content-Type=application/x-json-gzip so the
    SageMaker toolkit doesn't pre-decode the bytes to UTF-8 (which corrupts the
    gzip stream). The container's input_fn handles the gunzip explicitly.
    """
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
