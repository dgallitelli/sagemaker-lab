"""
AutoGluon Tabular inference script for SageMaker real-time endpoints.

Supports CSV, JSON, JSONL, and Parquet input formats.
Returns predictions + probabilities for classification, predictions for regression.

Based on: autogluon-tabular-containers/scripts/tabular_serve.py
"""
from io import BytesIO, StringIO

import pandas as pd
from autogluon.core.constants import REGRESSION
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.tabular import TabularPredictor


def model_fn(model_dir):
    """Load AutoGluon TabularPredictor from model directory."""
    model = TabularPredictor.load(model_dir)
    globals()["column_names"] = model.feature_metadata_in.get_features()
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    """Deserialize input, run prediction, serialize output."""
    # Deserialize — request_body may be bytes in DLC containers
    if isinstance(request_body, bytes):
        request_body_str = request_body.decode("utf-8")
    else:
        request_body_str = request_body

    if input_content_type == "application/x-parquet":
        data = pd.read_parquet(BytesIO(request_body if isinstance(request_body, bytes) else request_body.encode()))
    elif input_content_type == "text/csv":
        data = pd.read_csv(StringIO(request_body_str))
    elif input_content_type == "application/json":
        data = pd.read_json(StringIO(request_body_str))
    elif input_content_type == "application/jsonl":
        data = pd.read_json(StringIO(request_body_str), orient="records", lines=True)
    else:
        raise ValueError(f"{input_content_type} input content type not supported.")

    # Predict
    if model.problem_type != REGRESSION:
        pred_proba = model.predict_proba(data, as_pandas=True)
        pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
        pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
        pred.name = str(pred.name) + "_pred" if pred.name else "pred"
        prediction = pd.concat([pred, pred_proba], axis=1)
    else:
        prediction = model.predict(data, as_pandas=True)

    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()

    # Serialize
    if "application/x-parquet" in output_content_type:
        prediction.columns = prediction.columns.astype(str)
        return prediction.to_parquet(), "application/x-parquet"
    elif "application/json" in output_content_type:
        return prediction.to_json(), "application/json"
    elif "text/csv" in output_content_type:
        return prediction.to_csv(index=None), "text/csv"
    else:
        raise ValueError(f"{output_content_type} content type not supported")
