"""
AutoGluon Tabular batch transform inference script.

Accepts headerless CSV input (column order must match training data).
Returns JSON predictions + probabilities.

Based on: autogluon-tabular-containers/scripts/tabular_serve-batch.py
"""
from io import StringIO

import pandas as pd
from autogluon.tabular import TabularPredictor


def model_fn(model_dir):
    """Load AutoGluon TabularPredictor from model directory."""
    model = TabularPredictor.load(model_dir)
    globals()["column_names"] = model.feature_metadata_in.get_features()
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    """Deserialize headerless CSV, predict, return JSON."""
    if input_content_type != "text/csv":
        raise ValueError(f"{input_content_type} content type not supported")

    # request_body may be bytes in DLC containers
    body_str = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    data = pd.read_csv(StringIO(body_str), header=None)
    if len(data.columns) != len(column_names):
        raise ValueError(
            f"Input has {len(data.columns)} columns but model expects {len(column_names)}"
        )
    data.columns = column_names

    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    prediction = pd.concat([pred, pred_proba], axis=1)
    return prediction.to_json(), output_content_type
