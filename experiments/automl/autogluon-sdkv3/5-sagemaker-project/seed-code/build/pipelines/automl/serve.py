"""Pluggable AutoGluon real-time inference handler.

Detects the predictor type from the loaded model directory (AutoGluon writes
a `learner.pkl`/predictor-class marker readable via `TabularPredictor.load`
et al. — each predictor class only loads its own artifact shape) and
dispatches accordingly. Tabular path is adapted verbatim from
1-tabular-classification/2-inference/serve.py.
"""
from io import BytesIO, StringIO

import pandas as pd


def model_fn(model_dir):
    """Try each AutoGluon predictor class in turn; the one that loads wins."""
    from autogluon.tabular import TabularPredictor

    try:
        model = TabularPredictor.load(model_dir)
        globals()["task_type"] = "tabular"
        globals()["column_names"] = model.feature_metadata_in.get_features()
        return model
    except Exception:
        pass

    from autogluon.timeseries import TimeSeriesPredictor

    try:
        model = TimeSeriesPredictor.load(model_dir)
        globals()["task_type"] = "timeseries"
        return model
    except Exception:
        pass

    from autogluon.multimodal import MultiModalPredictor

    try:
        model = MultiModalPredictor.load(model_dir)
        globals()["task_type"] = "multimodal"
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load any AutoGluon predictor from {model_dir}: {e}") from e


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    from autogluon.core.constants import REGRESSION
    from autogluon.core.utils import get_pred_from_proba_df

    request_body_str = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    column_names = globals().get("column_names")

    if input_content_type == "application/x-parquet":
        data = pd.read_parquet(BytesIO(request_body if isinstance(request_body, bytes) else request_body.encode()))
    elif input_content_type == "text/csv":
        # This same model/container is used for both real-time invocations (which send a CSV
        # header, matching the training column order) and batch-transform jobs (which send
        # headerless CSV, one row per raw record). Detect which case this is: if the
        # parsed header row doesn't match the model's known training feature names, re-parse as
        # headerless and assign the known column order instead.
        data = pd.read_csv(StringIO(request_body_str))
        if column_names is not None and list(data.columns) != list(column_names):
            data = pd.read_csv(StringIO(request_body_str), header=None)
            if len(data.columns) != len(column_names):
                raise ValueError(f"Input has {len(data.columns)} columns but model expects {len(column_names)}")
            data.columns = column_names
    elif input_content_type == "application/json":
        data = pd.read_json(StringIO(request_body_str))
    elif input_content_type == "application/jsonl":
        data = pd.read_json(StringIO(request_body_str), orient="records", lines=True)
    else:
        raise ValueError(f"{input_content_type} input content type not supported.")

    task_type = globals().get("task_type", "tabular")

    if task_type == "tabular":
        if model.problem_type != REGRESSION:
            pred_proba = model.predict_proba(data, as_pandas=True)
            pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
            pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
            pred.name = str(pred.name) + "_pred" if pred.name else "pred"
            prediction = pd.concat([pred, pred_proba], axis=1)
        else:
            prediction = model.predict(data, as_pandas=True)
    else:
        prediction = model.predict(data)

    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()

    if "application/x-parquet" in output_content_type:
        prediction.columns = prediction.columns.astype(str)
        return prediction.to_parquet(), "application/x-parquet"
    elif "application/json" in output_content_type:
        return prediction.to_json(), "application/json"
    elif "text/csv" in output_content_type:
        return prediction.to_csv(index=None), "text/csv"
    raise ValueError(f"{output_content_type} content type not supported")
