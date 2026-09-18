"""
AutoGluon TimeSeries inference script for SageMaker endpoints.

Supports CSV, JSON, JSONL, and Parquet input formats.
Input must contain columns: [item_id, timestamp, target].

Based on: autogluon-cloud/sagemaker_scripts/timeseries_serve.py
"""
import os
import shutil
from io import BytesIO, StringIO

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def model_fn(model_dir):
    """Load TimeSeriesPredictor from model directory."""
    # AutoGluon TimeSeries needs write access; copy to /tmp
    tmp_model_dir = os.path.join("/tmp", "model")
    if os.path.exists(tmp_model_dir):
        shutil.rmtree(tmp_model_dir)
    shutil.copytree(model_dir, tmp_model_dir)
    model = TimeSeriesPredictor.load(tmp_model_dir)
    if hasattr(model, "persist"):
        model.persist()
    print("Model loaded successfully")
    return model


def prepare_timeseries_dataframe(df, predictor):
    """Convert flat DataFrame to TimeSeriesDataFrame format."""
    target = predictor.target
    cols = df.columns.tolist()
    id_column = cols[0]
    timestamp_column = cols[1]
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Handle static features if present after target column
    static_features = None
    if target != cols[-1]:
        target_index = cols.index(target)
        static_columns = cols[target_index + 1:]
        static_features = df[[id_column] + static_columns].groupby([id_column], sort=False).head(1)
        static_features.set_index(id_column, inplace=True)
        df.drop(columns=static_columns, inplace=True)

    df = TimeSeriesDataFrame.from_data_frame(df, id_column=id_column, timestamp_column=timestamp_column)
    if static_features is not None:
        df.static_features = static_features
    return df


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    """Deserialize input, run forecast, serialize output."""
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

    data = prepare_timeseries_dataframe(data, model)
    prediction = model.predict(data)
    prediction = pd.DataFrame(prediction)

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
