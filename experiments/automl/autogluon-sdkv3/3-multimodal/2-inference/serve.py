"""
AutoGluon Multimodal inference script for SageMaker endpoints.

Accepts CSV input with columns matching the trained feature set
(numerical + categorical + text features, no header for batch).

Based on: churn_prediction_multimodality_of_text_and_tabular/
          containers/autogluon_multimodal_fusion/inference.py
"""
import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from sagemaker_inference import encoder


def model_fn(model_dir: str) -> MultiModalPredictor:
    """Load MultiModalPredictor from model directory."""
    try:
        model = MultiModalPredictor.load(model_dir)

        # Extract column names from df_preprocessor (most reliable source)
        col_names = []
        if hasattr(model, "_df_preprocessor") and model._df_preprocessor is not None:
            dfp = model._df_preprocessor
            for attr in ["numerical_feature_names", "categorical_feature_names", "text_feature_names"]:
                if hasattr(dfp, attr):
                    col_names += getattr(dfp, attr)
            # Fallback: try column_types dict
            if not col_names and hasattr(dfp, "column_types"):
                label = getattr(model, "_label_column", None) or getattr(dfp, "label_column", None)
                col_names = [c for c in dfp.column_types.keys() if c != label]

        # Fallback: try _data_processors
        if not col_names and hasattr(model, "_data_processors"):
            for key in ["numerical", "categorical", "text"]:
                if key in model._data_processors:
                    proc = model._data_processors[key][0]
                    for attr in [f"{key}_column_names", f"{key}_columns"]:
                        if hasattr(proc, attr):
                            val = getattr(proc, attr)
                            if val:
                                col_names += list(val)
                                break

        # Last resort: hardcoded for this model
        if not col_names:
            col_names = ["CustServ Calls", "Account Length", "plan", "limit", "text"]
            logging.warning(f"Using hardcoded column names: {col_names}")

        globals()["column_names"] = col_names
        logging.warning(f"Column names: {col_names}")
        return model
    except Exception:
        logging.exception("Failed to load model")
        raise


def transform_fn(task: MultiModalPredictor, input_data: Any, content_type: str, accept: str) -> np.ndarray:
    """Predict and return serialized probabilities."""
    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")

    if content_type == "text/csv":
        data = pd.read_csv(io.StringIO(input_data), sep=",", header=None)
        data.columns = column_names

        try:
            model_output = task.predict_proba(data).values
            output = {"probabilities": model_output.tolist()}
            return encoder.encode(output, accept)
        except Exception:
            logging.exception("Failed to predict")
            raise

    elif content_type == "application/json":
        data = pd.read_json(io.StringIO(input_data))
        model_output = task.predict_proba(data).values
        output = {"probabilities": model_output.tolist()}
        return encoder.encode(output, accept)

    elif content_type == "application/jsonl":
        data = pd.read_json(io.StringIO(input_data), orient="records", lines=True)
        model_output = task.predict_proba(data).values
        output = {"probabilities": model_output.tolist()}
        return encoder.encode(output, accept)

    raise ValueError(f"{content_type} content type not supported")
