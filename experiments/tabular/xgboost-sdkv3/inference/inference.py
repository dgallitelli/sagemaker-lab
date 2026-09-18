"""Inference handlers for SageMaker SKLearn container.

This file must be placed in code/ subdirectory of model.tar.gz:
    model.tar.gz/
    ├── model.joblib
    ├── feature_names.json
    └── code/
        ├── inference.py      <-- this file
        └── requirements.txt

Required environment variables when creating the Model:
    SAGEMAKER_PROGRAM: inference.py
    SAGEMAKER_SUBMIT_DIRECTORY: /opt/ml/model/code
"""

import os
import json
import joblib
import numpy as np


def model_fn(model_dir):
    """Load model from model directory.

    Args:
        model_dir: Path to model artifacts (contains model.joblib)

    Returns:
        Loaded model object
    """
    model_path = os.path.join(model_dir, 'model.joblib')
    return joblib.load(model_path)


def input_fn(request_body, request_content_type):
    """Parse input data from request.

    Args:
        request_body: Raw request body
        request_content_type: MIME type of request

    Returns:
        numpy array of input features

    Supported content types:
        - application/json: [[f1, f2, ...], [f1, f2, ...]]
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions using the model.

    Args:
        input_data: numpy array from input_fn
        model: Model object from model_fn

    Returns:
        numpy array of predictions
    """
    return model.predict(input_data)


def output_fn(prediction, response_content_type):
    """Format predictions for response.

    Args:
        prediction: numpy array from predict_fn
        response_content_type: Requested MIME type for response

    Returns:
        Formatted prediction string

    Supported content types:
        - application/json: [pred1, pred2, ...]
    """
    if response_content_type == 'application/json':
        return json.dumps(prediction.tolist())
    raise ValueError(f"Unsupported content type: {response_content_type}")
