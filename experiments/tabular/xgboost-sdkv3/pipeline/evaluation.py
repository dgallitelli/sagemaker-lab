"""Model Evaluation Script for SageMaker Pipeline

Extracts model from model.tar.gz, runs predictions on test data,
and outputs evaluation.json with R2/RMSE/MAE metrics.

Input:
  - /opt/ml/processing/model/model.tar.gz
  - /opt/ml/processing/test/*.csv

Output:
  - /opt/ml/processing/evaluation/evaluation.json
"""

import os
import json
import tarfile
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def extract_model(model_dir: str) -> str:
    """Extract model.tar.gz and return path to model.joblib."""
    tarball = os.path.join(model_dir, "model.tar.gz")

    if not os.path.exists(tarball):
        # Model might already be extracted
        model_path = os.path.join(model_dir, "model.joblib")
        if os.path.exists(model_path):
            return model_path
        raise FileNotFoundError(f"No model.tar.gz or model.joblib in {model_dir}")

    print(f"Extracting {tarball}")
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(path=model_dir)

    return os.path.join(model_dir, "model.joblib")


def load_test_data(test_dir: str, target_column: str = "production"):
    """Load test data and separate features/target."""
    test_path = Path(test_dir)
    csv_files = list(test_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {test_dir}")

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"Test data shape: {df.shape}")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    return X, y


def evaluate_model(model, X, y):
    """Calculate regression metrics."""
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


def save_evaluation(metrics: dict, output_dir: str):
    """Save evaluation.json in format expected by pipeline condition step."""
    os.makedirs(output_dir, exist_ok=True)

    # Format required by JsonGet: regression_metrics.r2.value
    evaluation = {
        "regression_metrics": {
            "r2": {"value": metrics["r2"]},
            "rmse": {"value": metrics["rmse"]},
            "mae": {"value": metrics["mae"]},
        }
    }

    output_path = os.path.join(output_dir, "evaluation.json")
    with open(output_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"Saved evaluation to {output_path}")
    return output_path


def main():
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    # Paths (SageMaker Processing convention)
    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    output_dir = "/opt/ml/processing/evaluation"

    # Extract and load model
    model_path = extract_model(model_dir)
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    # Load test data
    X, y = load_test_data(test_dir, target_column="production")

    # Evaluate
    metrics = evaluate_model(model, X, y)

    # Save
    save_evaluation(metrics, output_dir)

    print("EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
