"""
Evaluation script for TimeSeries SageMaker Processing step.

Loads trained AutoGluon TimeSeriesPredictor and evaluates on test data.
Writes evaluation.json for pipeline ConditionStep.
"""
import json
import os
import tarfile

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


if __name__ == "__main__":
    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    output_dir = "/opt/ml/processing/evaluation"

    os.makedirs(output_dir, exist_ok=True)

    # Extract model.tar.gz
    model_tar = os.path.join(model_dir, "model.tar.gz")
    extract_dir = "/opt/ml/processing/model_extracted"
    with tarfile.open(model_tar) as tar:
        tar.extractall(path=extract_dir, filter="data")

    # Load model
    predictor = TimeSeriesPredictor.load(extract_dir)
    print(f"Loaded model, prediction_length={predictor.prediction_length}")

    # Load test data
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
    test_df = pd.read_csv(os.path.join(test_dir, test_files[0]))
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    test_data = TimeSeriesDataFrame.from_data_frame(
        test_df, id_column="item_id", timestamp_column="timestamp",
    )
    print(f"Test data: {len(test_data)} rows, {test_data.num_items} items")

    # Evaluate
    scores = predictor.evaluate(test_data)
    print(f"Test scores: {scores}")

    # Leaderboard
    lb = predictor.leaderboard(test_data, silent=True)
    lb.to_csv(os.path.join(output_dir, "leaderboard.csv"), index=False)

    # Write metrics for pipeline
    eval_metric = predictor.eval_metric
    # evaluate() returns a dict in AG 1.x; extract the scalar value
    score_value = scores[eval_metric] if isinstance(scores, dict) else scores
    metrics = {"metrics": {eval_metric: abs(score_value)}}
    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation written to {eval_path}: {metrics}")
