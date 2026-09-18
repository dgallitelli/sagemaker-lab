"""
Evaluation script for Multimodal SageMaker Processing step.

Loads the trained AutoGluon MultiModalPredictor and test data,
computes metrics, and writes evaluation.json for pipeline ConditionStep.
"""
import json
import os
import tarfile

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

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
    predictor = MultiModalPredictor.load(extract_dir)
    print(f"Loaded model, problem_type={predictor.problem_type}")

    # Load test data
    test_files = [f for f in os.listdir(test_dir) if f.endswith((".jsonl", ".csv"))]
    if not test_files:
        raise FileNotFoundError(f"No .jsonl or .csv files in {test_dir}")
    test_file = os.path.join(test_dir, test_files[0])
    if test_file.endswith(".jsonl"):
        with open(test_file) as f:
            test_data = pd.DataFrame([json.loads(line) for line in f])
    else:
        test_data = pd.read_csv(test_file)
    print(f"Test data: {len(test_data)} rows")

    # Evaluate
    perf = predictor.evaluate(test_data)
    print(f"Performance: {perf}")

    # Write metrics
    # evaluate() returns a dict in AG 1.x; extract the scalar value
    eval_metric = predictor.eval_metric if hasattr(predictor, "eval_metric") else "roc_auc"
    metric_value = perf[eval_metric] if isinstance(perf, dict) else perf
    metrics = {"metrics": {eval_metric: abs(metric_value)}}

    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation written to {eval_path}: {metrics}")
