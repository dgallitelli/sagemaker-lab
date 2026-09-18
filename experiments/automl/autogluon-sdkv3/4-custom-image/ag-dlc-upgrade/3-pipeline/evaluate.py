"""
Evaluation script for SageMaker Processing step.

Loads the trained AutoGluon model and test data, computes metrics,
and writes evaluation.json for pipeline ConditionStep consumption.
"""
import json
import os
import tarfile

from autogluon.tabular import TabularDataset, TabularPredictor

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
    predictor = TabularPredictor.load(extract_dir)
    print(f"Loaded model with problem_type={predictor.problem_type}")

    # Load test data
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
    test_data = TabularDataset(os.path.join(test_dir, test_files[0]))
    print(f"Test data: {len(test_data)} rows")

    # Evaluate
    perf = predictor.evaluate(test_data)
    leaderboard = predictor.leaderboard(test_data, silent=True)

    print(f"Performance: {perf}")
    print(f"Leaderboard:\n{leaderboard}")

    # Write metrics in the format expected by JsonGet in ConditionStep
    eval_metric = predictor.eval_metric.name if hasattr(predictor.eval_metric, "name") else str(predictor.eval_metric)
    # evaluate() returns a dict in AG 1.x; extract the scalar value
    metric_value = perf[eval_metric] if isinstance(perf, dict) else perf
    metrics = {
        "metrics": {
            eval_metric: abs(metric_value),  # abs() because some metrics are negative (e.g., -log_loss)
        }
    }

    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation written to {eval_path}: {metrics}")
