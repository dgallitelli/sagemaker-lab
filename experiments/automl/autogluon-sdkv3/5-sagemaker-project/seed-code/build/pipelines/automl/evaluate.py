"""Pluggable evaluation script for the AutoML pipeline's ProcessingStep.

Loads a trained AutoGluon predictor (task_type read from the config channel)
and test data, computes metrics, and writes evaluation.json in the shape the
pipeline's ConditionStep expects: {"metrics": {<eval_metric>: <value>}}.
"""
import json
import os
import tarfile

import yaml


def predictor_class_for_task_type(task_type: str) -> str:
    mapping = {
        "tabular": "TabularPredictor",
        "timeseries": "TimeSeriesPredictor",
        "multimodal": "MultiModalPredictor",
    }
    if task_type not in mapping:
        raise ValueError(f"Unknown task_type: {task_type}")
    return mapping[task_type]


def get_input_path(path: str) -> str:
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    if not files:
        raise FileNotFoundError(f"No files found in {path}")
    return os.path.join(path, files[0])


def main() -> None:
    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    config_dir = "/opt/ml/processing/config"
    output_dir = "/opt/ml/processing/evaluation"

    os.makedirs(output_dir, exist_ok=True)

    with open(get_input_path(config_dir)) as f:
        config = yaml.safe_load(f)
    task_type = config["task_type"]

    model_tar = os.path.join(model_dir, "model.tar.gz")
    extract_dir = "/opt/ml/processing/model_extracted"
    with tarfile.open(model_tar) as tar:
        tar.extractall(path=extract_dir, filter="data")

    if task_type == "tabular":
        from autogluon.tabular import TabularDataset, TabularPredictor

        predictor = TabularPredictor.load(extract_dir)
        test_data = TabularDataset(get_input_path(test_dir))
        perf = predictor.evaluate(test_data)
        eval_metric = predictor.eval_metric.name if hasattr(predictor.eval_metric, "name") else str(predictor.eval_metric)

    elif task_type == "timeseries":
        import pandas as pd
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        predictor = TimeSeriesPredictor.load(extract_dir)
        test_df = pd.read_csv(get_input_path(test_dir))
        test_df[config.get("timestamp_column", "timestamp")] = pd.to_datetime(
            test_df[config.get("timestamp_column", "timestamp")]
        )
        test_data = TimeSeriesDataFrame.from_data_frame(
            test_df,
            id_column=config.get("id_column", "item_id"),
            timestamp_column=config.get("timestamp_column", "timestamp"),
        )
        perf = predictor.evaluate(test_data)
        eval_metric = predictor.eval_metric.name if hasattr(predictor.eval_metric, "name") else str(predictor.eval_metric)

    elif task_type == "multimodal":
        import pandas as pd
        from autogluon.multimodal import MultiModalPredictor

        predictor = MultiModalPredictor.load(extract_dir)
        test_file = get_input_path(test_dir)
        if test_file.endswith(".jsonl"):
            with open(test_file) as f:
                test_data = pd.DataFrame([json.loads(line) for line in f])
        else:
            test_data = pd.read_csv(test_file)
        perf = predictor.evaluate(test_data)
        eval_metric = predictor.eval_metric if hasattr(predictor, "eval_metric") else config.get("eval_metric", "roc_auc")

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    metric_value = perf[eval_metric] if isinstance(perf, dict) else perf
    metrics = {"metrics": {eval_metric: abs(metric_value)}}

    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation written to {eval_path}: {metrics}")


if __name__ == "__main__":
    main()
