# NOTE: This script is identical to 1-tabular-classification/1-training/train.py
"""
AutoGluon Tabular training script for SageMaker.

Runs inside the AutoGluon DLC container. Reads config from the 'config' channel,
training data from 'train', optional test data from 'test', and an optional
serving script from 'serving'.

"""
import argparse
import json
import os
import shutil
from pprint import pprint

import yaml
from autogluon.tabular import TabularDataset, TabularPredictor


def get_input_path(path: str) -> str:
    """Return the first file found in the given directory."""
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    if not files:
        raise FileNotFoundError(f"No files found in {path}")
    if len(files) > 1:
        print(f"WARN: multiple files found in {path}, using first: {files[0]}")
    filename = os.path.join(path, files[0])
    print(f"Using {filename}")
    return filename


def get_env(name: str):
    return os.environ.get(name)


if __name__ == "__main__":
    print("Starting AutoGluon Tabular Training")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-dir", type=str, default=get_env("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=get_env("SM_MODEL_DIR"))
    parser.add_argument("--n_gpus", type=int, default=int(get_env("SM_NUM_GPUS") or 0))
    parser.add_argument("--train_dir", type=str, default=get_env("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test_dir", type=str, required=False, default=get_env("SM_CHANNEL_TEST"))
    parser.add_argument("--ag_config", type=str, default=get_env("SM_CHANNEL_CONFIG"))
    parser.add_argument("--serving_script", type=str, default=get_env("SM_CHANNEL_SERVING"))
    args, _ = parser.parse_known_args()
    print(f"Args: {args}")

    os.makedirs(args.output_data_dir, exist_ok=True)

    # Load YAML config
    config_file = get_input_path(args.ag_config)
    with open(config_file) as f:
        config = yaml.safe_load(f)

    print("Training config:")
    pprint(config)

    # Load training data
    train_file = get_input_path(args.train_dir)
    train_data = TabularDataset(train_file)

    # Train
    save_path = os.path.normpath(args.model_dir)
    ag_predictor_args = config["ag_predictor_args"]
    ag_predictor_args["path"] = save_path
    ag_fit_args = config["ag_fit_args"]

    predictor = TabularPredictor(**ag_predictor_args).fit(train_data, **ag_fit_args)

    # Evaluate on test data if provided
    if args.test_dir:
        test_file = get_input_path(args.test_dir)
        test_data = TabularDataset(test_file)

        y_pred_proba = predictor.predict_proba(test_data)
        output_fmt = config.get("output_prediction_format", "csv")
        if output_fmt == "parquet":
            y_pred_proba.to_parquet(f"{args.output_data_dir}/predictions.parquet")
        else:
            y_pred_proba.to_csv(f"{args.output_data_dir}/predictions.csv")

        if config.get("leaderboard", False):
            lb = predictor.leaderboard(test_data, silent=False)
            lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")

        if config.get("feature_importance", False):
            fi = predictor.feature_importance(test_data)
            fi.to_csv(f"{args.output_data_dir}/feature_importance.csv")

        # Write evaluation metrics for pipeline consumption
        perf = predictor.evaluate(test_data)
        eval_metric = config["ag_predictor_args"].get("eval_metric", "accuracy")
        # evaluate() returns a dict in AG 1.x; extract the scalar value
        metric_value = perf[eval_metric] if isinstance(perf, dict) else perf
        metrics = {"metrics": {eval_metric: abs(metric_value)}}
        with open(os.path.join(args.output_data_dir, "evaluation.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics))
    else:
        if config.get("leaderboard", False):
            lb = predictor.leaderboard(silent=False)
            lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")

    # Copy serving script into model artifact if provided
    if args.serving_script:
        serving_code_dir = os.path.join(save_path, "code")
        os.makedirs(serving_code_dir, exist_ok=True)
        serving_script_path = get_input_path(args.serving_script)
        shutil.move(serving_script_path, os.path.join(serving_code_dir, os.path.basename(serving_script_path)))
        print(f"Serving script saved to {serving_code_dir}")

    print(f"Model saved to {save_path}")
    print(f"Model dir contents: {os.listdir(save_path)}")
