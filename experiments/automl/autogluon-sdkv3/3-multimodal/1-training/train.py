"""
AutoGluon Multimodal (fusion) training script for SageMaker.

Uses MultiModalPredictor to fuse text + numerical + categorical features
in a single neural network. Runs inside the AutoGluon DLC container.

Based on: churn_prediction_multimodality_of_text_and_tabular/
          containers/autogluon_multimodal_fusion/train.py
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor


def get_env(name: str):
    return os.environ.get(name)


def find_filepath(path: str, ext: str = ".jsonl") -> str:
    """Find the first file with given extension in directory."""
    files = list(Path(path).glob(f"**/*{ext}"))
    if not files:
        # Try CSV fallback
        files = list(Path(path).glob("**/*.csv"))
    if not files:
        raise FileNotFoundError(f"No {ext} or .csv files found in {path}")
    return str(files[0])


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from JSONL or CSV."""
    if filepath.endswith(".jsonl"):
        with open(filepath) as f:
            data = [json.loads(line) for line in f]
        return pd.DataFrame(data)
    else:
        return pd.read_csv(filepath)


if __name__ == "__main__":
    print("Starting AutoGluon Multimodal Training")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=get_env("SM_MODEL_DIR"))
    parser.add_argument("--output-data-dir", type=str, default=get_env("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--train-dir", type=str, default=get_env("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation-dir", type=str, default=get_env("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--n_gpus", type=int, default=int(get_env("SM_NUM_GPUS") or 0))
    # Feature configuration
    parser.add_argument("--numerical-feature-names", type=str, default="CustServ Calls,Account Length")
    parser.add_argument("--categorical-feature-names", type=str, default="plan,limit")
    parser.add_argument("--textual-feature-names", type=str, default="text")
    parser.add_argument("--label-name", type=str, default="y")
    # AutoGluon configuration
    parser.add_argument("--problem_type", type=str, default="classification")
    parser.add_argument("--eval_metric", type=str, default="roc_auc")
    parser.add_argument("--presets", type=str, default="medium_quality")
    parser.add_argument("--pretrained-transformer", type=str, default="google/electra-small-discriminator")
    parser.add_argument("--verbosity", type=int, default=2)
    args, _ = parser.parse_known_args()
    print(f"Args: {args}")

    os.makedirs(args.output_data_dir, exist_ok=True)

    # Load data
    print("Loading training data")
    train_file = find_filepath(args.train_dir)
    train_data = load_data(train_file)
    print(f"Training data: {len(train_data)} rows")

    print("Loading validation data")
    val_file = find_filepath(args.validation_dir)
    validation_data = load_data(val_file)
    print(f"Validation data: {len(validation_data)} rows")

    # Parse feature names
    numerical_features = args.numerical_feature_names.split(",")
    categorical_features = args.categorical_feature_names.split(",")
    textual_features = args.textual_feature_names.split(",")
    label = args.label_name

    # Select columns
    all_features = numerical_features + categorical_features + textual_features + [label]
    train_data = train_data[all_features]
    validation_data = validation_data[all_features]

    # Determine problem type
    if args.problem_type == "classification":
        num_classes = len(np.unique(train_data[label].values))
        problem_type = "binary" if num_classes == 2 else "multiclass"
    else:
        problem_type = "regression"
    print(f"Problem type: {problem_type}")

    # Train
    predictor_args = {
        "label": label,
        "path": args.model_dir,
        "eval_metric": args.eval_metric,
        "problem_type": problem_type,
        "verbosity": args.verbosity,
    }

    hyperparameters = {
        "model.hf_text.checkpoint_name": args.pretrained_transformer,
    }

    predictor = MultiModalPredictor(**predictor_args)
    predictor.fit(
        train_data=train_data,
        tuning_data=validation_data,
        hyperparameters=hyperparameters,
    )

    # Evaluate
    perf = predictor.evaluate(validation_data)
    print(f"Validation performance: {perf}")

    # Write evaluation metrics
    # evaluate() returns a dict in AG 1.x; extract the scalar value
    metric_value = perf[args.eval_metric] if isinstance(perf, dict) else perf
    metrics = {"metrics": {args.eval_metric: abs(metric_value)}}
    with open(os.path.join(args.output_data_dir, "evaluation.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics))

    print(f"Model saved to {args.model_dir}")
