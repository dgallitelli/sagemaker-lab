"""
AutoGluon TimeSeries training script for SageMaker.

Runs inside the AutoGluon DLC container. Reads long-format CSV data
with columns [item_id, timestamp, target] from the 'train' channel.

Based on: autogluon/autogluon-cloud sagemaker_scripts/train.py (timeseries branch)
         + aws-samples/modern-time-series-forecasting-on-aws Lab 5
"""
import argparse
import json
import os
from pprint import pprint

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def get_input_path(path: str) -> str:
    """Return the first file found in the given directory."""
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    if not files:
        raise FileNotFoundError(f"No files found in {path}")
    if len(files) > 1:
        print(f"WARN: multiple files in {path}, using first: {files[0]}")
    return os.path.join(path, files[0])


def get_env(name: str):
    return os.environ.get(name)


def prepare_timeseries_dataframe(df: pd.DataFrame, id_column: str, timestamp_column: str) -> TimeSeriesDataFrame:
    """Convert a flat DataFrame to AutoGluon TimeSeriesDataFrame."""
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    tsdf = TimeSeriesDataFrame.from_data_frame(
        df, id_column=id_column, timestamp_column=timestamp_column,
    )
    return tsdf


if __name__ == "__main__":
    print("Starting AutoGluon TimeSeries Training")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-dir", type=str, default=get_env("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=get_env("SM_MODEL_DIR"))
    parser.add_argument("--n_gpus", type=int, default=int(get_env("SM_NUM_GPUS") or 0))
    parser.add_argument("--train_dir", type=str, default=get_env("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test_dir", type=str, required=False, default=get_env("SM_CHANNEL_TEST"))
    # TimeSeries-specific args
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--id-column", type=str, default="item_id")
    parser.add_argument("--timestamp-column", type=str, default="timestamp")
    parser.add_argument("--prediction-length", type=int, default=84)
    parser.add_argument("--eval-metric", type=str, default="MASE")
    parser.add_argument("--presets", type=str, default="medium_quality")
    parser.add_argument("--time-limit", type=int, default=3600)
    args, _ = parser.parse_known_args()
    print(f"Args: {args}")

    os.makedirs(args.output_data_dir, exist_ok=True)

    # Load training data
    train_file = get_input_path(args.train_dir)
    print(f"Loading training data from {train_file}")
    train_df = pd.read_csv(train_file)
    train_data = prepare_timeseries_dataframe(train_df, args.id_column, args.timestamp_column)
    print(f"Training data: {len(train_data)} rows, {train_data.num_items} items")

    # Train
    save_path = os.path.normpath(args.model_dir)
    predictor = TimeSeriesPredictor(
        target=args.target,
        prediction_length=args.prediction_length,
        eval_metric=args.eval_metric,
        path=save_path,
    )
    predictor.fit(
        train_data=train_data,
        presets=args.presets,
        time_limit=args.time_limit,
    )

    # Leaderboard
    lb = predictor.leaderboard(silent=False)
    lb.to_csv(os.path.join(args.output_data_dir, "leaderboard.csv"), index=False)

    # Evaluate on test data if provided
    if args.test_dir:
        test_file = get_input_path(args.test_dir)
        print(f"Loading test data from {test_file}")
        test_df = pd.read_csv(test_file)
        test_data = prepare_timeseries_dataframe(test_df, args.id_column, args.timestamp_column)

        # Evaluate
        scores = predictor.evaluate(test_data)
        print(f"Test scores: {scores}")

        # Predictions
        predictions = predictor.predict(train_data)
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(os.path.join(args.output_data_dir, "predictions.csv"))

        # Write evaluation metrics for pipeline consumption
        # evaluate() returns a dict in AG 1.x; extract the scalar value
        score_value = scores[args.eval_metric] if isinstance(scores, dict) else scores
        metrics = {"metrics": {args.eval_metric: abs(score_value)}}
        with open(os.path.join(args.output_data_dir, "evaluation.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics))

    print(f"Model saved to {save_path}")
