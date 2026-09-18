"""XGBoost Training Script for Amazon SageMaker AI

This script trains an XGBoost regression model on tabular data.
Used with SKLearn 1.4-2 container + custom requirements.txt for XGBoost.

Usage:
    python train.py --n-estimators 100 --max-depth 6 --target-column production
"""

import argparse
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    parser.add_argument('--min-child-weight', type=int, default=3)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--target-column', type=str, default='production')
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    return parser.parse_args()


def load_data(train_path, target_column):
    """Load training data from CSV or Parquet files."""
    train_dir = Path(train_path)

    if train_dir.is_file():
        df = pd.read_csv(train_dir) if str(train_dir).endswith('.csv') else pd.read_parquet(train_dir)
    else:
        csv_files = list(train_dir.glob('*.csv'))
        parquet_files = list(train_dir.glob('*.parquet'))
        if csv_files:
            df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        elif parquet_files:
            df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        else:
            raise ValueError(f"No CSV or Parquet files found in {train_path}")

    print(f"Loaded data: {df.shape}")
    y = df[target_column]
    X = df.drop(columns=[target_column]).select_dtypes(include=[np.number]).fillna(0)
    return X, y


def train_model(X, y, args):
    """Train XGBoost regressor with train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return model, {
        'rmse': float(rmse),
        'r2': float(r2),
        'feature_names': list(X.columns)
    }


def save_model(model, metrics, model_dir, output_data_dir):
    """Save model artifacts and metrics."""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)

    # Save model
    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))

    # Save feature names for inference
    with open(os.path.join(model_dir, 'feature_names.json'), 'w') as f:
        json.dump({'feature_names': metrics['feature_names']}, f)

    # Save metrics
    with open(os.path.join(output_data_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Model saved.")


def model_fn(model_dir):
    """Load model for SageMaker inference (required by SKLearn container)."""
    return joblib.load(os.path.join(model_dir, 'model.joblib'))


if __name__ == '__main__':
    print("=" * 50)
    print("XGBOOST TRAINING JOB")
    print("=" * 50)
    args = parse_args()
    X, y = load_data(args.train, args.target_column)
    model, metrics = train_model(X, y, args)
    save_model(model, metrics, args.model_dir, args.output_data_dir)
    print("TRAINING COMPLETE!")
