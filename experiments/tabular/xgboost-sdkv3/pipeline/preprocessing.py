"""Preprocessing Script for SageMaker Pipeline

Loads CSV/Parquet data, keeps numeric columns, fills NaN with 0,
and creates 80/20 train/test split.

Input: /opt/ml/processing/input/
Output: /opt/ml/processing/train/, /opt/ml/processing/test/
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data(input_dir: str) -> pd.DataFrame:
    """Load all CSV/Parquet files from input directory."""
    input_path = Path(input_dir)

    csv_files = list(input_path.glob("*.csv"))
    parquet_files = list(input_path.glob("*.parquet"))

    if csv_files:
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        print(f"Loaded {len(csv_files)} CSV files")
    elif parquet_files:
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        print(f"Loaded {len(parquet_files)} Parquet files")
    else:
        raise ValueError(f"No CSV or Parquet files found in {input_dir}")

    return df


def preprocess(df: pd.DataFrame, target_column: str = "production") -> pd.DataFrame:
    """Preprocess data: keep numeric columns, fill NaN with 0."""
    print(f"Raw data shape: {df.shape}")

    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Columns: {list(df.columns)}")

    # Keep target + numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_column not in numeric_cols:
        numeric_cols.append(target_column)

    df_processed = df[numeric_cols].fillna(0)
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Columns: {list(df_processed.columns)}")

    return df_processed


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data into train/test sets."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


def main():
    print("=" * 50)
    print("PREPROCESSING")
    print("=" * 50)

    # Paths (SageMaker Processing convention)
    input_dir = "/opt/ml/processing/input"
    train_dir = "/opt/ml/processing/train"
    test_dir = "/opt/ml/processing/test"

    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Load and preprocess
    df = load_data(input_dir)
    df_processed = preprocess(df, target_column="production")

    # Split
    train_df, test_df = split_data(df_processed, test_size=0.2)

    # Save
    train_df.to_csv(os.path.join(train_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(test_dir, "test.csv"), index=False)

    print(f"Saved train.csv to {train_dir}")
    print(f"Saved test.csv to {test_dir}")
    print("PREPROCESSING COMPLETE!")


if __name__ == "__main__":
    main()
