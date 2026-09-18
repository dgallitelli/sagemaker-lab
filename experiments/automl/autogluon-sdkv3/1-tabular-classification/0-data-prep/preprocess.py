"""
Preprocessing script for SageMaker Processing.

Reads raw UCI Adult Census data, cleans it, and splits into train/test.
Runs inside a ScriptProcessor container.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "class",
]

LABEL = "class"

if __name__ == "__main__":
    input_dir = "/opt/ml/processing/input"
    train_output_dir = "/opt/ml/processing/train"
    test_output_dir = "/opt/ml/processing/test"

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Load all CSV files from input
    dfs = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".csv") or fname.endswith(".data"):
            fpath = os.path.join(input_dir, fname)
            try:
                df = pd.read_csv(fpath, header=None, names=COLUMNS, skipinitialspace=True)
                dfs.append(df)
            except (pd.errors.ParserError, ValueError):
                df = pd.read_csv(fpath, skipinitialspace=True)
                dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(data)} rows with columns: {data.columns.tolist()}")

    # Clean label column (test file has trailing periods)
    data[LABEL] = data[LABEL].astype(str).str.rstrip(".")

    # Drop rows with missing values
    data = data.replace("?", pd.NA).dropna()
    print(f"After cleaning: {len(data)} rows")

    # Split
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data[LABEL])

    train_df.to_csv(os.path.join(train_output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(test_output_dir, "test.csv"), index=False)

    print(f"Train: {len(train_df)} rows")
    print(f"Test:  {len(test_df)} rows")
