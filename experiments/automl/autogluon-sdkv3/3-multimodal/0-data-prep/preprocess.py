"""
Preprocessing script for SageMaker Processing.

Loads synthetic churn JSONL data, selects relevant features,
and outputs train/validation/test JSONL files.
Runs inside a ScriptProcessor container.
"""
import json
import os

import pandas as pd


NUMERICAL_FEATURES = ["CustServ Calls", "Account Length"]
CATEGORICAL_FEATURES = ["plan", "limit"]
TEXT_FEATURES = ["text"]
LABEL = "y"
ALL_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES + [LABEL]


def load_jsonl(filepath):
    with open(filepath) as f:
        return pd.DataFrame([json.loads(line) for line in f])


if __name__ == "__main__":
    input_dir = "/opt/ml/processing/input"
    train_output = "/opt/ml/processing/train"
    val_output = "/opt/ml/processing/validation"
    test_output = "/opt/ml/processing/test"

    for d in [train_output, val_output, test_output]:
        os.makedirs(d, exist_ok=True)

    input_files = os.listdir(input_dir)
    print(f"Input files: {input_files}")

    train_df = val_df = test_df = None
    for f in input_files:
        fpath = os.path.join(input_dir, f)
        if "train" in f:
            train_df = load_jsonl(fpath)
        elif "validation" in f or "val" in f:
            val_df = load_jsonl(fpath)
        elif "test" in f:
            test_df = load_jsonl(fpath)

    if train_df is None:
        raise FileNotFoundError("No train file found")

    # Select relevant columns
    train_df = train_df[ALL_COLUMNS]
    if val_df is not None:
        val_df = val_df[ALL_COLUMNS]
    if test_df is not None:
        test_df = test_df[ALL_COLUMNS]

    # If no validation set, split from train
    if val_df is None:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df[LABEL])

    # Save as JSONL (preserves text with commas)
    train_df.to_json(os.path.join(train_output, "train.jsonl"), orient="records", lines=True)
    val_df.to_json(os.path.join(val_output, "validation.jsonl"), orient="records", lines=True)
    if test_df is not None:
        test_df.to_json(os.path.join(test_output, "test.jsonl"), orient="records", lines=True)

    print(f"Train:      {len(train_df)} rows")
    print(f"Validation: {len(val_df)} rows")
    if test_df is not None:
        print(f"Test:       {len(test_df)} rows")
