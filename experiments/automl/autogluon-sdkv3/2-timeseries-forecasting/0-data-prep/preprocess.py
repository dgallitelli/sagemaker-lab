"""
Preprocessing script for SageMaker Processing.

Converts UCI Electricity data from wide to long format,
resamples to 2-hour intervals, and splits into train/test.
Runs inside a ScriptProcessor container.
"""
import os
import zipfile

import pandas as pd


def main():
    input_dir = "/opt/ml/processing/input"
    train_output_dir = "/opt/ml/processing/train"
    test_output_dir = "/opt/ml/processing/test"

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    input_files = os.listdir(input_dir)
    print(f"Input files: {input_files}")

    # Handle zip file
    for f in input_files:
        if f.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(input_dir, f), "r") as z:
                for member in z.namelist():
                    member_path = os.path.realpath(os.path.join(input_dir, member))
                    if not member_path.startswith(os.path.realpath(input_dir)):
                        raise ValueError(f"Zip member {member} would extract outside target directory")
                z.extractall(input_dir)
            input_files = os.listdir(input_dir)
            break

    # Find the data file
    data_file = None
    for f in input_files:
        if f.endswith(".txt") or f.endswith(".csv"):
            data_file = os.path.join(input_dir, f)
            break

    if data_file is None:
        raise FileNotFoundError(f"No data file found in {input_dir}: {input_files}")

    print(f"Loading {data_file}...")
    if data_file.endswith(".txt"):
        df = pd.read_csv(data_file, sep=";", decimal=",", index_col=0, parse_dates=True)
    else:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    df.index.name = "timestamp"

    # Resample to 2-hour intervals
    df = df.resample("2h").sum()

    # Select top 50 customers by total consumption
    n_customers = 50
    top_customers = df.sum().nlargest(n_customers).index.tolist()
    df = df[top_customers]

    # Wide to long
    print(f"Converting to long format ({n_customers} customers)...")
    df_long = df.reset_index().melt(id_vars="timestamp", var_name="item_id", value_name="target")
    df_long = df_long.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    # Train/test split: hold out last 7 days
    prediction_length = 84
    cutoff = df_long.groupby("item_id")["timestamp"].transform(
        lambda x: x.max() - pd.Timedelta(days=7)
    )
    train_df = df_long[df_long["timestamp"] <= cutoff]
    test_df = df_long  # Full data for evaluation

    train_df.to_csv(os.path.join(train_output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(test_output_dir, "test.csv"), index=False)

    print(f"Train: {len(train_df)} rows")
    print(f"Test:  {len(test_df)} rows")
    print(f"Customers: {n_customers}, Prediction length: {prediction_length}")


if __name__ == "__main__":
    main()
