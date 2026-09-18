import os
import subprocess
import sys

import pandas as pd


def test_preprocess_splits_train_test(tmp_path):
    input_dir = tmp_path / "input"
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    input_dir.mkdir()

    # Minimal Adult-Census-shaped fixture: 20 rows, balanced classes
    rows = []
    for i in range(20):
        label = " <=50K" if i % 2 == 0 else " >50K"
        rows.append(
            f"{25+i}, State-gov, 77000, Bachelors, 13, Never-married, "
            f"Adm-clerical, Not-in-family, White, Male, 0, 0, 40, United-States,{label}"
        )
    (input_dir / "sample.data").write_text("\n".join(rows) + "\n")

    script = os.path.join(os.path.dirname(__file__), "..", "pipelines", "automl", "preprocess.py")
    env = {
        **os.environ,
        "PROCESSING_INPUT_DIR": str(input_dir),
        "PROCESSING_TRAIN_DIR": str(train_dir),
        "PROCESSING_TEST_DIR": str(test_dir),
    }
    subprocess.run([sys.executable, script], check=True, env=env)

    train_df = pd.read_csv(train_dir / "train.csv")
    test_df = pd.read_csv(test_dir / "test.csv")
    assert len(train_df) + len(test_df) == 20
    assert "class" in train_df.columns
    assert set(train_df["class"].unique()) <= {"<=50K", ">50K"}
