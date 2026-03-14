"""
Prepare Amazon ESCI (E-commerce) dataset and BM25 baseline.

4-level graded relevance: E (Exact), S (Substitute), C (Complement), I (Irrelevant).
Supports subsampling via --max-pairs for local experimentation.

Usage:
  python datasets/prepare_esci.py --output-dir data/esci
  python datasets/prepare_esci.py --output-dir data/esci --max-pairs 200000
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import LABEL_SCORES, logger, prepare_and_save

# tasksource/esci uses full words; map to single-letter ESCI labels
_LABEL_MAP = {"Exact": "E", "Substitute": "S", "Complement": "C", "Irrelevant": "I"}


def load_esci(max_pairs: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Load Amazon ESCI from Hugging Face Hub (US locale only).
    Optionally subsample training pairs (stratified by label).
    Returns (train_rows, test_rows) in normalized schema.
    """
    from datasets import load_dataset

    logger.info("Loading tasksource/esci from Hugging Face...")
    ds = load_dataset("tasksource/esci")

    train_rows_raw = [row for row in ds["train"] if row.get("product_locale") == "us"]
    test_split = "test" if "test" in ds else "validation"
    test_rows_raw = [row for row in ds[test_split] if row.get("product_locale") == "us"]

    logger.info(f"Loaded ESCI: {len(train_rows_raw)} train, {len(test_rows_raw)} test (US locale)")

    def normalize(row: Dict) -> Dict:
        return {
            "query_id": str(row.get("query_id", "")),
            "query": str(row.get("query", "")).strip(),
            "product_id": str(row.get("product_id", "")),
            "esci_label": _LABEL_MAP.get(row.get("esci_label", ""), "I"),
            "query_locale": "us",
            "title": str(row.get("product_title", "") or ""),
            "description": str(row.get("product_description", "") or ""),
            "bullet_points": str(row.get("product_bullet_point", "") or ""),
        }

    train_rows = [normalize(r) for r in train_rows_raw]
    test_rows = [normalize(r) for r in test_rows_raw]

    # Stratified subsampling for local experimentation
    if max_pairs and len(train_rows) > max_pairs:
        train_rows = _stratified_subsample(train_rows, max_pairs)

    return train_rows, test_rows


def _stratified_subsample(rows: List[Dict], n: int) -> List[Dict]:
    """Subsample n rows, preserving original label distribution."""
    random.seed(42)
    by_label: Dict[str, List[Dict]] = {}
    for row in rows:
        by_label.setdefault(row["esci_label"], []).append(row)

    total = len(rows)
    sampled = []
    for label, label_rows in by_label.items():
        label_n = max(1, round(n * len(label_rows) / total))
        random.shuffle(label_rows)
        sampled.extend(label_rows[:label_n])

    random.shuffle(sampled)
    logger.info(
        f"Stratified subsample: {len(rows)} -> {len(sampled)} pairs | "
        + " ".join(f"{l}={sum(1 for r in sampled if r['esci_label']==l)}" for l in LABEL_SCORES)
    )
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Amazon ESCI dataset")
    parser.add_argument("--output-dir", default="data/esci", help="Output directory")
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Max training pairs (stratified subsample). None = use all.",
    )
    args = parser.parse_args()

    train_rows, test_rows = load_esci(max_pairs=args.max_pairs)
    prepare_and_save(train_rows, test_rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
