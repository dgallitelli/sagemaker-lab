"""
Prepare mteb/fiqa (Financial Question Answering) dataset and BM25 baseline.

Usage:
  python datasets/prepare_fiqa.py --output-dir data/fiqa
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from common import logger, prepare_and_save


def load_fiqa() -> Tuple[List[Dict], List[Dict]]:
    """
    Load mteb/fiqa. Binary relevance (score >= 1.0 -> E, else I).
    Returns (train_rows, test_rows) in normalized schema.
    """
    from datasets import load_dataset

    logger.info("Loading mteb/fiqa...")
    qrels_ds = load_dataset("mteb/fiqa")
    queries_ds = load_dataset("mteb/fiqa", "queries")["queries"]
    corpus_ds = load_dataset("mteb/fiqa", "corpus")["corpus"]

    query_map = {row["_id"]: row["text"] for row in queries_ds}
    corpus_map = {
        row["_id"]: {"title": row.get("title", ""), "description": row.get("text", "")}
        for row in corpus_ds
    }

    def qrels_to_rows(split: str) -> List[Dict]:
        rows = []
        for row in qrels_ds[split]:
            qid = str(row["query-id"])
            pid = str(row["corpus-id"])
            score = float(row["score"])
            label = "E" if score >= 1.0 else "I"
            query_text = query_map.get(qid, "")
            product = corpus_map.get(pid, {})
            if not query_text or not product:
                continue
            rows.append({
                "query_id": qid,
                "query": query_text,
                "product_id": pid,
                "esci_label": label,
                "query_locale": "us",
                "title": product.get("title", ""),
                "description": product.get("description", ""),
                "bullet_points": "",
            })
        return rows

    train_rows = qrels_to_rows("train")
    test_rows = qrels_to_rows("test")

    logger.info(
        f"Loaded mteb/fiqa: {len(train_rows)} train pairs, {len(test_rows)} test pairs, "
        f"{len(corpus_map)} corpus documents"
    )

    def load_full_corpus() -> List[Dict]:
        return [
            {"product_id": str(row["_id"]), "title": row.get("title", ""),
             "description": row.get("text", ""), "bullet_points": ""}
            for row in corpus_ds
        ]

    return train_rows, test_rows, load_full_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FiQA dataset")
    parser.add_argument("--output-dir", default="data/fiqa", help="Output directory")
    args = parser.parse_args()

    train_rows, test_rows, corpus_loader = load_fiqa()
    prepare_and_save(train_rows, test_rows, Path(args.output_dir), full_corpus_loader=corpus_loader)


if __name__ == "__main__":
    main()
