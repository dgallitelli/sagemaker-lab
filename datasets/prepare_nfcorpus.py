"""
Prepare mteb/nfcorpus (Biomedical) dataset and BM25 baseline.

Graded relevance: raw scores (0, 1, 2) are preserved as raw_score in test pairs
so that NDCG uses proper gradations instead of collapsing to ESCI label buckets.

Usage:
  python datasets/prepare_nfcorpus.py --output-dir data/nfcorpus
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from common import logger, prepare_and_save


def load_nfcorpus() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load mteb/nfcorpus (3633 corpus docs, graded relevance).
    Score mapping: 2 -> E, 1 -> S, 0 -> I. Raw scores preserved for NDCG.
    Returns (train_rows, test_rows, extra_corpus_rows).
    """
    from datasets import load_dataset

    logger.info("Loading mteb/nfcorpus...")
    qrels_ds = load_dataset("mteb/nfcorpus")
    queries_ds = load_dataset("mteb/nfcorpus", "queries")["queries"]
    corpus_ds = load_dataset("mteb/nfcorpus", "corpus")["corpus"]

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
            label = "E" if score >= 2.0 else ("S" if score >= 1.0 else "I")
            query_text = query_map.get(qid, "")
            product = corpus_map.get(pid, {})
            if not query_text or not product:
                continue
            rows.append({
                "query_id": qid,
                "query": query_text,
                "product_id": pid,
                "esci_label": label,
                "raw_score": score,
                "query_locale": "us",
                "title": product.get("title", ""),
                "description": product.get("description", ""),
                "bullet_points": "",
            })
        return rows

    train_rows = qrels_to_rows("train")
    test_rows = qrels_to_rows("test")

    # Full corpus (all 3633 docs) for retrieval
    corpus_rows = [
        {"product_id": str(r["_id"]), "title": r.get("title", ""),
         "description": r.get("text", ""), "bullet_points": ""}
        for r in corpus_ds
    ]

    logger.info(
        f"Loaded mteb/nfcorpus: {len(train_rows)} train pairs, {len(test_rows)} test pairs, "
        f"{len(corpus_map)} corpus documents"
    )
    return train_rows, test_rows, corpus_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NFCorpus dataset")
    parser.add_argument("--output-dir", default="data/nfcorpus", help="Output directory")
    args = parser.parse_args()

    train_rows, test_rows, extra_corpus = load_nfcorpus()
    prepare_and_save(train_rows, test_rows, Path(args.output_dir), extra_corpus_rows=extra_corpus)


if __name__ == "__main__":
    main()
