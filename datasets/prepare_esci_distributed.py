"""
Distributed ESCI dataset preparation across multiple SageMaker Processing instances.

Each instance downloads the full dataset, normalizes its shard of training rows,
and writes a train shard file. The leader (rank 0) also builds the corpus, test set,
and BM25 baseline.

Output files:
  - train_shard_{rank}.jsonl  (one per instance)
  - test.jsonl                (leader only)
  - corpus.jsonl              (leader only)
  - bm25_baseline_results.json (leader only)
  - dataset_stats.json        (leader only)

Usage (SageMaker Processing — invoked via sagemaker_processing.py --distributed):
  python prepare_esci_distributed.py --output-dir /opt/ml/processing/output
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import (
    LABEL_SCORES,
    build_corpus,
    build_qrels,
    build_test_pairs,
    build_train_pairs,
    evaluate_bm25,
    logger,
    print_summary_table,
    write_jsonl,
)

_LABEL_MAP = {"Exact": "E", "Substitute": "S", "Complement": "C", "Irrelevant": "I"}

RESOURCE_CONFIG_PATH = "/opt/ml/config/resourceconfig.json"


def get_rank_and_world_size() -> Tuple[int, int]:
    """Read SageMaker resource config to determine this instance's rank."""
    try:
        with open(RESOURCE_CONFIG_PATH) as f:
            config = json.load(f)
        hosts = sorted(config["hosts"])
        current = config["current_host"]
        rank = hosts.index(current)
        return rank, len(hosts)
    except (FileNotFoundError, KeyError):
        logger.info("No resource config found — running as single instance (rank 0/1)")
        return 0, 1


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


def load_and_shard_esci(
    rank: int,
    world_size: int,
    max_pairs: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load ESCI, normalize, and return this instance's shard of training data.
    All instances return the full test set (only leader uses it).
    """
    from datasets import load_dataset

    logger.info(f"[Rank {rank}/{world_size}] Loading tasksource/esci...")
    t0 = time.time()
    ds = load_dataset("tasksource/esci")
    logger.info(f"[Rank {rank}/{world_size}] Dataset loaded in {time.time() - t0:.1f}s")

    # Filter US locale
    train_raw = [r for r in ds["train"] if r.get("product_locale") == "us"]
    test_split = "test" if "test" in ds else "validation"
    test_raw = [r for r in ds[test_split] if r.get("product_locale") == "us"]
    logger.info(f"[Rank {rank}/{world_size}] US locale: {len(train_raw)} train, {len(test_raw)} test")

    # Normalize all rows (each instance normalizes its own shard of train + all test)
    # Shard training data deterministically by index
    my_train_raw = train_raw[rank::world_size]
    logger.info(f"[Rank {rank}/{world_size}] My train shard: {len(my_train_raw)} rows")

    t0 = time.time()
    my_train_rows = [normalize(r) for r in my_train_raw]
    logger.info(f"[Rank {rank}/{world_size}] Normalized {len(my_train_rows)} train rows in {time.time() - t0:.1f}s")

    # Only leader needs test rows and full train for corpus/stats
    if rank == 0:
        all_train_rows = [normalize(r) for r in train_raw]
        test_rows = [normalize(r) for r in test_raw]
    else:
        all_train_rows = None
        test_rows = None

    # Subsample this shard proportionally
    if max_pairs and len(my_train_rows) > max_pairs // world_size:
        shard_max = max(1, max_pairs // world_size)
        my_train_rows = _stratified_subsample(my_train_rows, shard_max, rank)

    return my_train_rows, test_rows, all_train_rows


def _stratified_subsample(rows: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """Subsample n rows, preserving original label distribution."""
    random.seed(seed)
    by_label = {}  # type: Dict[str, List[Dict]]
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
    parser = argparse.ArgumentParser(description="Distributed ESCI dataset preparation")
    parser.add_argument("--output-dir", default="data/esci", help="Output directory")
    parser.add_argument("--max-pairs", type=int, default=None, help="Max training pairs (total across all shards)")
    args = parser.parse_args()

    rank, world_size = get_rank_and_world_size()
    logger.info(f"[Rank {rank}/{world_size}] Starting distributed ESCI preparation")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    my_train_rows, test_rows, all_train_rows = load_and_shard_esci(
        rank, world_size, max_pairs=args.max_pairs,
    )

    # Every instance writes its train shard
    train_pairs = build_train_pairs(my_train_rows)
    write_jsonl(output_dir / f"train_shard_{rank}.jsonl", train_pairs)
    logger.info(f"[Rank {rank}/{world_size}] Wrote {len(train_pairs)} train pairs")

    # Leader handles corpus, test, BM25
    if rank == 0:
        assert all_train_rows is not None and test_rows is not None
        test_pairs = build_test_pairs(test_rows)
        corpus = build_corpus(all_train_rows + test_rows)
        qrels = build_qrels(test_rows)

        # Dataset statistics (from full training set, not just shard)
        label_dist = {}
        for row in all_train_rows:
            label = row.get("esci_label", "?")
            label_dist[label] = label_dist.get(label, 0) + 1

        dataset_stats = {
            "train_pairs": len(build_train_pairs(all_train_rows)),
            "train_queries": len({r["query_id"] for r in all_train_rows}),
            "test_pairs": len(test_pairs),
            "test_queries": len({r["query_id"] for r in test_rows}),
            "corpus_products": len(corpus),
            "label_E": label_dist.get("E", 0),
            "label_S": label_dist.get("S", 0),
            "label_C": label_dist.get("C", 0),
            "label_I": label_dist.get("I", 0),
            "world_size": world_size,
        }

        # BM25 baseline
        logger.info(f"[Rank 0] Running BM25 baseline on {len(corpus)} products, {len(qrels)} queries...")
        bm25_results = evaluate_bm25(corpus, test_rows, qrels)
        logger.info(f"[Rank 0] BM25 baseline: {bm25_results}")

        # Write outputs
        write_jsonl(output_dir / "test.jsonl", test_pairs)
        write_jsonl(output_dir / "corpus.jsonl", corpus)

        with open(output_dir / "bm25_baseline_results.json", "w") as f:
            json.dump(bm25_results, f, indent=2)
        with open(output_dir / "dataset_stats.json", "w") as f:
            json.dump(dataset_stats, f, indent=2)

        print_summary_table(bm25_results, dataset_stats)
    else:
        logger.info(f"[Rank {rank}/{world_size}] Non-leader — done after writing train shard")

    logger.info(f"[Rank {rank}/{world_size}] Finished")


if __name__ == "__main__":
    main()
