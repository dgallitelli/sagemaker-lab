"""
Shared utilities for dataset preparation: BM25 baseline, metrics, I/O helpers.
"""

import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

LABEL_SCORES = {"E": 1.0, "S": 0.5, "C": 0.1, "I": 0.0}


# ---------------------------------------------------------------------------
# Data splitting and corpus building
# ---------------------------------------------------------------------------

def build_corpus(rows: List[Dict]) -> List[Dict]:
    """Deduplicate and build product corpus from dataset rows."""
    seen = {}
    for row in rows:
        pid = row["product_id"]
        if pid not in seen:
            seen[pid] = {
                "product_id": pid,
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "bullet_points": row.get("bullet_points", ""),
            }
    return list(seen.values())


def build_train_pairs(rows: List[Dict]) -> List[Dict]:
    """Build training pairs (keep all labels; train.py filters for positives)."""
    return [
        {
            "query_id": row["query_id"],
            "query": row["query"],
            "product_id": row["product_id"],
            "positive_id": row["product_id"],
            "esci_label": row["esci_label"],
        }
        for row in rows
    ]


def build_test_pairs(rows: List[Dict]) -> List[Dict]:
    pairs = []
    for row in rows:
        pair = {
            "query_id": row["query_id"],
            "query": row["query"],
            "product_id": row["product_id"],
            "esci_label": row["esci_label"],
        }
        if "raw_score" in row:
            pair["raw_score"] = row["raw_score"]
        pairs.append(pair)
    return pairs


def build_qrels(test_rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in test_rows:
        score = row.get("raw_score", LABEL_SCORES.get(row["esci_label"], 0.0))
        qrels[row["query_id"]][row["product_id"]] = score
    return dict(qrels)


# ---------------------------------------------------------------------------
# BM25 evaluation
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_product_text(product: Dict) -> str:
    parts = [
        product.get("title", ""),
        product.get("description", ""),
        product.get("bullet_points", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def evaluate_bm25(
    corpus: List[Dict],
    test_rows: List[Dict],
    qrels: Dict[str, Dict[str, float]],
) -> Dict:
    from rank_bm25 import BM25Okapi
    import numpy as np

    logger.info(f"Building BM25 index over {len(corpus)} products...")
    t0 = time.time()
    product_ids = [p["product_id"] for p in corpus]
    corpus_texts = [build_product_text(p) for p in corpus]
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info(f"BM25 index built in {time.time() - t0:.1f}s")

    query_map: Dict[str, str] = {}
    for row in test_rows:
        query_map[row["query_id"]] = row["query"]

    ndcg_scores, recall_scores, mrr_scores = [], [], []
    logger.info(f"Evaluating BM25 on {len(query_map)} test queries...")

    for qid, query_text in query_map.items():
        if qid not in qrels:
            continue
        tokens = tokenize(query_text)
        scores = bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked_pids = [product_ids[i] for i in ranked_indices]

        query_qrels = qrels[qid]
        ndcg_scores.append(ndcg_at_k(ranked_pids, query_qrels, k=10))
        recall_scores.append(recall_at_k(ranked_pids, query_qrels, k=100))
        mrr_scores.append(mrr_at_k(ranked_pids, query_qrels, k=10))

    results = {
        "ndcg@10":    float(np.mean(ndcg_scores)),
        "recall@100": float(np.mean(recall_scores)),
        "mrr@10":     float(np.mean(mrr_scores)),
        "n_queries":  len(ndcg_scores),
    }
    return results


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def ndcg_at_k(ranked_pids: List[str], qrels: Dict[str, float], k: int = 10) -> float:
    gains = [qrels.get(pid, 0.0) for pid in ranked_pids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_pids: List[str], qrels: Dict[str, float], k: int = 100) -> float:
    retrieved = set(ranked_pids[:k])
    relevant = {pid for pid, score in qrels.items() if score > 0}
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked_pids: List[str], qrels: Dict[str, float], k: int = 10) -> float:
    for i, pid in enumerate(ranked_pids[:k]):
        if qrels.get(pid, 0.0) > 0:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: List[Dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {path}")


def print_summary_table(bm25_results: Dict, dataset_stats: Dict) -> None:
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)
    for k, v in dataset_stats.items():
        print(f"  {k:<30} {v:>15,}" if isinstance(v, int) else f"  {k:<30} {v!r:>15}")

    print("\n" + "=" * 60)
    print("  BM25 BASELINE (ground truth to beat)")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Score':>10}")
    print("  " + "-" * 30)
    for metric in ["ndcg@10", "recall@100", "mrr@10"]:
        val = bm25_results.get(metric, 0.0)
        print(f"  {metric:<20} {val:>10.4f}")
    print(f"  {'Queries evaluated':<20} {bm25_results.get('n_queries', 0):>10,}")
    print("=" * 60 + "\n")


def prepare_and_save(
    train_rows_raw: List[Dict],
    test_rows_raw: List[Dict],
    output_dir: Path,
    extra_corpus_rows: Optional[List[Dict]] = None,
    full_corpus_loader=None,
) -> None:
    """
    Shared pipeline: build splits, extend corpus, compute BM25, save outputs.

    Args:
        train_rows_raw: Raw training rows (must have query_id, query, product_id, esci_label, title, description).
        test_rows_raw: Raw test rows (same schema; may include raw_score).
        output_dir: Where to write train.jsonl, test.jsonl, corpus.jsonl, etc.
        extra_corpus_rows: Additional corpus documents not in train/test (e.g., NFCorpus full corpus).
        full_corpus_loader: Optional callable that returns List[Dict] of extra corpus docs (e.g., for FiQA).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = train_rows_raw + test_rows_raw
    train_pairs = build_train_pairs(train_rows_raw)
    test_pairs = build_test_pairs(test_rows_raw)
    corpus = build_corpus(all_rows)

    # Extend corpus with full document pool
    if extra_corpus_rows:
        seen = {p["product_id"] for p in corpus}
        for row in extra_corpus_rows:
            if row["product_id"] and row["product_id"] not in seen:
                corpus.append({
                    "product_id": row["product_id"],
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "bullet_points": row.get("bullet_points", ""),
                })
                seen.add(row["product_id"])
        logger.info(f"Extended corpus to {len(corpus)} documents")
    elif full_corpus_loader:
        try:
            extra = full_corpus_loader()
            seen = {p["product_id"] for p in corpus}
            for doc in extra:
                if doc["product_id"] not in seen:
                    corpus.append(doc)
                    seen.add(doc["product_id"])
            logger.info(f"Extended corpus to {len(corpus)} documents (full corpus)")
        except Exception as e:
            logger.warning(f"Failed to extend corpus: {e}")

    qrels = build_qrels(test_rows_raw)

    # Dataset statistics
    label_dist = {}
    for row in train_rows_raw:
        label = row.get("esci_label", "?")
        label_dist[label] = label_dist.get(label, 0) + 1

    dataset_stats = {
        "train_pairs": len(train_pairs),
        "train_queries": len({r["query_id"] for r in train_rows_raw}),
        "test_pairs": len(test_pairs),
        "test_queries": len({r["query_id"] for r in test_rows_raw}),
        "corpus_products": len(corpus),
        "label_E": label_dist.get("E", 0),
        "label_S": label_dist.get("S", 0),
        "label_C": label_dist.get("C", 0),
        "label_I": label_dist.get("I", 0),
    }

    # BM25 evaluation
    logger.info("Running BM25 baseline evaluation...")
    bm25_results = evaluate_bm25(corpus, test_rows_raw, qrels)
    logger.info(f"BM25 baseline: {bm25_results}")

    # Save outputs
    write_jsonl(output_dir / "train.jsonl", train_pairs)
    write_jsonl(output_dir / "test.jsonl", test_pairs)
    write_jsonl(output_dir / "corpus.jsonl", corpus)

    with open(output_dir / "bm25_baseline_results.json", "w") as f:
        json.dump(bm25_results, f, indent=2)
    logger.info(f"BM25 baseline saved to {output_dir / 'bm25_baseline_results.json'}")

    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)

    print_summary_table(bm25_results, dataset_stats)
