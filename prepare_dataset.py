"""
Prepare Amazon ESCI dataset and compute BM25 baseline benchmark.

Outputs (in --output-dir):
  train.jsonl               — training pairs with ESCI labels
  test.jsonl                — test pairs with ESCI labels
  corpus.jsonl              — unique product documents
  bm25_baseline_results.json — BM25 NDCG@10 / Recall@100 / MRR@10
  dataset_stats.json        — dataset statistics

BM25 baseline is the ground truth that SPLADE training must beat.

Usage:
  python prepare_dataset.py --output-dir data/
  python prepare_dataset.py --output-dir data/ --synthetic  # force synthetic fallback
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

LABEL_SCORES = {"E": 1.0, "S": 0.5, "C": 0.1, "I": 0.0}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_esci_from_huggingface() -> Tuple[List[Dict], List[Dict]]:
    """
    Load Amazon ESCI dataset from Hugging Face Hub.
    Tries multiple known dataset IDs in order.
    Returns (train_rows, test_rows) as lists of dicts.
    """
    from datasets import load_dataset

    candidates = [
        "tasksource/amazon-esci",
        "amazon-esci-data",
        "amazon-research/esci-data",
    ]

    for dataset_id in candidates:
        try:
            logger.info(f"Trying HuggingFace dataset: {dataset_id}")
            ds = load_dataset(dataset_id)

            train_rows = [row for row in ds["train"] if row.get("query_locale") == "us"]
            # Use 'test' split, or fall back to 'validation'
            test_split = "test" if "test" in ds else "validation"
            test_rows = [row for row in ds[test_split] if row.get("query_locale") == "us"]

            logger.info(
                f"Loaded {dataset_id}: {len(train_rows)} train, {len(test_rows)} test (US locale)"
            )
            return train_rows, test_rows

        except Exception as e:
            logger.warning(f"Failed to load {dataset_id}: {e}")
            continue

    raise RuntimeError("All HuggingFace ESCI candidates failed")


def load_fiqa_from_huggingface() -> Tuple[List[Dict], List[Dict]]:
    """
    Load mteb/fiqa as a fallback retrieval dataset when ESCI is unavailable.
    FiQA (Financial Question Answering) is a standard BEIR benchmark.
    Returns (train_rows, test_rows) in the same normalized schema as ESCI rows.
    """
    from datasets import load_dataset

    logger.info("Loading mteb/fiqa as fallback retrieval benchmark...")

    qrels_ds = load_dataset("mteb/fiqa")
    queries_ds = load_dataset("mteb/fiqa", "queries")["queries"]
    corpus_ds = load_dataset("mteb/fiqa", "corpus")["corpus"]

    # Build lookup maps
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
            # FiQA is binary relevance: score=1.0 maps to "E", everything else "I"
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
    return train_rows, test_rows


def normalize_row(row: Dict) -> Dict:
    """Normalize a dataset row to our canonical schema."""
    return {
        "query_id": str(row.get("query_id", "")),
        "query": str(row.get("query", "")),
        "product_id": str(row.get("product_id", "")),
        "esci_label": str(row.get("esci_label", "I")),
        "query_locale": str(row.get("query_locale", "us")),
        "title": str(row.get("product_title", row.get("title", ""))),
        "description": str(row.get("product_description", row.get("description", ""))),
        "bullet_points": str(row.get("product_bullet_point", row.get("bullet_points", ""))),
    }


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_train: int = 8000,
    n_test: int = 2000,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate synthetic e-commerce product search pairs for testing when ESCI
    download fails. Creates realistic query/product pairs with ESCI-style labels.
    """
    logger.warning("ESCI download failed — generating synthetic dataset")
    random.seed(42)

    product_templates = [
        ("laptop {brand} {cpu} {ram}GB RAM {storage}GB SSD",
         "High-performance laptop with {cpu} processor and {ram}GB RAM",
         "Fast processor. Large storage. Lightweight design."),
        ("{color} running shoes size {size}",
         "Athletic running shoes for men and women",
         "Breathable mesh upper. Cushioned sole. Available in multiple colors."),
        ('USB-C cable {length}ft {speed}Gbps',
         "High-speed USB-C charging and data cable",
         "Fast charging support. Braided nylon. Universal compatibility."),
        ("{brand} coffee maker {cups} cup",
         "Drip coffee maker with programmable timer",
         "Programmable 24-hour delay brew. Keep warm plate. Easy-clean filter basket."),
        ("wireless headphones {brand} noise cancelling",
         "Over-ear Bluetooth headphones with active noise cancellation",
         "40-hour battery life. Premium sound. Foldable design."),
    ]

    brands = ["Samsung", "Apple", "Sony", "Bose", "Dell", "HP", "LG", "Logitech"]
    colors = ["black", "white", "blue", "red", "green", "silver"]
    all_products = []
    product_id_counter = 1

    for template_title, template_desc, bullet in product_templates:
        for brand in brands:
            for color in colors:
                pid = f"PROD{product_id_counter:06d}"
                title = template_title.format(
                    brand=brand, cpu="Intel i7", ram=16, storage=512,
                    size=10, length=6, speed=10, cups=12, color=color,
                )
                desc = template_desc.format(cpu="Intel i7", ram=16)
                all_products.append({
                    "product_id": pid,
                    "title": title,
                    "description": desc,
                    "bullet_points": bullet,
                })
                product_id_counter += 1

    query_templates = [
        ("laptop 16GB RAM", ["laptop {brand} {cpu} {ram}GB RAM {storage}GB SSD"]),
        ("black wireless headphones", ["wireless headphones {brand} noise cancelling"]),
        ("USB cable fast charging", ["USB-C cable {length}ft {speed}Gbps"]),
        ("coffee maker programmable", ["{brand} coffee maker {cups} cup"]),
        ("running shoes breathable", ["{color} running shoes size {size}"]),
    ]

    def make_pairs(n: int) -> List[Dict]:
        pairs = []
        qid_counter = 1
        labels = list(LABEL_SCORES.keys())
        label_weights = [0.3, 0.3, 0.2, 0.2]  # E, S, C, I distribution
        while len(pairs) < n:
            qt, _ = random.choice(query_templates)
            query = qt.replace("{brand}", random.choice(brands)).replace("{cpu}", "i7")
            qid = f"Q{qid_counter:06d}"
            qid_counter += 1
            for _ in range(random.randint(3, 8)):
                product = random.choice(all_products)
                label = random.choices(labels, weights=label_weights)[0]
                pairs.append({
                    "query_id": qid,
                    "query": query,
                    "product_id": product["product_id"],
                    "esci_label": label,
                    "query_locale": "us",
                    "title": product["title"],
                    "description": product["description"],
                    "bullet_points": product["bullet_points"],
                })
        return pairs[:n]

    train_rows = make_pairs(n_train)
    test_rows = make_pairs(n_test)

    logger.info(
        f"Synthetic dataset generated: {len(train_rows)} train, {len(test_rows)} test, "
        f"{len(all_products)} products"
    )
    return train_rows, test_rows


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
            "positive_id": row["product_id"],  # same field, aliased for train.py
            "esci_label": row["esci_label"],
        }
        for row in rows
    ]


def build_test_pairs(rows: List[Dict]) -> List[Dict]:
    return [
        {
            "query_id": row["query_id"],
            "query": row["query"],
            "product_id": row["product_id"],
            "esci_label": row["esci_label"],
        }
        for row in rows
    ]


def build_qrels(test_rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in test_rows:
        score = LABEL_SCORES.get(row["esci_label"], 0.0)
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
    """
    Build BM25 index over full corpus and evaluate on test queries.
    Returns NDCG@10, Recall@100, MRR@10.
    """
    from rank_bm25 import BM25Okapi

    logger.info(f"Building BM25 index over {len(corpus)} products...")
    t0 = time.time()
    product_ids = [p["product_id"] for p in corpus]
    corpus_texts = [build_product_text(p) for p in corpus]
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info(f"BM25 index built in {time.time() - t0:.1f}s")

    # Unique test queries
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

    import numpy as np
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_nfcorpus_from_huggingface() -> Tuple[List[Dict], List[Dict]]:
    """
    Load mteb/nfcorpus (biomedical retrieval, 3633 corpus docs, graded relevance).
    Score mapping: 2.0 → E (highly relevant), 1.0 → S (relevant), else → I.
    Returns (train_rows, test_rows) in the normalized ESCI schema.
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
                "query_locale": "us",
                "title": product.get("title", ""),
                "description": product.get("description", ""),
                "bullet_points": "",
            })
        return rows

    train_rows = qrels_to_rows("train")
    test_rows = qrels_to_rows("test")

    # Include full corpus (all 3633 docs) so retrieval is against complete pool
    corpus_rows = [
        {"query_id": "", "query": "", "product_id": str(r["_id"]),
         "esci_label": "I", "query_locale": "us",
         "title": r.get("title", ""), "description": r.get("text", ""), "bullet_points": ""}
        for r in corpus_ds
    ]

    logger.info(
        f"Loaded mteb/nfcorpus: {len(train_rows)} train pairs, {len(test_rows)} test pairs, "
        f"{len(corpus_map)} corpus documents"
    )
    return train_rows, test_rows, corpus_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare retrieval dataset and BM25 baseline")
    parser.add_argument("--output-dir", default="data", help="Output directory for data files")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset")
    parser.add_argument(
        "--dataset",
        choices=["esci", "fiqa", "nfcorpus", "synthetic"],
        default=None,
        help="Force a specific dataset (default: try esci → fiqa → synthetic)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset_choice = args.dataset or ("synthetic" if args.synthetic else "auto")
    extra_corpus_rows: List[Dict] = []

    if dataset_choice == "synthetic":
        train_rows_raw, test_rows_raw = generate_synthetic_dataset()
    elif dataset_choice == "fiqa":
        train_rows_raw, test_rows_raw = load_fiqa_from_huggingface()
    elif dataset_choice == "nfcorpus":
        train_rows_raw, test_rows_raw, extra_corpus_rows = load_nfcorpus_from_huggingface()
    elif dataset_choice == "esci":
        train_rows_raw, test_rows_raw = load_esci_from_huggingface()
        train_rows_raw = [normalize_row(r) for r in train_rows_raw]
        test_rows_raw = [normalize_row(r) for r in test_rows_raw]
    else:  # auto
        try:
            train_rows_raw, test_rows_raw = load_esci_from_huggingface()
            train_rows_raw = [normalize_row(r) for r in train_rows_raw]
            test_rows_raw = [normalize_row(r) for r in test_rows_raw]
            dataset_choice = "esci"
        except Exception as e:
            logger.warning(f"ESCI failed: {e}. Falling back to mteb/fiqa")
            try:
                train_rows_raw, test_rows_raw = load_fiqa_from_huggingface()
                dataset_choice = "fiqa"
            except Exception as e2:
                logger.warning(f"FiQA failed: {e2}. Falling back to synthetic")
                train_rows_raw, test_rows_raw = generate_synthetic_dataset()
                dataset_choice = "synthetic"

    logger.info(f"Dataset: {dataset_choice}")
    all_rows = train_rows_raw + test_rows_raw

    # ── Build splits ─────────────────────────────────────────────────────────
    train_pairs = build_train_pairs(train_rows_raw)
    test_pairs = build_test_pairs(test_rows_raw)
    corpus = build_corpus(all_rows)

    # Extend corpus with full document pool when available
    if extra_corpus_rows:
        seen = {p["product_id"] for p in corpus}
        for row in extra_corpus_rows:
            if row["product_id"] and row["product_id"] not in seen:
                corpus.append({
                    "product_id": row["product_id"],
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "bullet_points": "",
                })
                seen.add(row["product_id"])
        logger.info(f"Extended corpus to {len(corpus)} documents")
    elif dataset_choice == "fiqa":
        try:
            from datasets import load_dataset as _ld
            corpus_ds = _ld("mteb/fiqa", "corpus")["corpus"]
            seen = {p["product_id"] for p in corpus}
            for row in corpus_ds:
                pid = str(row["_id"])
                if pid not in seen:
                    corpus.append({"product_id": pid, "title": row.get("title", ""),
                                   "description": row.get("text", ""), "bullet_points": ""})
                    seen.add(pid)
            logger.info(f"Extended corpus to {len(corpus)} documents (full fiqa corpus)")
        except Exception:
            pass
    qrels = build_qrels(test_rows_raw)

    # ── Dataset statistics ───────────────────────────────────────────────────
    label_dist = {}
    for row in train_rows_raw:
        label = row.get("esci_label", "?")
        label_dist[label] = label_dist.get(label, 0) + 1

    n_train_queries = len({r["query_id"] for r in train_rows_raw})
    n_test_queries = len({r["query_id"] for r in test_rows_raw})

    dataset_stats = {
        "train_pairs": len(train_pairs),
        "train_queries": n_train_queries,
        "test_pairs": len(test_pairs),
        "test_queries": n_test_queries,
        "corpus_products": len(corpus),
        "label_E": label_dist.get("E", 0),
        "label_S": label_dist.get("S", 0),
        "label_C": label_dist.get("C", 0),
        "label_I": label_dist.get("I", 0),
    }

    # ── BM25 evaluation ──────────────────────────────────────────────────────
    logger.info("Running BM25 baseline evaluation...")
    bm25_results = evaluate_bm25(corpus, test_rows_raw, qrels)
    logger.info(f"BM25 baseline: {bm25_results}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    write_jsonl(output_dir / "train.jsonl", train_pairs)
    write_jsonl(output_dir / "test.jsonl", test_pairs)
    write_jsonl(output_dir / "corpus.jsonl", corpus)

    with open(output_dir / "bm25_baseline_results.json", "w") as f:
        json.dump(bm25_results, f, indent=2)
    logger.info(f"BM25 baseline saved to {output_dir / 'bm25_baseline_results.json'}")

    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)
    logger.info(f"Dataset stats saved to {output_dir / 'dataset_stats.json'}")

    print_summary_table(bm25_results, dataset_stats)


if __name__ == "__main__":
    main()
