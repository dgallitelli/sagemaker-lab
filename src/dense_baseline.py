"""
Dense embedding baseline evaluation using BGE-large-en-v1.5.

Evaluates NDCG@10, Recall@100, MRR@10 on the same test sets used for SPLADE,
providing a direct comparison between sparse (SPLADE) and dense retrieval.

Runs as a SageMaker Training job with multiple dataset channels.
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-large-en-v1.5"
LABEL_SCORES = {"E": 1.0, "S": 0.5, "C": 0.1, "I": 0.0}


# ---------------------------------------------------------------------------
# Metric implementations (identical to evaluate.py)
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
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


def mrr_at_k(ranked_pids: List[str], qrels: Dict[str, float], k: int = 10) -> float:
    for i, pid in enumerate(ranked_pids[:k]):
        if qrels.get(pid, 0.0) > 0:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Data loading (same format as SPLADE pipeline)
# ---------------------------------------------------------------------------

def load_corpus(path: Path) -> List[Dict]:
    corpus = []
    with open(path) as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus


def load_test_queries(path: Path):
    """Load test.jsonl, return (queries, qrels)."""
    queries_by_id = {}
    qrels = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            qid = row["query_id"]
            pid = row["product_id"]

            if qid not in queries_by_id:
                queries_by_id[qid] = row["query"]

            # Convert label to relevance score
            label = row.get("esci_label", row.get("label", "E"))
            if isinstance(label, (int, float)):
                score = float(label)
            elif label in LABEL_SCORES:
                score = LABEL_SCORES[label]
            else:
                score = float(label) if label.replace(".", "").isdigit() else 1.0

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][pid] = score

    queries = [{"query_id": qid, "query": text} for qid, text in queries_by_id.items()]
    return queries, qrels


def build_doc_text(doc: Dict) -> str:
    parts = [doc.get("title", ""), doc.get("description", ""), doc.get("bullet_points", "")]
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Dense encoding + evaluation
# ---------------------------------------------------------------------------

def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode texts to dense vectors, normalized for cosine similarity."""
    # BGE-large uses "Represent this sentence: " prefix for queries in some setups,
    # but bge-large-en-v1.5 works well without it for retrieval. We use the
    # instruction-based approach for queries as recommended by the model card.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


def evaluate_dataset(model: SentenceTransformer, data_dir: Path, dataset_name: str) -> Dict:
    """Evaluate BGE on a single dataset directory."""
    corpus_path = data_dir / "corpus.jsonl"
    test_path = data_dir / "test.jsonl"
    bm25_path = data_dir / "bm25_baseline_results.json"

    if not corpus_path.exists() or not test_path.exists():
        logger.warning(f"Skipping {dataset_name}: missing corpus.jsonl or test.jsonl in {data_dir}")
        return {}

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {dataset_name}: {data_dir}")
    logger.info(f"{'='*60}")

    # Load data
    corpus = load_corpus(corpus_path)
    queries, qrels = load_test_queries(test_path)
    logger.info(f"Loaded {len(corpus)} corpus docs, {len(queries)} test queries")

    product_ids = [doc["product_id"] for doc in corpus]

    # Load BM25 baseline
    bm25_results = None
    if bm25_path.exists():
        with open(bm25_path) as f:
            bm25_results = json.load(f)
        logger.info(f"Loaded BM25 baseline from {bm25_path}")

    # Encode corpus
    t0 = time.time()
    logger.info(f"Encoding {len(corpus)} corpus documents with {MODEL_NAME}...")
    corpus_texts = [build_doc_text(doc) for doc in corpus]
    corpus_embeddings = encode_texts(model, corpus_texts, batch_size=128)
    corpus_time = time.time() - t0
    logger.info(f"Corpus encoded in {corpus_time:.1f}s — shape: {corpus_embeddings.shape}")

    # Encode queries (BGE recommends "Represent this sentence: " prefix for retrieval queries)
    t0 = time.time()
    logger.info(f"Encoding {len(queries)} queries...")
    query_texts = [q["query"] for q in queries]
    query_embeddings = encode_texts(model, query_texts, batch_size=128)
    query_time = time.time() - t0
    logger.info(f"Queries encoded in {query_time:.1f}s — shape: {query_embeddings.shape}")

    # Compute similarities in batches to avoid OOM on large datasets
    logger.info("Computing cosine similarity rankings...")
    ndcg_scores, recall_scores, mrr_scores = [], [], []
    query_batch_size = 500  # process 500 queries at a time

    for batch_start in tqdm(range(0, len(queries), query_batch_size), desc="Ranking"):
        batch_end = min(batch_start + query_batch_size, len(queries))
        batch_queries = queries[batch_start:batch_end]
        batch_embeddings = query_embeddings[batch_start:batch_end]

        # (batch, dim) @ (dim, n_docs) -> (batch, n_docs)
        sim_matrix = batch_embeddings @ corpus_embeddings.T

        for j, query in enumerate(batch_queries):
            qid = query["query_id"]
            if qid not in qrels or not qrels[qid]:
                continue

            scores = sim_matrix[j]
            top_k = min(100, len(scores))
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            ranked_pids = [product_ids[idx] for idx in top_indices]

            query_qrels = qrels[qid]
            ndcg_scores.append(ndcg_at_k(ranked_pids, query_qrels, k=10))
            recall_scores.append(recall_at_k(ranked_pids, query_qrels, k=100))
            mrr_scores.append(mrr_at_k(ranked_pids, query_qrels, k=10))

    results = {
        "ndcg@10": float(np.mean(ndcg_scores)),
        "recall@100": float(np.mean(recall_scores)),
        "mrr@10": float(np.mean(mrr_scores)),
    }

    # Print results table
    print(f"\n{'='*75}")
    print(f"  {dataset_name.upper()} — Dense Baseline ({MODEL_NAME})")
    print(f"  {len(corpus)} corpus docs, {len(queries)} test queries")
    print(f"  Corpus encode: {corpus_time:.1f}s, Query encode: {query_time:.1f}s")
    print(f"{'='*75}")
    print(f"{'Metric':<15} {'BM25':>10} {'BGE-large':>10} {'vs BM25':>10}")
    print(f"{'-'*75}")

    for metric in ["ndcg@10", "recall@100", "mrr@10"]:
        dense_val = results[metric]
        if bm25_results:
            bm25_val = bm25_results.get(metric, 0.0)
            vs_bm25 = f"{(dense_val - bm25_val) / bm25_val * 100:+.1f}%" if bm25_val > 0 else "N/A"
            print(f"{metric:<15} {bm25_val:>10.4f} {dense_val:>10.4f} {vs_bm25:>10}")
        else:
            print(f"{metric:<15} {'N/A':>10} {dense_val:>10.4f} {'N/A':>10}")

    print(f"{'='*75}\n")

    # Log CloudWatch-compatible metrics
    for metric, value in results.items():
        print(json.dumps({"metric_name": f"dense_{dataset_name}_{metric}", "value": value}))

    return results


def main():
    # SageMaker puts input channels at /opt/ml/input/data/<channel>
    # We expect channels named: fiqa, nfcorpus, esci (any subset)
    sm_input = Path("/opt/ml/input/data")
    output_dir = Path("/opt/ml/model")

    if sm_input.exists():
        # Running on SageMaker
        channels = sorted([d.name for d in sm_input.iterdir() if d.is_dir()])
        logger.info(f"SageMaker mode — found channels: {channels}")
    else:
        # Local mode fallback
        logger.info("Local mode — looking for --data-dirs argument")
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dirs", nargs="+", required=True,
                            help="Paths to dataset directories (e.g., data/fiqa data/nfcorpus)")
        args = parser.parse_args()
        channels = []
        for d in args.data_dirs:
            p = Path(d)
            channels.append(p.name)
            # Symlink to fake SageMaker structure
            sm_input.mkdir(parents=True, exist_ok=True)
            target = sm_input / p.name
            if not target.exists():
                target.symlink_to(p.resolve())
        output_dir = Path("dense_baseline_output")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    logger.info(f"Loading {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    logger.info(f"Model loaded on {device}, embedding dim={model.get_sentence_embedding_dimension()}")

    all_results = {}
    for channel in channels:
        data_dir = sm_input / channel
        results = evaluate_dataset(model, data_dir, channel)
        if results:
            all_results[channel] = results

    # Save combined results
    results_path = output_dir / "dense_baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Print final summary
    print(f"\n{'='*75}")
    print(f"  SUMMARY — Dense Baseline ({MODEL_NAME})")
    print(f"{'='*75}")
    for dataset, results in all_results.items():
        print(f"\n  {dataset}:")
        for metric, value in results.items():
            print(f"    {metric}: {value:.4f}")
    print(f"\n{'='*75}")


if __name__ == "__main__":
    main()
