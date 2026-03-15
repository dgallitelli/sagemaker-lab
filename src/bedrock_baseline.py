"""
Bedrock embedding baseline evaluation.

Evaluates Amazon Titan Text v2, Amazon Nova Multimodal Embeddings, and Cohere Embed v4
on the same test sets used for SPLADE, providing a direct comparison.

Usage:
    python src/bedrock_baseline.py --data-dirs data/nfcorpus data/fiqa data/esci
    python src/bedrock_baseline.py --data-dirs data/nfcorpus data/fiqa  # skip ESCI for speed
"""

import argparse
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MODELS = {
    "titan-v2": {
        "model_id": "amazon.titan-embed-text-v2:0",
        "dimensions": 1024,
        "batch_size": 1,  # single text per call
    },
    "nova-multimodal": {
        "model_id": "amazon.nova-2-multimodal-embeddings-v1:0",
        "dimensions": 1024,
        "batch_size": 1,
    },
    "cohere-v4": {
        "model_id": "cohere.embed-v4:0",
        "dimensions": 1024,
        "batch_size": 96,  # supports batch
    },
}

LABEL_SCORES = {"E": 1.0, "S": 0.5, "C": 0.1, "I": 0.0}


# ---------------------------------------------------------------------------
# Metrics (identical to evaluate.py / dense_baseline.py)
# ---------------------------------------------------------------------------

def ndcg_at_k(ranked_pids, qrels, k=10):
    gains = [qrels.get(pid, 0.0) for pid in ranked_pids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_pids, qrels, k=100):
    retrieved = set(ranked_pids[:k])
    relevant = {pid for pid, score in qrels.items() if score > 0}
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked_pids, qrels, k=10):
    for i, pid in enumerate(ranked_pids[:k]):
        if qrels.get(pid, 0.0) > 0:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_corpus(path: Path):
    corpus = []
    with open(path) as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus


def load_test_queries(path: Path):
    queries_by_id = {}
    qrels = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            qid = row["query_id"]
            pid = row["product_id"]
            if qid not in queries_by_id:
                queries_by_id[qid] = row["query"]
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


def build_doc_text(doc):
    parts = [doc.get("title", ""), doc.get("description", ""), doc.get("bullet_points", "")]
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Bedrock encoding
# ---------------------------------------------------------------------------

def _invoke_titan(client, model_id, texts, dimensions):
    """Encode a single text with Titan."""
    body = json.dumps({
        "inputText": texts[0],
        "dimensions": dimensions,
        "normalize": True,
    })
    resp = client.invoke_model(modelId=model_id, body=body)
    result = json.loads(resp["body"].read())
    return [result["embedding"]]


def _invoke_nova(client, model_id, texts, dimensions):
    """Encode a single text with Nova Multimodal Embeddings."""
    body = json.dumps({
        "inputText": texts[0],
        "embeddingConfig": {"outputEmbeddingLength": dimensions},
    })
    resp = client.invoke_model(modelId=model_id, body=body)
    result = json.loads(resp["body"].read())
    return [result["embedding"]]


def _invoke_cohere(client, model_id, texts, dimensions, input_type="search_document"):
    """Encode a batch of texts with Cohere Embed v4."""
    body = json.dumps({
        "texts": texts,
        "input_type": input_type,
        "embedding_types": ["float"],
        "output_dimension": dimensions,
    })
    resp = client.invoke_model(modelId=model_id, body=body)
    result = json.loads(resp["body"].read())
    return result["embeddings"]["float"]


def encode_bedrock(
    client,
    model_name: str,
    texts: List[str],
    dimensions: int,
    input_type: str = "search_document",
    max_workers: int = 20,
) -> np.ndarray:
    """Encode texts using a Bedrock embedding model with concurrency."""
    cfg = MODELS[model_name]
    model_id = cfg["model_id"]
    batch_size = cfg["batch_size"]
    all_embeddings = [None] * len(texts)

    # Build batches
    batches = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batches.append((i, batch_texts))

    completed = 0
    retries = {}
    log_interval = max(1, len(batches) // 20)

    def _call(batch_idx, batch_texts):
        for attempt in range(5):
            try:
                if model_name == "titan-v2":
                    return batch_idx, _invoke_titan(client, model_id, batch_texts, dimensions)
                elif model_name == "nova-multimodal":
                    return batch_idx, _invoke_nova(client, model_id, batch_texts, dimensions)
                elif model_name == "cohere-v4":
                    return batch_idx, _invoke_cohere(client, model_id, batch_texts, dimensions, input_type)
            except Exception as e:
                err = str(e)
                if "ThrottlingException" in err or "Too many requests" in err.lower():
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError(f"Max retries for batch {batch_idx}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_call, idx, bt): idx for idx, bt in batches}
        for future in as_completed(futures):
            batch_idx, embs = future.result()
            for j, emb in enumerate(embs):
                all_embeddings[batch_idx + j] = emb
            completed += 1
            if completed % log_interval == 0:
                logger.info(f"  {model_name}: {completed}/{len(batches)} batches ({completed * batch_size}/{len(texts)} texts)")

    # Normalize (Titan already normalized; Cohere/Nova may not be)
    embeddings = np.array(all_embeddings, dtype=np.float32)
    if model_name != "titan-v2":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = embeddings / norms
    return embeddings


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(client, model_name, corpus, queries, qrels, product_ids, bm25_results, dataset_name):
    """Evaluate a single Bedrock model on a dataset."""
    cfg = MODELS[model_name]
    dimensions = cfg["dimensions"]

    # Cohere uses different workers (batched = fewer calls)
    max_workers = 10 if model_name == "cohere-v4" else 30

    # Encode corpus
    t0 = time.time()
    corpus_texts = [build_doc_text(doc) for doc in corpus]
    logger.info(f"Encoding {len(corpus_texts)} corpus docs with {model_name}...")
    corpus_emb = encode_bedrock(client, model_name, corpus_texts, dimensions, "search_document", max_workers)
    corpus_time = time.time() - t0
    logger.info(f"  Corpus encoded in {corpus_time:.1f}s — shape: {corpus_emb.shape}")

    # Encode queries
    t0 = time.time()
    query_texts = [q["query"] for q in queries]
    logger.info(f"Encoding {len(query_texts)} queries with {model_name}...")
    query_emb = encode_bedrock(client, model_name, query_texts, dimensions, "search_query", max_workers)
    query_time = time.time() - t0
    logger.info(f"  Queries encoded in {query_time:.1f}s — shape: {query_emb.shape}")

    # Rank
    ndcg_scores, recall_scores, mrr_scores = [], [], []
    for batch_start in range(0, len(queries), 500):
        batch_end = min(batch_start + 500, len(queries))
        batch_queries = queries[batch_start:batch_end]
        sim_matrix = query_emb[batch_start:batch_end] @ corpus_emb.T
        for j, query in enumerate(batch_queries):
            qid = query["query_id"]
            if qid not in qrels or not qrels[qid]:
                continue
            scores = sim_matrix[j]
            top_k = min(100, len(scores))
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            ranked_pids = [product_ids[idx] for idx in top_indices]
            ndcg_scores.append(ndcg_at_k(ranked_pids, qrels[qid], k=10))
            recall_scores.append(recall_at_k(ranked_pids, qrels[qid], k=100))
            mrr_scores.append(mrr_at_k(ranked_pids, qrels[qid], k=10))

    results = {
        "ndcg@10": float(np.mean(ndcg_scores)),
        "recall@100": float(np.mean(recall_scores)),
        "mrr@10": float(np.mean(mrr_scores)),
        "corpus_time": corpus_time,
        "query_time": query_time,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--output", default="bedrock_baseline_results.json")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    import boto3
    client = boto3.client("bedrock-runtime", region_name=args.region)

    all_results = {}

    # Process smallest datasets first
    data_dirs = sorted(args.data_dirs, key=lambda d: Path(d).stat().st_size if Path(d).exists() else 0)

    for data_dir_str in data_dirs:
        data_dir = Path(data_dir_str)
        dataset_name = data_dir.name
        corpus_path = data_dir / "corpus.jsonl"
        test_path = data_dir / "test.jsonl"
        bm25_path = data_dir / "bm25_baseline_results.json"

        if not corpus_path.exists() or not test_path.exists():
            logger.warning(f"Skipping {dataset_name}: missing files")
            continue

        corpus = load_corpus(corpus_path)
        queries, qrels = load_test_queries(test_path)
        product_ids = [doc["product_id"] for doc in corpus]
        bm25_results = None
        if bm25_path.exists():
            with open(bm25_path) as f:
                bm25_results = json.load(f)

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name} | {len(corpus)} corpus, {len(queries)} queries")
        logger.info(f"{'='*60}")

        all_results[dataset_name] = {}

        for model_name in args.models:
            try:
                results = evaluate_model(
                    client, model_name, corpus, queries, qrels,
                    product_ids, bm25_results, dataset_name,
                )
                all_results[dataset_name][model_name] = results

                # Print results
                print(f"\n  {dataset_name} / {model_name}:")
                for m in ["ndcg@10", "recall@100", "mrr@10"]:
                    bm25_val = bm25_results.get(m, 0) if bm25_results else 0
                    vs = f"{(results[m] - bm25_val) / bm25_val * 100:+.1f}%" if bm25_val else "N/A"
                    print(f"    {m}: {results[m]:.4f} (vs BM25: {vs})")

            except Exception as e:
                logger.error(f"Failed {model_name} on {dataset_name}: {e}")
                all_results[dataset_name][model_name] = {"error": str(e)}

        # Save after each dataset (partial results on failure)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Final summary
    print(f"\n{'='*75}")
    print(f"  BEDROCK BASELINE SUMMARY")
    print(f"{'='*75}")
    for ds, models in all_results.items():
        print(f"\n  {ds}:")
        for mn, res in models.items():
            if "error" in res:
                print(f"    {mn}: ERROR — {res['error'][:80]}")
            else:
                print(f"    {mn}: NDCG@10={res['ndcg@10']:.4f}  Recall@100={res['recall@100']:.4f}  MRR@10={res['mrr@10']:.4f}")
    print(f"\n{'='*75}")


if __name__ == "__main__":
    main()
