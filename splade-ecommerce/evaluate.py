"""
In-memory SPLADE evaluator.

Computes NDCG@10, Recall@100, MRR@10 by encoding corpus and test queries
with the current model, then ranking via dot-product similarity.
Loads BM25 baseline from bm25_baseline_results.json for comparison.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sentence_transformers import SparseEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ESCI graded relevance
LABEL_SCORES = {"E": 1.0, "S": 0.5, "C": 0.1, "I": 0.0}


class SpladeEvaluator:
    """
    In-memory evaluator for SPLADE sparse embedding models.

    Encodes corpus once, then scores all test queries via sparse dot-product.
    No index required — purely in-memory matrix operations.
    """

    def __init__(self, bm25_baseline_path: Optional[str] = None):
        self.bm25_results: Optional[Dict] = None
        if bm25_baseline_path and Path(bm25_baseline_path).exists():
            with open(bm25_baseline_path) as f:
                self.bm25_results = json.load(f)
            logger.info(f"Loaded BM25 baseline from {bm25_baseline_path}")

    def evaluate(
        self,
        model: SparseEncoder,
        corpus: List[Dict],
        test_queries: List[Dict],
        qrels: Dict[str, Dict[str, float]],
        eval_batch_size: int = 64,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate SPLADE model on test queries.

        Args:
            model: SparseEncoder to evaluate.
            corpus: List of product dicts (product_id, title, description, bullet_points).
            test_queries: List of query dicts (query_id, query).
            qrels: Nested dict: {query_id: {product_id: relevance_score}}.
            eval_batch_size: Batch size for encoding.
            show_progress: Show tqdm progress bars.

        Returns:
            Dict with ndcg@10, recall@100, mrr@10 keys.
        """
        product_ids = [p["product_id"] for p in corpus]
        product_id_to_idx = {pid: i for i, pid in enumerate(product_ids)}

        # Encode corpus into sparse matrix [n_docs, vocab_size]
        logger.info(f"Encoding {len(corpus)} corpus documents...")
        corpus_texts = [_build_product_text(p) for p in corpus]
        corpus_matrix = _encode_to_sparse_matrix(
            model, corpus_texts, eval_batch_size, show_progress
        )
        logger.info(
            f"Corpus encoded: shape={corpus_matrix.shape}, "
            f"nnz={corpus_matrix.nnz}, "
            f"sparsity={1 - corpus_matrix.nnz / (corpus_matrix.shape[0] * corpus_matrix.shape[1]):.4f}"
        )

        # Encode queries
        logger.info(f"Encoding {len(test_queries)} test queries...")
        query_texts = [q["query"] for q in test_queries]
        query_matrix = _encode_to_sparse_matrix(
            model, query_texts, eval_batch_size, show_progress
        )

        # Compute metrics per query
        ndcg_scores, recall_scores, mrr_scores = [], [], []

        q_iterator = enumerate(test_queries)
        if show_progress:
            q_iterator = tqdm(
                q_iterator, total=len(test_queries), desc="Evaluating queries"
            )

        for i, query in q_iterator:
            qid = query["query_id"]
            if qid not in qrels or not qrels[qid]:
                continue

            # Dot product: query (1, vocab) @ corpus.T (vocab, n_docs) → (n_docs,)
            scores = corpus_matrix.dot(query_matrix[i].T).toarray().flatten()

            # Rank documents by score descending
            ranked_indices = np.argsort(scores)[::-1]
            ranked_pids = [product_ids[idx] for idx in ranked_indices]

            query_qrels = qrels[qid]
            ndcg_scores.append(ndcg_at_k(ranked_pids, query_qrels, k=10))
            recall_scores.append(recall_at_k(ranked_pids, query_qrels, k=100))
            mrr_scores.append(mrr_at_k(ranked_pids, query_qrels, k=10))

        if not ndcg_scores:
            logger.warning("No queries with qrels found — returning zeros")
            return {"ndcg@10": 0.0, "recall@100": 0.0, "mrr@10": 0.0}

        results = {
            "ndcg@10": float(np.mean(ndcg_scores)),
            "recall@100": float(np.mean(recall_scores)),
            "mrr@10": float(np.mean(mrr_scores)),
        }

        self._print_results(results)
        return results

    def _print_results(self, splade_results: Dict[str, float]) -> None:
        """Print comparison table: BM25 vs SPLADE."""
        print("\n" + "=" * 60)
        print(f"{'Metric':<15} {'BM25':>12} {'SPLADE':>12} {'Delta':>12}")
        print("-" * 60)

        for metric in ["ndcg@10", "recall@100", "mrr@10"]:
            splade_val = splade_results.get(metric, 0.0)
            if self.bm25_results:
                bm25_val = self.bm25_results.get(metric, 0.0)
                delta = splade_val - bm25_val
                delta_pct = (delta / bm25_val * 100) if bm25_val > 0 else 0.0
                delta_str = f"{delta_pct:+.1f}%"
                print(
                    f"{metric:<15} {bm25_val:>12.4f} {splade_val:>12.4f} {delta_str:>12}"
                )
            else:
                print(f"{metric:<15} {'N/A':>12} {splade_val:>12.4f} {'N/A':>12}")

        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


def ndcg_at_k(
    ranked_pids: List[str], qrels: Dict[str, float], k: int = 10
) -> float:
    """Graded NDCG@K. Ideal DCG uses all qrels, not just top-K."""
    gains = [qrels.get(pid, 0.0) for pid in ranked_pids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    # IDCG: best possible ranking of all known relevant docs
    ideal_gains = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    ranked_pids: List[str], qrels: Dict[str, float], k: int = 100
) -> float:
    """Recall@K: fraction of relevant docs (score > 0) found in top-K."""
    retrieved = set(ranked_pids[:k])
    relevant = {pid for pid, score in qrels.items() if score > 0}
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


def mrr_at_k(
    ranked_pids: List[str], qrels: Dict[str, float], k: int = 10
) -> float:
    """MRR@K: reciprocal rank of the first relevant (score > 0) result."""
    for i, pid in enumerate(ranked_pids[:k]):
        if qrels.get(pid, 0.0) > 0:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Sparse encoding utilities
# ---------------------------------------------------------------------------


def _build_product_text(product: Dict) -> str:
    parts = [
        product.get("title", ""),
        product.get("description", ""),
        product.get("bullet_points", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _encode_to_sparse_matrix(
    model: SparseEncoder,
    texts: List[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> csr_matrix:
    """
    Encode texts and return a scipy CSR sparse matrix [n_texts, vocab_size].

    Using CSR for memory efficiency — SPLADE vocab_size ~30K but most values are 0.
    """
    all_rows, all_cols, all_data = [], [], []
    vocab_size: Optional[int] = None

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding batches", unit="batch")

    row_offset = 0
    for i in iterator:
        batch = texts[i : i + batch_size]
        result = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_sparse_tensor=False,
        )
        embeddings = result.cpu().numpy() if hasattr(result, "cpu") else result

        if vocab_size is None:
            vocab_size = embeddings.shape[1]

        for j, emb in enumerate(embeddings):
            nonzero_idx = np.where(emb > 0)[0]
            all_rows.extend([row_offset + j] * len(nonzero_idx))
            all_cols.extend(nonzero_idx.tolist())
            all_data.extend(emb[nonzero_idx].tolist())

        row_offset += len(batch)

    if vocab_size is None:
        raise RuntimeError("No texts were encoded")

    return csr_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(len(texts), vocab_size),
        dtype=np.float32,
    )
