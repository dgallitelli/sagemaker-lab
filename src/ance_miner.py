"""
ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation) hard negative miner.

Uses scipy sparse matrix dot product for fast batch retrieval.
Encodes corpus and queries with the current model, computes Q @ C.T,
then extracts top-k hard negatives per query (excluding true positives).
"""

import logging
from typing import Dict, List, Optional, Set

import numpy as np
from sentence_transformers import SparseEncoder

from evaluate import _encode_to_sparse_matrix

logger = logging.getLogger(__name__)


class ANCEMiner:
    """
    In-memory hard negative miner using ANCE methodology.

    Encodes corpus with the current model state into a scipy sparse matrix,
    then retrieves nearest neighbors via batch dot-product similarity.
    """

    def __init__(self):
        self._corpus_matrix = None
        self._product_ids: List[str] = []

    def build_index(
        self,
        model: SparseEncoder,
        corpus: List[Dict],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> None:
        """
        Encode the full corpus into a scipy sparse matrix.

        Args:
            model: Current SparseEncoder checkpoint (gets updated each ANCE iteration).
            corpus: List of dicts with keys: product_id, title, description, bullet_points.
            batch_size: Encoding batch size.
            show_progress: Show tqdm progress bar.
        """
        self._product_ids = [p["product_id"] for p in corpus]

        corpus_texts = [_build_product_text(p) for p in corpus]
        logger.info(f"Encoding {len(corpus_texts)} corpus documents for ANCE index...")
        self._corpus_matrix = _encode_to_sparse_matrix(
            model, corpus_texts, batch_size=batch_size, show_progress=show_progress
        )
        logger.info(
            f"ANCE index built: {len(corpus)} documents indexed | "
            f"matrix shape={self._corpus_matrix.shape}, nnz={self._corpus_matrix.nnz}"
        )

    def mine(
        self,
        model: SparseEncoder,
        queries: List[Dict],
        k: int = 50,
        n_hard: int = 5,
        batch_size: int = 64,
        show_progress: bool = True,
        query_batch_size: int = 2000,
    ) -> List[Dict]:
        """
        Retrieve hard negatives for each query via batched sparse matrix dot product.

        Args:
            model: Current SparseEncoder checkpoint.
            queries: List of dicts with keys: query_id, query, positive_ids (list of str).
            k: How many candidates to retrieve per query.
            n_hard: How many hard negatives to keep per query.
            batch_size: Encoding batch size for query encoding.
            show_progress: Show tqdm progress bar.
            query_batch_size: Number of queries to score against corpus at once
                (controls peak memory: query_batch_size × n_docs score matrix).

        Returns:
            List of dicts with keys: query_id, query, positive_ids, negative_ids.
        """
        if self._corpus_matrix is None:
            raise RuntimeError("Call build_index() before mine()")

        query_texts = [q["query"] for q in queries]
        logger.info(f"Encoding {len(query_texts)} queries for hard negative mining...")
        query_matrix = _encode_to_sparse_matrix(
            model, query_texts, batch_size=batch_size, show_progress=show_progress
        )

        results = []
        total_found = 0
        all_neg_scores: List[float] = []
        n_skipped = 0

        # Process in batches to bound memory (query_batch_size × n_docs score matrix)
        for batch_start in range(0, len(queries), query_batch_size):
            batch_end = min(batch_start + query_batch_size, len(queries))
            q_batch = queries[batch_start:batch_end]
            qm_batch = query_matrix[batch_start:batch_end]

            # (batch_size, n_docs) sparse score matrix
            score_matrix = qm_batch @ self._corpus_matrix.T

            for i, query in enumerate(q_batch):
                positive_ids: Set[str] = set(query.get("positive_ids", []))

                # Skip degenerate queries (zero sparse activations)
                if qm_batch[i].nnz == 0:
                    results.append({
                        "query_id": query["query_id"],
                        "query": query["query"],
                        "positive_ids": list(positive_ids),
                        "negative_ids": [],
                    })
                    n_skipped += 1
                    continue

                # Get scores for this query as dense array
                scores = score_matrix[i].toarray().flatten()

                # Top-k candidates (retrieve extra to account for positive filtering)
                retrieve_k = min(k + len(positive_ids), len(scores))
                if retrieve_k >= len(scores):
                    top_indices = np.argsort(-scores)[:retrieve_k]
                else:
                    top_indices = np.argpartition(-scores, retrieve_k)[:retrieve_k]
                    top_indices = top_indices[np.argsort(-scores[top_indices])]

                # Filter true positives, take hardest n_hard
                hard_neg_ids = []
                hard_neg_scores = []
                for idx in top_indices:
                    pid = self._product_ids[idx]
                    if pid not in positive_ids:
                        hard_neg_ids.append(pid)
                        hard_neg_scores.append(float(scores[idx]))
                        if len(hard_neg_ids) >= n_hard:
                            break

                all_neg_scores.extend(hard_neg_scores)
                total_found += len(hard_neg_ids)

                results.append({
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "positive_ids": list(positive_ids),
                    "negative_ids": hard_neg_ids,
                })

            if show_progress:
                logger.info(
                    f"Mining progress: {batch_end}/{len(queries)} queries "
                    f"({100 * batch_end / len(queries):.0f}%)"
                )

        avg_score = np.mean(all_neg_scores) if all_neg_scores else 0.0
        coverage = total_found / (len(queries) * n_hard) * 100 if queries else 0.0
        logger.info(
            f"ANCE mining complete | queries={len(queries)} | "
            f"hard_negatives={total_found} | coverage={coverage:.1f}% | "
            f"avg_neg_similarity={avg_score:.4f}"
            + (f" | skipped_degenerate={n_skipped}" if n_skipped else "")
        )

        return results

    def reset(self) -> None:
        """Clear state and free memory."""
        self._corpus_matrix = None
        self._product_ids = []
        logger.info("ANCE index reset")


def _build_product_text(product: Dict) -> str:
    """Concatenate product fields into a single text for encoding."""
    parts = [
        product.get("title", ""),
        product.get("description", ""),
        product.get("bullet_points", ""),
    ]
    return " ".join(p for p in parts if p).strip()
