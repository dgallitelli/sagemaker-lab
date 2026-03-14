"""
ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation) hard negative miner.

Uses Qdrant in-memory to index corpus sparse vectors and retrieve hard negatives.
No external network calls — QdrantClient(":memory:") only.
"""

import logging
from typing import Dict, List, Optional, Set

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SparseEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ANCEMiner:
    """
    In-memory hard negative miner using ANCE methodology.

    Encodes corpus with the current model state, indexes into Qdrant in-memory,
    then retrieves nearest neighbors as hard negatives (excluding true positives).
    """

    def __init__(self, collection_name: str = "ance_corpus"):
        # Always in-memory — no network calls, dies with the process
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self._idx_to_product_id: Dict[int, str] = {}
        self._product_id_to_idx: Dict[str, int] = {}
        self._indexed = False

    def _encode_to_sparse(
        self,
        model: SparseEncoder,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> List[Dict]:
        """Encode texts to Qdrant-compatible sparse vector dicts."""
        sparse_vectors = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", unit="batch")

        for i in iterator:
            batch = texts[i : i + batch_size]
            result = model.encode(
                batch,
                show_progress_bar=False,
                convert_to_sparse_tensor=False,
            )
            embeddings = result.cpu().numpy() if hasattr(result, "cpu") else result
            for emb in embeddings:
                # SPLADE output: dense array with many zeros
                nonzero_mask = emb > 0
                indices = np.where(nonzero_mask)[0].tolist()
                values = emb[nonzero_mask].tolist()
                sparse_vectors.append({"indices": indices, "values": values})

        return sparse_vectors

    def build_index(
        self,
        model: SparseEncoder,
        corpus: List[Dict],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> None:
        """
        Encode the full corpus and load into Qdrant in-memory.

        Args:
            model: Current SparseEncoder checkpoint (gets updated each ANCE iteration).
            corpus: List of dicts with keys: product_id, title, description, bullet_points.
            batch_size: Encoding batch size.
            show_progress: Show tqdm progress bar.
        """
        # Recreate collection from scratch (model weights change between iterations)
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={},  # no dense vectors
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

        # Build bidirectional ID maps (Qdrant uses int IDs)
        self._idx_to_product_id = {i: p["product_id"] for i, p in enumerate(corpus)}
        self._product_id_to_idx = {p["product_id"]: i for i, p in enumerate(corpus)}

        # Build document texts
        corpus_texts = [
            _build_product_text(p) for p in corpus
        ]

        logger.info(f"Encoding {len(corpus_texts)} corpus documents for ANCE index...")
        sparse_vecs = self._encode_to_sparse(
            model, corpus_texts, batch_size=batch_size, show_progress=show_progress
        )

        # Insert in batches (Qdrant in-memory handles large collections fine)
        insert_batch = 1000
        for i in range(0, len(corpus), insert_batch):
            batch_vecs = sparse_vecs[i : i + insert_batch]
            points = [
                models.PointStruct(
                    id=i + j,
                    vector={
                        "sparse": models.SparseVector(
                            indices=sv["indices"],
                            values=sv["values"],
                        )
                    },
                    payload={"product_id": corpus[i + j]["product_id"]},
                )
                for j, sv in enumerate(batch_vecs)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

        self._indexed = True
        logger.info(f"ANCE index built: {len(corpus)} documents indexed")

    def mine(
        self,
        model: SparseEncoder,
        queries: List[Dict],
        k: int = 50,
        n_hard: int = 5,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Retrieve hard negatives for each query.

        For each query, retrieves top-K similar corpus documents, filters out
        true positives, and returns the hardest N as negatives.

        Args:
            model: Current SparseEncoder checkpoint.
            queries: List of dicts with keys: query_id, query, positive_ids (list of str).
            k: How many candidates to retrieve per query.
            n_hard: How many hard negatives to keep per query.
            batch_size: Encoding batch size.
            show_progress: Show tqdm progress bar.

        Returns:
            List of dicts with keys: query_id, query, positive_ids, negative_ids.
        """
        if not self._indexed:
            raise RuntimeError("Call build_index() before mine()")

        query_texts = [q["query"] for q in queries]
        logger.info(f"Encoding {len(query_texts)} queries for hard negative mining...")
        query_sparse = self._encode_to_sparse(
            model, query_texts, batch_size=batch_size, show_progress=show_progress
        )

        results = []
        total_found = 0
        all_neg_scores: List[float] = []

        iterator = list(zip(queries, query_sparse))
        if show_progress:
            iterator = tqdm(iterator, desc="Mining negatives", unit="query")

        for query, qsv in iterator:
            positive_ids: Set[str] = set(query.get("positive_ids", []))

            # Skip queries with no sparse activation (degenerate encoding)
            if not qsv["indices"]:
                results.append(
                    {
                        "query_id": query["query_id"],
                        "query": query["query"],
                        "positive_ids": list(positive_ids),
                        "negative_ids": [],
                    }
                )
                continue

            result = self.client.query_points(
                collection_name=self.collection_name,
                query=models.SparseVector(
                    indices=qsv["indices"],
                    values=qsv["values"],
                ),
                using="sparse",
                limit=k,
                with_payload=True,
            )
            hits = result.points

            # Filter true positives; take hardest N (already sorted by score desc)
            hard_negatives = [
                h for h in hits if h.payload["product_id"] not in positive_ids
            ][:n_hard]

            neg_ids = [h.payload["product_id"] for h in hard_negatives]
            all_neg_scores.extend([h.score for h in hard_negatives])
            total_found += len(hard_negatives)

            results.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "positive_ids": list(positive_ids),
                    "negative_ids": neg_ids,
                }
            )

        avg_score = np.mean(all_neg_scores) if all_neg_scores else 0.0
        coverage = total_found / (len(queries) * n_hard) * 100
        logger.info(
            f"ANCE mining complete | queries={len(queries)} | "
            f"hard_negatives={total_found} | coverage={coverage:.1f}% | "
            f"avg_neg_similarity={avg_score:.4f}"
        )

        return results

    def reset(self) -> None:
        """Clear the Qdrant collection and reset state."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self._idx_to_product_id = {}
        self._product_id_to_idx = {}
        self._indexed = False
        logger.info("ANCE index reset")


def _build_product_text(product: Dict) -> str:
    """Concatenate product fields into a single text for encoding."""
    parts = [
        product.get("title", ""),
        product.get("description", ""),
        product.get("bullet_points", ""),
    ]
    return " ".join(p for p in parts if p).strip()
