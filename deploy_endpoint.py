"""
Deploy trained SPLADE model to a SageMaker real-time endpoint using HuggingFace TEI.

Usage:
  python deploy_endpoint.py --model-artifact s3://my-bucket/output/model.tar.gz
  python deploy_endpoint.py --model-artifact s3://... --endpoint-name splade-prod
  python deploy_endpoint.py --endpoint-name splade-prod --skip-deploy  # evaluate existing
  python deploy_endpoint.py --endpoint-name splade-prod --delete       # cleanup
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT_NAME = "splade-esci-endpoint"
INSTANCE_TYPE = "ml.g5.xlarge"


# ---------------------------------------------------------------------------
# Endpoint deployment
# ---------------------------------------------------------------------------

def deploy_endpoint(
    model_artifact_uri: str,
    endpoint_name: str,
    region: str,
) -> str:
    """
    Deploy SPLADE model to SageMaker endpoint using HuggingFace TEI container.

    Returns the endpoint name.
    """
    import sagemaker
    from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
    from sagemaker.core.helper.session_helper import get_execution_role

    session = sagemaker.Session()
    role = get_execution_role()

    # TEI (Text Embeddings Inference) image with SPLADE pooling support
    tei_image_uri = get_huggingface_llm_image_uri(
        "huggingface-tei",
        version="latest",
        region=region,
    )
    logger.info(f"TEI image: {tei_image_uri}")

    model = HuggingFaceModel(
        model_data=model_artifact_uri,
        role=role,
        image_uri=tei_image_uri,
        env={
            "POOLING": "splade",
            "MAX_BATCH_TOKENS": "16384",
            "MAX_CONCURRENT_REQUESTS": "64",
        },
        sagemaker_session=session,
    )

    logger.info(f"Deploying to endpoint: {endpoint_name} ({INSTANCE_TYPE})")
    predictor = model.deploy(
        endpoint_name=endpoint_name,
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
    )
    logger.info(f"Endpoint deployed: {endpoint_name}")
    return endpoint_name


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def encode_texts(
    endpoint_name: str,
    texts: List[str],
    region: str,
    batch_size: int = 32,
) -> List[Dict]:
    """
    Call TEI endpoint and return sparse vectors.

    TEI SPLADE output format:
      [{"index": int, "value": float}, ...]  per text
    """
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = json.dumps({"inputs": batch})
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        result = json.loads(response["Body"].read())
        # Normalize output: TEI may return list of sparse dicts or dense arrays
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                # Dense output — convert to sparse format
                for dense_vec in result:
                    sparse = [
                        {"index": i, "value": float(v)}
                        for i, v in enumerate(dense_vec)
                        if v > 0
                    ]
                    all_vectors.append(sparse)
            else:
                all_vectors.extend(result)
        else:
            all_vectors.extend(result)

    return all_vectors


def sparse_dot_product(vec_a: List[Dict], vec_b: List[Dict]) -> float:
    """Compute dot product between two sparse vectors (list of {index, value} dicts)."""
    b_map = {item["index"]: item["value"] for item in vec_b}
    return sum(item["value"] * b_map.get(item["index"], 0.0) for item in vec_a)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(endpoint_name: str, region: str) -> None:
    """Send 5 test queries and print top-10 activated terms."""
    try:
        import boto3
        sm = boto3.client("sagemaker", region_name=region)
        tokenizer_name = "naver/splade-cocondenser-ensembledistil"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except ImportError:
        logger.warning("transformers not installed — skipping token decoding in smoke test")
        tokenizer = None

    test_queries = [
        "iPhone 256GB storage",
        "wireless noise cancelling headphones",
        "USB-C fast charging cable 6ft",
        "coffee maker programmable 12 cup",
        "running shoes breathable lightweight",
    ]

    logger.info("Running smoke test...")
    vectors = encode_texts(endpoint_name, test_queries, region)

    print("\n=== SMOKE TEST: Top-10 Activated Terms ===")
    for query, vec in zip(test_queries, vectors):
        top_terms = sorted(vec, key=lambda x: x["value"], reverse=True)[:10]
        print(f"\nQuery: {query!r}")
        if tokenizer:
            for item in top_terms:
                token = tokenizer.convert_ids_to_tokens([item["index"]])[0]
                print(f"  {token:<20} {item['value']:.4f}")
        else:
            for item in top_terms:
                print(f"  token_id={item['index']:<8} {item['value']:.4f}")

    print(f"\nSmoke test passed: {len(vectors)} queries encoded successfully\n")


# ---------------------------------------------------------------------------
# Full evaluation via endpoint
# ---------------------------------------------------------------------------

def evaluate_endpoint(
    endpoint_name: str,
    data_dir: Path,
    region: str,
    eval_batch_size: int = 32,
) -> Dict:
    """
    Run full evaluation against test set using the deployed endpoint.
    Compares against BM25 baseline and saves final_results.json.
    """
    import math
    import numpy as np
    from collections import defaultdict

    logger.info("Loading evaluation data...")
    with open(data_dir / "corpus.jsonl") as f:
        corpus = [json.loads(line) for line in f if line.strip()]
    with open(data_dir / "test.jsonl") as f:
        test_pairs = [json.loads(line) for line in f if line.strip()]

    bm25_results = {}
    bm25_path = data_dir / "bm25_baseline_results.json"
    if bm25_path.exists():
        with open(bm25_path) as f:
            bm25_results = json.load(f)

    LABEL_SCORES = {"E": 1.0, "S": 0.5, "C": 0.1, "I": 0.0}
    qrels: Dict[str, Dict[str, float]] = defaultdict(dict)
    query_map: Dict[str, str] = {}
    for pair in test_pairs:
        qrels[pair["query_id"]][pair["product_id"]] = LABEL_SCORES.get(
            pair.get("esci_label", "I"), 0.0
        )
        query_map[pair["query_id"]] = pair["query"]

    # Encode corpus
    corpus_texts = [
        " ".join(filter(None, [p.get("title"), p.get("description"), p.get("bullet_points")]))
        for p in corpus
    ]
    product_ids = [p["product_id"] for p in corpus]

    logger.info(f"Encoding {len(corpus)} corpus documents via endpoint...")
    corpus_vectors = encode_texts(endpoint_name, corpus_texts, region, batch_size=eval_batch_size)

    # Encode queries
    test_queries = list(query_map.items())
    query_texts = [q for _, q in test_queries]
    logger.info(f"Encoding {len(query_texts)} test queries via endpoint...")
    query_vectors = encode_texts(endpoint_name, query_texts, region, batch_size=eval_batch_size)

    # Evaluate
    ndcg_scores, recall_scores, mrr_scores = [], [], []
    for (qid, _), qvec in zip(test_queries, query_vectors):
        if qid not in qrels:
            continue
        scores = [sparse_dot_product(qvec, cvec) for cvec in corpus_vectors]
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked_pids = [product_ids[i] for i in ranked_indices]
        query_qrels = qrels[qid]

        ndcg_scores.append(_ndcg_at_k(ranked_pids, query_qrels, 10))
        recall_scores.append(_recall_at_k(ranked_pids, query_qrels, 100))
        mrr_scores.append(_mrr_at_k(ranked_pids, query_qrels, 10))

    splade_results = {
        "ndcg@10":    float(np.mean(ndcg_scores)),
        "recall@100": float(np.mean(recall_scores)),
        "mrr@10":     float(np.mean(mrr_scores)),
    }

    # Build delta
    delta_pct = {}
    for metric in ["ndcg@10", "recall@100", "mrr@10"]:
        bm25_val = bm25_results.get(metric, 0.0)
        splade_val = splade_results[metric]
        if bm25_val > 0:
            delta_pct[metric] = f"{(splade_val - bm25_val) / bm25_val * 100:+.1f}%"
        else:
            delta_pct[metric] = "N/A"

    # Print comparison
    print("\n" + "=" * 65)
    print(f"  {'Metric':<15} {'BM25':>12} {'SPLADE':>12} {'Delta':>12}")
    print("  " + "-" * 55)
    for metric in ["ndcg@10", "recall@100", "mrr@10"]:
        bm25_val = bm25_results.get(metric, 0.0)
        splade_val = splade_results[metric]
        print(f"  {metric:<15} {bm25_val:>12.4f} {splade_val:>12.4f} {delta_pct[metric]:>12}")
    print("=" * 65 + "\n")

    return {
        "bm25_baseline": bm25_results,
        "splade_trained": splade_results,
        "delta_pct": delta_pct,
    }


def _ndcg_at_k(ranked_pids, qrels, k):
    gains = [qrels.get(pid, 0.0) for pid in ranked_pids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked_pids, qrels, k):
    retrieved = set(ranked_pids[:k])
    relevant = {pid for pid, s in qrels.items() if s > 0}
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0


def _mrr_at_k(ranked_pids, qrels, k):
    for i, pid in enumerate(ranked_pids[:k]):
        if qrels.get(pid, 0.0) > 0:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy SPLADE endpoint and evaluate")
    parser.add_argument("--model-artifact", help="S3 URI of model.tar.gz from training job")
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT_NAME)
    parser.add_argument("--data-dir", default="data", help="Directory with test/corpus JSONL files")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip deployment, use existing endpoint")
    parser.add_argument("--skip-eval", action="store_true", help="Skip full evaluation")
    parser.add_argument("--delete", action="store_true", help="Delete endpoint and exit")
    parser.add_argument("--region", default=None, help="AWS region (defaults to boto3 default)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import boto3
    region = args.region or boto3.session.Session().region_name or "us-east-1"
    data_dir = Path(args.data_dir)

    # ── Delete mode ──────────────────────────────────────────────────────────
    if args.delete:
        sm = boto3.client("sagemaker", region_name=region)
        logger.info(f"Deleting endpoint: {args.endpoint_name}")
        sm.delete_endpoint(EndpointName=args.endpoint_name)
        logger.info("Endpoint deleted.")
        return

    # ── Deploy ───────────────────────────────────────────────────────────────
    if not args.skip_deploy:
        if not args.model_artifact:
            logger.error("--model-artifact is required for deployment")
            sys.exit(1)
        deploy_endpoint(args.model_artifact, args.endpoint_name, region)

    # ── Smoke test ───────────────────────────────────────────────────────────
    run_smoke_test(args.endpoint_name, region)

    # ── Full evaluation ──────────────────────────────────────────────────────
    results = {}
    if not args.skip_eval:
        results = evaluate_endpoint(args.endpoint_name, data_dir, region)

    # ── Save final_results.json ──────────────────────────────────────────────
    final_results = {
        **results,
        "model_artifact_s3": args.model_artifact or "N/A",
        "endpoint_name": args.endpoint_name,
        "instance_type": INSTANCE_TYPE,
        # training_cost_usd and training_duration_minutes populated by sagemaker_launcher.py
        "training_cost_usd": None,
        "training_duration_minutes": None,
    }

    with open("final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info("Results saved to final_results.json")

    print("\n=== DONE ===")
    print(f"Endpoint:          {args.endpoint_name}")
    print(f"Results saved to:  final_results.json")
    print(f"\nTo clean up: python deploy_endpoint.py --endpoint-name {args.endpoint_name} --delete")


if __name__ == "__main__":
    main()
