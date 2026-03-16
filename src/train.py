"""
SageMaker training entry point for SPLADE sparse embedding fine-tuning.

Supports multiple datasets (ESCI, FiQA, NFCorpus) via normalized JSONL format.

Training phases:
  1. Contrastive learning with in-batch negatives (E-labeled positives)
  2. ANCE hard negative mining (1 iteration by default, configurable)
  3. Best-model selection: compares Phase 1 vs ANCE, keeps best NDCG@10

Reads hyperparameters from /opt/ml/input/config/hyperparameters.json (SageMaker standard).
Override paths via SM_MODEL_DIR / SM_CHANNEL_TRAINING env vars for local mode.

Logs metrics in CloudWatch-parseable format: {"metric_name": "ndcg@10", "value": 0.72}
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import time

from evaluate import _build_product_text

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution (works for both SageMaker and --local mode)
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
TRAINING_DIR = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
HP_PATH = Path(os.environ.get("SM_HPS_PATH", "/opt/ml/input/config/hyperparameters.json"))
BM25_BASELINE_PATH = TRAINING_DIR / "bm25_baseline_results.json"


# ---------------------------------------------------------------------------
# Hyperparameter loading
# ---------------------------------------------------------------------------

def load_hyperparameters(config_path: Path = HP_PATH) -> Dict:
    """
    Load hyperparameters from SageMaker HP file or fall back to config.yaml.
    SageMaker passes all HP values as strings — convert numeric types.
    """
    defaults = _load_config_yaml()

    if config_path.exists() and config_path.stat().st_size > 0:
        with open(config_path) as f:
            sm_hps = json.load(f)
        # SageMaker stringifies everything
        for key, val in sm_hps.items():
            try:
                sm_hps[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
        defaults.update(sm_hps)
        logger.info(f"Loaded SageMaker hyperparameters: {sm_hps}")
    else:
        logger.info("No SageMaker HP file found, using config.yaml defaults")

    return defaults


def _load_config_yaml() -> Dict:
    config_file = Path(__file__).parent / "config.yaml"
    if not config_file.exists():
        return {}
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    # Flatten nested config for easy access
    flat = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(training_dir: Path, local_mode: bool = False) -> Tuple[List, List, List, Dict]:
    """
    Load train pairs, test pairs, corpus, and qrels from training directory.

    Returns:
        train_pairs: [{query_id, query, positive_id, negative_id?, esci_label}]
        test_queries: [{query_id, query}]
        corpus: [{product_id, title, description, bullet_points}]
        qrels: {query_id: {product_id: score}}
    """
    train_pairs = load_jsonl(training_dir / "train.jsonl")
    test_pairs = load_jsonl(training_dir / "test.jsonl")
    corpus = load_jsonl(training_dir / "corpus.jsonl")

    if local_mode:
        logger.info("Local mode: truncating dataset for fast iteration")
        max_train = int(os.environ.get("LOCAL_MAX_TRAIN", 500))
        max_test = int(os.environ.get("LOCAL_MAX_TEST", 100))
        max_corpus = int(os.environ.get("LOCAL_MAX_CORPUS", 5000))
        train_pairs = train_pairs[:max_train]
        test_pairs = test_pairs[:max_test]
        # Keep only corpus products that appear in our subset
        relevant_pids = (
            {p["positive_id"] for p in train_pairs}
            | {p["product_id"] for p in test_pairs}
        )
        corpus = [p for p in corpus if p["product_id"] in relevant_pids][:max_corpus]

    # Build qrels from test pairs (prefer raw_score for graded datasets like NFCorpus)
    LABEL_SCORES = {"E": 1.0, "S": 0.5, "C": 0.1, "I": 0.0}
    qrels: Dict[str, Dict[str, float]] = {}
    test_queries_map: Dict[str, str] = {}
    for pair in test_pairs:
        qid = pair["query_id"]
        pid = pair["product_id"]
        score = pair.get("raw_score", LABEL_SCORES.get(pair.get("esci_label", "I"), 0.0))
        qrels.setdefault(qid, {})[pid] = score
        test_queries_map[qid] = pair["query"]

    test_queries = [{"query_id": qid, "query": q} for qid, q in test_queries_map.items()]

    logger.info(
        f"Dataset loaded | train_pairs={len(train_pairs)} | "
        f"test_queries={len(test_queries)} | corpus={len(corpus)}"
    )
    return train_pairs, test_queries, corpus, qrels


# ---------------------------------------------------------------------------
# Training dataset construction
# ---------------------------------------------------------------------------

def build_hf_dataset(
    train_pairs: List[Dict],
    corpus_map: Dict[str, Dict],
    hard_negatives_map: Optional[Dict[str, List[str]]] = None,
):
    """
    Build a HuggingFace Dataset from training pairs.

    Columns: anchor (query), positive (doc text), optional negative (doc text).
    SparseEncoderMultipleNegativesRankingLoss uses in-batch negatives when
    no explicit negative column is present.
    """
    from datasets import Dataset

    rows = []
    for pair in train_pairs:
        pos_product = corpus_map.get(pair.get("positive_id") or pair.get("product_id"))
        if not pos_product:
            continue

        entry = {
            "anchor": pair["query"],
            "positive": _build_product_text(pos_product),
        }

        # Add hard negative if available for this query
        if hard_negatives_map:
            neg_ids = hard_negatives_map.get(pair["query_id"], [])
            neg_text = None
            for nid in neg_ids:
                neg_product = corpus_map.get(nid)
                if neg_product:
                    neg_text = _build_product_text(neg_product)
                    if neg_text:
                        break
            if neg_text:
                entry["negative"] = neg_text
            else:
                continue  # skip pairs without hard negatives to keep columns consistent

        rows.append(entry)

    logger.info(
        f"Training dataset: {len(rows)} pairs, "
        f"with_negatives={sum(1 for r in rows if 'negative' in r)}"
    )
    return Dataset.from_list(rows)



# ---------------------------------------------------------------------------
# Metric logging (CloudWatch-parseable)
# ---------------------------------------------------------------------------

def log_metric(name: str, value: float, step: Optional[int] = None) -> None:
    """Emit metric in CloudWatch metric filter format."""
    payload = {"metric_name": name, "value": round(value, 6)}
    if step is not None:
        payload["step"] = step
    print(json.dumps(payload), flush=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training_phase(
    model,
    train_dataset,
    loss,
    output_dir: Path,
    hp: Dict,
    phase_name: str,
    batch_size_override: Optional[int] = None,
    learning_rate_override: Optional[float] = None,
    warmup_override: Optional[float] = None,
) -> None:
    """Run one training phase with SparseEncoderTrainer."""
    from sentence_transformers.sparse_encoder.trainer import SparseEncoderTrainer
    from sentence_transformers.sparse_encoder.training_args import SparseEncoderTrainingArguments

    batch_size = batch_size_override or hp.get("batch_size", 32)
    lr = learning_rate_override or float(hp.get("learning_rate", 2e-5))
    warmup = warmup_override if warmup_override is not None else float(hp.get("warmup_ratio", 0.1))
    phase_dir = output_dir / phase_name
    args = SparseEncoderTrainingArguments(
        output_dir=str(phase_dir),
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup,  # float = warmup ratio in Transformers v5+
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # 0 avoids fork issues on macOS
        logging_steps=50,
        save_strategy="no",  # we save manually at end
        report_to="none",    # log to stdout only
    )

    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    logger.info(f"Starting training phase: {phase_name}")
    train_result = trainer.train()
    log_metric(f"{phase_name}/train_loss", train_result.training_loss)
    logger.info(f"Phase {phase_name} complete | loss={train_result.training_loss:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline_start = time.time()
    hp = load_hyperparameters()
    local_mode = os.environ.get("LOCAL_MODE", "0") == "1"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model output dir: {MODEL_DIR}")
    logger.info(f"Training data dir: {TRAINING_DIR}")
    logger.info(f"Hyperparameters: {hp}")
    logger.info(f"Local mode: {local_mode}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    else:
        logger.info("No CUDA GPUs available — running on CPU")

    # ── Imports (done here so errors surface early) ──────────────────────────
    from sentence_transformers import SparseEncoder
    from sentence_transformers.sparse_encoder.losses import (
        SpladeLoss,
        SparseMultipleNegativesRankingLoss,
    )
    from ance_miner import ANCEMiner
    from evaluate import SpladeEvaluator

    # ── Load data ────────────────────────────────────────────────────────────
    t0 = time.time()
    train_pairs, test_queries, corpus, qrels = load_dataset(
        TRAINING_DIR, local_mode=local_mode
    )
    corpus_map = {p["product_id"]: p for p in corpus}
    logger.info(f"Data loaded in {time.time() - t0:.1f}s")

    # Build initial easy-negative dataset:
    # Positives = E-labeled pairs; negatives = I-labeled products (in-batch)
    easy_pairs = [p for p in train_pairs if p.get("esci_label") == "E"]
    if not easy_pairs:
        easy_pairs = train_pairs  # fallback if labels not available

    # ── Initialize model ─────────────────────────────────────────────────────
    base_model = hp.get("base", "naver/splade-cocondenser-ensembledistil")
    max_seq_len = hp.get("max_seq_length", 256)
    logger.info(f"Loading base model: {base_model}")
    t0 = time.time()
    model = SparseEncoder(base_model)
    model.max_seq_length = max_seq_len
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Multi-GPU: sentence-transformers Trainer handles DataParallel internally
    # via HuggingFace accelerate. Explicit DataParallel not needed.
    if torch.cuda.device_count() > 1:
        logger.info(f"Multi-GPU training: {torch.cuda.device_count()} GPUs available")

    flops_weight = float(hp.get("flops_weight", 4e-4))

    def make_loss(m):
        return SpladeLoss(
            model=m,
            loss=SparseMultipleNegativesRankingLoss(model=m),
            query_regularizer_weight=flops_weight,
            document_regularizer_weight=flops_weight,
        )

    # ── Zero-shot baseline (before any training, model IS the base model) ───
    eval_batch_size = hp.get("eval_batch_size", 64)
    logger.info("=== ZERO-SHOT BASELINE ===")
    t0 = time.time()
    zeroshot_evaluator = SpladeEvaluator(
        bm25_baseline_path=str(BM25_BASELINE_PATH)
        if BM25_BASELINE_PATH.exists()
        else None
    )
    zeroshot_results = zeroshot_evaluator.evaluate(
        model, corpus, test_queries, qrels, eval_batch_size=eval_batch_size
    )
    logger.info(f"Zero-shot eval completed in {time.time() - t0:.1f}s")
    for metric, value in zeroshot_results.items():
        log_metric(f"zeroshot/{metric}", value)

    # ── Phase 1: Initial training on easy negatives ──────────────────────────
    logger.info("=== PHASE 1: Initial training (easy negatives) ===")
    t0 = time.time()
    train_dataset = build_hf_dataset(easy_pairs, corpus_map)
    run_training_phase(model, train_dataset, make_loss(model), MODEL_DIR, hp, "phase1_easy")
    logger.info(f"Phase 1 training completed in {time.time() - t0:.1f}s")

    eval_results_per_phase = []

    # ── P0: Evaluate after Phase 1 ──────────────────────────────────────────
    logger.info("Evaluating after Phase 1...")
    t0 = time.time()
    phase1_evaluator = SpladeEvaluator(
        bm25_baseline_path=str(BM25_BASELINE_PATH)
        if BM25_BASELINE_PATH.exists()
        else None
    )
    phase1_evaluator.zeroshot_results = zeroshot_results
    phase1_results = phase1_evaluator.evaluate(
        model, corpus, test_queries, qrels, eval_batch_size=eval_batch_size
    )
    logger.info(f"Phase 1 eval completed in {time.time() - t0:.1f}s")
    eval_results_per_phase.append(phase1_results)
    for metric, value in phase1_results.items():
        log_metric(f"phase1/{metric}", value, step=0)

    # ── P1: Best-model selection — save Phase 1 as initial best ─────────────
    best_ndcg = phase1_results.get("ndcg@10", 0.0)
    best_phase = "phase1_easy"
    best_model_dir = MODEL_DIR / "_best_checkpoint"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(best_model_dir))
    logger.info(f"Best model checkpoint: {best_phase} (NDCG@10={best_ndcg:.4f})")

    # ── ANCE iterations ──────────────────────────────────────────────────────
    ance_iters = hp.get("ance_iterations", 1)
    k_mining = hp.get("top_k_mining", 50)
    n_hard = hp.get("hard_negatives_per_query", 5)
    ance_lr = float(hp.get("ance_learning_rate", hp.get("learning_rate", 2e-5)))
    miner = ANCEMiner()

    # Build query structs for mining (unique queries with their positive IDs)
    query_to_positives: Dict[str, List[str]] = {}
    for pair in train_pairs:
        if pair.get("esci_label") in ("E", "S"):
            qid = pair["query_id"]
            pid = pair.get("positive_id") or pair.get("product_id")
            query_to_positives.setdefault(qid, []).append(pid)

    query_text_map = {p["query_id"]: p["query"] for p in train_pairs}
    mining_queries = [
        {"query_id": qid, "query": query_text_map.get(qid, ""), "positive_ids": pids}
        for qid, pids in query_to_positives.items()
        if query_text_map.get(qid)
    ]

    for ance_iter in range(1, ance_iters + 1):
        logger.info(f"=== ANCE ITERATION {ance_iter} ===")
        ance_iter_start = time.time()

        # Mine hard negatives with current model state
        t0 = time.time()
        miner.build_index(model, corpus, batch_size=eval_batch_size, show_progress=True)
        hard_neg_results = miner.mine(
            model, mining_queries, k=k_mining, n_hard=n_hard, batch_size=eval_batch_size
        )
        miner.reset()  # free mining state before training

        # {query_id: [neg_pid, ...]}
        hard_neg_map = {r["query_id"]: r["negative_ids"] for r in hard_neg_results}
        logger.info(
            f"ANCE iter {ance_iter}: {sum(len(v) for v in hard_neg_map.values())} "
            f"hard negatives mined across {len(hard_neg_map)} queries "
            f"in {time.time() - t0:.1f}s"
        )

        # P2: Lower LR for ANCE, P4: No warmup on continuation phases
        t0 = time.time()
        ance_batch_size = hp.get("ance_batch_size", hp.get("batch_size", 32))
        ance_dataset = build_hf_dataset(easy_pairs, corpus_map, hard_negatives_map=hard_neg_map)
        run_training_phase(
            model, ance_dataset, make_loss(model), MODEL_DIR, hp, f"phase_ance{ance_iter}",
            batch_size_override=ance_batch_size,
            learning_rate_override=ance_lr,
            warmup_override=0.0,
        )

        logger.info(f"ANCE iter {ance_iter} training completed in {time.time() - t0:.1f}s")

        # Evaluate after ANCE — re-encode corpus with updated model weights
        logger.info(f"Evaluating after ANCE iteration {ance_iter} (fresh corpus encoding)...")
        t0 = time.time()
        evaluator = SpladeEvaluator(
            bm25_baseline_path=str(BM25_BASELINE_PATH)
            if BM25_BASELINE_PATH.exists()
            else None
        )
        evaluator.zeroshot_results = zeroshot_results
        iter_results = evaluator.evaluate(
            model, corpus, test_queries, qrels, eval_batch_size=eval_batch_size,
        )
        logger.info(f"ANCE iter {ance_iter} eval completed in {time.time() - t0:.1f}s")
        eval_results_per_phase.append(iter_results)

        for metric, value in iter_results.items():
            log_metric(f"ance_iter{ance_iter}/{metric}", value, step=ance_iter)

        # P1: Update best model if this ANCE iteration improved
        iter_ndcg = iter_results.get("ndcg@10", 0.0)
        if iter_ndcg > best_ndcg:
            best_ndcg = iter_ndcg
            best_phase = f"phase_ance{ance_iter}"
            model.save(str(best_model_dir))
            logger.info(f"New best model: {best_phase} (NDCG@10={best_ndcg:.4f})")
        else:
            logger.info(
                f"ANCE iter {ance_iter} did not improve (NDCG@10={iter_ndcg:.4f} "
                f"vs best={best_ndcg:.4f} from {best_phase}). Keeping best checkpoint."
            )
        logger.info(f"ANCE iteration {ance_iter} total: {time.time() - ance_iter_start:.1f}s")

    # ── P1: Restore best checkpoint if last phase wasn't the best ───────────
    last_phase = f"phase_ance{ance_iters}" if ance_iters > 0 else "phase1_easy"
    if best_phase != last_phase:
        logger.info(f"Restoring best model from {best_phase} (NDCG@10={best_ndcg:.4f})")
        model = SparseEncoder(str(best_model_dir))
        model.max_seq_length = max_seq_len

    # ── Final evaluation ─────────────────────────────────────────────────────
    # Skip redundant re-encoding if best model is from a phase we already evaluated
    if best_phase == "phase1_easy" and eval_results_per_phase:
        logger.info(f"=== FINAL EVALUATION (reusing Phase 1 results — best_phase={best_phase}) ===")
        final_results = eval_results_per_phase[0]
    elif best_phase.startswith("phase_ance") and len(eval_results_per_phase) > 1:
        ance_idx = int(best_phase.replace("phase_ance", ""))
        logger.info(f"=== FINAL EVALUATION (reusing {best_phase} results) ===")
        final_results = eval_results_per_phase[ance_idx]  # index 0=phase1, 1=ance1, ...
    else:
        logger.info("=== FINAL EVALUATION ===")
        final_evaluator = SpladeEvaluator(
            bm25_baseline_path=str(BM25_BASELINE_PATH) if BM25_BASELINE_PATH.exists() else None
        )
        final_evaluator.zeroshot_results = zeroshot_results
        final_results = final_evaluator.evaluate(
            model, corpus, test_queries, qrels, eval_batch_size=eval_batch_size
        )

    for metric, value in final_results.items():
        log_metric(f"final/{metric}", value)

    # ── Save model ───────────────────────────────────────────────────────────
    logger.info(f"Saving best model ({best_phase}) to {MODEL_DIR}...")
    model.save(str(MODEL_DIR))

    metrics_path = MODEL_DIR / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "final": final_results,
                "zeroshot": zeroshot_results,
                "best_phase": best_phase,
                "per_phase": eval_results_per_phase,
                "hyperparameters": hp,
            },
            f,
            indent=2,
        )
    # Clean up temporary best checkpoint
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)

    logger.info(f"Metrics saved to {metrics_path}")
    total_elapsed = time.time() - pipeline_start
    logger.info(
        f"=== PIPELINE COMPLETE === "
        f"Total: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} h) | "
        f"Best phase: {best_phase} (NDCG@10={best_ndcg:.4f})"
    )


if __name__ == "__main__":
    main()
