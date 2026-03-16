"""
Standalone evaluation script for a pre-trained SPLADE model.

Loads a fine-tuned model from SM_MODEL_DIR (or --model-dir), evaluates on test data,
and saves eval_metrics.json. Also evaluates the zero-shot baseline for comparison.

Usage (SageMaker):
  Entry script for a training job where the model artifact is provided as input.

Usage (local):
  SM_MODEL_DIR=/tmp/model SM_CHANNEL_TRAINING=data/esci python src/eval_only.py
"""

import json
import logging
import os
import sys
import tarfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
TRAINING_DIR = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
MODEL_CHANNEL = Path(os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model"))
BM25_BASELINE_PATH = TRAINING_DIR / "bm25_baseline_results.json"


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    pipeline_start = time.time()
    from sentence_transformers import SparseEncoder
    from evaluate import SpladeEvaluator

    # Resolve model path: either extracted dir or tar.gz to extract
    model_path = MODEL_CHANNEL
    if not (model_path / "config.json").exists():
        # Check for _best_checkpoint inside
        best_ckpt = model_path / "_best_checkpoint"
        if best_ckpt.exists() and (best_ckpt / "config.json").exists():
            model_path = best_ckpt
            logger.info(f"Using _best_checkpoint from model artifact")
        else:
            # Try extracting tar.gz
            tarballs = list(model_path.glob("*.tar.gz"))
            if tarballs:
                extract_dir = model_path / "extracted"
                extract_dir.mkdir(exist_ok=True)
                logger.info(f"Extracting {tarballs[0]} to {extract_dir}")
                with tarfile.open(tarballs[0]) as tf:
                    tf.extractall(extract_dir)
                if (extract_dir / "_best_checkpoint" / "config.json").exists():
                    model_path = extract_dir / "_best_checkpoint"
                else:
                    model_path = extract_dir

    logger.info(f"Model path: {model_path}")
    logger.info(f"Training data: {TRAINING_DIR}")
    logger.info(f"Output dir: {MODEL_DIR}")

    # Load test data and corpus
    logger.info("Loading test data and corpus...")
    t0 = time.time()
    from train import load_dataset
    _, test_queries, corpus, qrels = load_dataset(TRAINING_DIR, local_mode=False)
    logger.info(f"Data loaded in {time.time() - t0:.1f}s | test_queries={len(test_queries)} | corpus={len(corpus)}")

    eval_batch_size = 64
    base_model_name = "naver/splade-cocondenser-ensembledistil"

    # Read max_seq_length from fine-tuned model config for fair comparison
    ft_config_path = model_path / "config_sentence_transformers.json"
    max_seq_len = 256  # default from training config
    if ft_config_path.exists():
        with open(ft_config_path) as f:
            ft_config = json.load(f)
            max_seq_len = ft_config.get("max_seq_length", 256)
    logger.info(f"Using max_seq_length={max_seq_len} (from fine-tuned model config)")

    # Zero-shot baseline
    logger.info("=== ZERO-SHOT BASELINE ===")
    t0 = time.time()
    logger.info(f"Loading zero-shot model: {base_model_name}")
    zs_model = SparseEncoder(base_model_name)
    zs_model.max_seq_length = max_seq_len
    logger.info(f"Zero-shot model loaded in {time.time() - t0:.1f}s (max_seq_length={max_seq_len})")
    t0 = time.time()
    zs_evaluator = SpladeEvaluator(
        bm25_baseline_path=str(BM25_BASELINE_PATH) if BM25_BASELINE_PATH.exists() else None
    )
    zeroshot_results = zs_evaluator.evaluate(
        zs_model, corpus, test_queries, qrels, eval_batch_size=eval_batch_size
    )
    logger.info(f"Zero-shot eval completed in {time.time() - t0:.1f}s")
    for metric, value in zeroshot_results.items():
        print(json.dumps({"metric_name": f"zeroshot/{metric}", "value": round(value, 6)}), flush=True)
    del zs_model

    # Fine-tuned model
    logger.info("=== FINE-TUNED MODEL EVALUATION ===")
    t0 = time.time()
    logger.info(f"Loading fine-tuned model from: {model_path}")
    model = SparseEncoder(str(model_path))
    logger.info(f"Fine-tuned model loaded in {time.time() - t0:.1f}s")
    t0 = time.time()
    ft_evaluator = SpladeEvaluator(
        bm25_baseline_path=str(BM25_BASELINE_PATH) if BM25_BASELINE_PATH.exists() else None
    )
    ft_evaluator.zeroshot_results = zeroshot_results
    final_results = ft_evaluator.evaluate(
        model, corpus, test_queries, qrels, eval_batch_size=eval_batch_size
    )
    logger.info(f"Fine-tuned eval completed in {time.time() - t0:.1f}s")
    for metric, value in final_results.items():
        print(json.dumps({"metric_name": f"final/{metric}", "value": round(value, 6)}), flush=True)

    # Save results
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = MODEL_DIR / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "final": final_results,
            "zeroshot": zeroshot_results,
            "source_model": str(model_path),
        }, f, indent=2)

    total_elapsed = time.time() - pipeline_start
    logger.info(f"Results saved to {metrics_path}")
    logger.info(f"=== EVALUATION COMPLETE === Total: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} h)")


if __name__ == "__main__":
    main()
