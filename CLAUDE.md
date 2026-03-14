# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Fine-tune a SPLADE sparse embedding model (SentenceTransformers v5 SparseEncoder) using ANCE hard negative mining, targeting performance above BM25 baseline. Primary use case: Amazon ESCI (e-commerce). Extended to FiQA (financial) and NFCorpus (biomedical) domains. Runs locally or on AWS SageMaker.

## Project Structure

```
src/                            # Shared training pipeline (dataset-agnostic)
  train.py                      # Entry point (SageMaker + local mode)
  ance_miner.py                 # ANCE hard negative mining via Qdrant in-memory
  evaluate.py                   # NDCG@10 / Recall@100 / MRR@10 + zero-shot baseline
  sagemaker_launcher.py         # Launch job or run locally with --local [--no-truncate]
  config.yaml                   # Hyperparameters
  requirements.txt
datasets/                       # Per-dataset data preparation
  common.py                     # Shared: BM25 eval, metrics, corpus building, I/O
  prepare_esci.py               # Amazon ESCI (e-commerce, 4-level graded)
  prepare_fiqa.py               # FiQA (financial QA, binary relevance)
  prepare_nfcorpus.py           # NFCorpus (biomedical, graded relevance with raw_score)
data/                           # Generated output (gitignored)
  esci/                         # train.jsonl, test.jsonl, corpus.jsonl, bm25_baseline_results.json
  fiqa/
  nfcorpus/
deploy_endpoint.py              # Deploy trained model to SageMaker TEI endpoint
```

## Execution Order

```bash
# 1. Prepare data and BM25 baseline (pick one or more datasets)
python datasets/prepare_esci.py --output-dir data/esci
python datasets/prepare_esci.py --output-dir data/esci --max-pairs 200000  # subsample
python datasets/prepare_fiqa.py --output-dir data/fiqa
python datasets/prepare_nfcorpus.py --output-dir data/nfcorpus

# 2. Train locally (truncated for fast iteration)
python src/sagemaker_launcher.py --local --data-dir data/fiqa

# 3. Train locally (full dataset, no truncation)
python src/sagemaker_launcher.py --local --data-dir data/fiqa --no-truncate \
  --local-output local_model_output_fiqa_full

# 4. Train on SageMaker (large datasets like ESCI)
python src/sagemaker_launcher.py --s3-bucket <bucket> --data-prefix splade-esci/data

# 5. Deploy / cleanup
python deploy_endpoint.py --model-artifact s3://<bucket>/splade-training-output/model.tar.gz
python deploy_endpoint.py --endpoint-name splade-esci-endpoint --delete
```

## SageMaker SDK v3

Always use v3 — see global `~/.claude/docs/sagemaker-v3.md`. Key imports:
```python
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import InputData, Compute, SourceCode, OutputDataConfig
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.image_uris import retrieve
```
- `SourceCode(entry_script="train.py", ...)` — NOT `entry_point`
- `ModelTrainer(role=role, sagemaker_session=session, ...)` — role/session on ModelTrainer
- `trainer.train(..., wait=True, logs=True)` — `wait`/`logs` go on `train()`, NOT `ModelTrainer()`
- `output_data_config=OutputDataConfig(s3_output_path=...)` not `output_path=`
- **SDK v3.4.1 bug**: `get_training_code_hash()` crashes when `requirements` is a str. Patch applied in `sagemaker_launcher.py` before job submission. Bug tracked: https://github.com/aws/sagemaker-python-sdk/issues/5518

## Key Design Decisions

- **Qdrant always in-memory**: `QdrantClient(":memory:")` — no external connections, no cleanup needed
- **Local mode**: `sagemaker_launcher.py --local` truncates to 500/100/5000 by default; `--no-truncate` uses full data
- **Zero-shot baseline**: `evaluate.py` evaluates base model (no fine-tuning) before fine-tuned model. Results table: BM25 | zero-shot | fine-tuned
- **NFCorpus raw scores**: `raw_score` field preserves graded relevance (0/1/2) for NDCG instead of collapsing through ESCI label buckets
- **CloudWatch metrics**: `train.py` logs `{"metric_name": "...", "value": ...}` JSON to stdout
- **BM25 baseline**: `bm25_baseline_results.json` per dataset is the ground truth to beat

## AWS Conventions

- Always clean up endpoints: `python deploy_endpoint.py --endpoint-name <name> --delete`
- Spot instances enabled by default in launcher (max_wait=7200s)
