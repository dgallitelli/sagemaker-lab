# SPLADE Fine-Tuning on SageMaker

Fine-tune a [SPLADE](https://arxiv.org/abs/2109.10086) sparse embedding model using ANCE hard negative mining on Amazon SageMaker. Built on [sentence-transformers v5](https://sbert.net/) `SparseEncoder` with the `naver/splade-cocondenser-ensembledistil` base model. Targets three domains: Amazon ESCI (e-commerce), FiQA (financial QA), and NFCorpus (biomedical).

## Results

Fine-tuned SPLADE vs BM25 baseline (NDCG@10 / Recall@100 / MRR@10):

```
Dataset     Corpus     BM25                        Zero-shot                   Fine-tuned
            size       NDCG   R@100  MRR           NDCG   R@100  MRR           NDCG   R@100  MRR
----------  --------   -----  -----  -----         -----  -----  -----         -----  -----  -----
ESCI-200k   503,839    0.377  0.535  0.586         0.463  0.640  0.671         0.489  0.685  0.686
FiQA         57,638    0.159  0.359  0.199         0.356  0.636  0.426         0.365  0.694  0.442
NFCorpus      3,633    0.267  0.211  0.467         0.348  0.284  0.575         0.368  0.416  0.556
```

ANCE hard negative mining improves recall significantly but can regress on small datasets. Best-model selection (Phase 1 checkpoint vs ANCE) is applied automatically.

## Project Structure

```
src/                            # Shared training pipeline (dataset-agnostic)
  train.py                      # Entry point (SageMaker + local mode)
  ance_miner.py                 # ANCE hard negative mining via scipy sparse matrices
  evaluate.py                   # NDCG@10 / Recall@100 / MRR@10 + zero-shot baseline
  sagemaker_launcher.py         # Launch SageMaker job or run locally with --local
  config.yaml                   # Hyperparameters
  requirements.txt
datasets/                       # Per-dataset data preparation
  common.py                     # Shared: BM25 eval, metrics, corpus building, I/O
  prepare_esci.py               # Amazon ESCI (e-commerce, 4-level graded relevance)
  prepare_fiqa.py               # FiQA (financial QA, binary relevance)
  prepare_nfcorpus.py           # NFCorpus (biomedical, graded relevance)
  sagemaker_processing.py       # Launch SageMaker Processing for large datasets
data/                           # Generated output (gitignored)
  esci/                         # train.jsonl, test.jsonl, corpus.jsonl, bm25_baseline_results.json
  fiqa/
  nfcorpus/
deploy_endpoint.py              # Deploy trained model to SageMaker TEI endpoint
```

## Training Pipeline

```
Phase 1: In-batch negatives (positives only, contrastive learning)
    |
ANCE Iter 1: Encode corpus -> mine hard negatives -> retrain with triplets
    |
Best-model selection: compare Phase 1 vs ANCE checkpoints on dev set
    |
Final evaluation: BM25 | zero-shot | fine-tuned comparison
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install sentence-transformers datasets bm25s scipy numpy tqdm pyyaml boto3 accelerate
```

## Usage

### 1. Prepare data

```bash
# Small datasets -- run locally
python datasets/prepare_fiqa.py --output-dir data/fiqa
python datasets/prepare_nfcorpus.py --output-dir data/nfcorpus

# Large datasets -- run on SageMaker Processing
python datasets/sagemaker_processing.py --dataset esci
python datasets/sagemaker_processing.py --dataset esci --max-pairs 200000  # subsample
```

Each script produces `train.jsonl`, `test.jsonl`, `corpus.jsonl`, and `bm25_baseline_results.json`.

### 2. Train locally (fast iteration)

```bash
# Truncated dataset (500 train, 100 test, 5000 corpus) for quick validation
python src/sagemaker_launcher.py --local --data-dir data/fiqa

# Full dataset, no truncation
python src/sagemaker_launcher.py --local --data-dir data/fiqa --no-truncate \
  --local-output local_model_output_fiqa_full
```

### 3. Train on SageMaker

```bash
python src/sagemaker_launcher.py --data-prefix splade-data/esci --skip-upload
```

Uses `ml.g5.2xlarge` with spot instances by default (max_wait=7200s).

### 4. Deploy and cleanup

```bash
# Deploy to SageMaker real-time endpoint (TEI container)
python deploy_endpoint.py --model-artifact s3://<bucket>/splade-training-output/model.tar.gz

# Delete endpoint when done
python deploy_endpoint.py --endpoint-name splade-esci-endpoint --delete
```

## Key Design Decisions

- **Base model**: `naver/splade-cocondenser-ensembledistil` -- strongest open SPLADE checkpoint
- **ANCE mining**: Uses scipy sparse matrices for in-memory approximate nearest neighbor search. No external vector DB required.
- **Best-model selection**: Saves Phase 1 checkpoint and compares against ANCE iterations. ANCE can regress on small datasets due to false negatives from unlabeled corpus entries.
- **Zero-shot baseline**: `evaluate.py` runs the base model (no fine-tuning) before evaluating the fine-tuned model, producing a three-row comparison: BM25 / zero-shot / fine-tuned.
- **Graded relevance**: NFCorpus preserves `raw_score` (0/1/2) for NDCG; ESCI uses 4-level E/S/C/I labels.
- **CloudWatch metrics**: `train.py` logs `{"metric_name": "...", "value": ...}` JSON to stdout for SageMaker metric tracking.

## SageMaker SDK v3

This project uses the SageMaker Python SDK v3 `ModelTrainer` API exclusively (not legacy framework estimators):

```python
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import InputData, Compute, SourceCode, OutputDataConfig
```

Note: SDK v3.4.1 has a known bug where `get_training_code_hash()` crashes when `requirements` is a string. A patch is applied in `sagemaker_launcher.py`. See [issue #5518](https://github.com/aws/sagemaker-python-sdk/issues/5518).
