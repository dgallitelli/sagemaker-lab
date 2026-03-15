# SPLADE Fine-Tuning on SageMaker

Fine-tune a [SPLADE](https://arxiv.org/abs/2109.10086) sparse embedding model with ANCE hard negative mining on Amazon SageMaker, using the [FiQA](https://huggingface.co/datasets/mteb/fiqa) retrieval benchmark (with Amazon ESCI as the intended target when accessible).

## Results

Validated end-to-end locally on FiQA (financial QA retrieval, 57K corpus). Local mode uses a 600-doc subset for fast iteration; the SageMaker job runs against the full corpus.

**FiQA** (`mteb/fiqa`) — financial question answering, 57,638-document corpus:

| Metric | BM25 | SPLADE | Delta |
|--------|:---:|:---:|:---:|
| NDCG@10 | 0.159 | 0.280 | +76% |
| Recall@100 | 0.359 | 0.585 | +63% |
| MRR@10 | 0.199 | 0.341 | +72% |

**NFCorpus** (`mteb/nfcorpus`) — biomedical literature retrieval, 3,633-document corpus:

| Metric | BM25 | SPLADE | Delta |
|--------|:---:|:---:|:---:|
| NDCG@10 | 0.266 | 0.410 | +54% |
| Recall@100 | 0.210 | 0.729 | +247% |
| MRR@10 | 0.467 | 0.750 | +61% |

Evaluated after 3 training phases (easy negatives → ANCE iter 1 → ANCE iter 2) on Apple Silicon MPS. The SageMaker job on `ml.g5.12xlarge` with full dataset and more epochs is expected to improve further.

## Architecture

```
prepare_dataset.py          # Download dataset, build corpus, compute BM25 baseline
splade-ecommerce/
  train.py                  # SageMaker entry point — 3-phase training with ANCE
  ance_miner.py             # ANCE hard negative mining via Qdrant in-memory
  evaluate.py               # NDCG@10 / Recall@100 / MRR@10 evaluation
  sagemaker_launcher.py     # Launch SageMaker job or run locally with --local
  config.yaml               # Hyperparameters
  requirements.txt          # Training container dependencies
deploy_endpoint.py          # Deploy trained model to SageMaker TEI endpoint
```

### Training Pipeline

```
Phase 1: Easy negatives (E-labeled positives, in-batch negatives)
    ↓
ANCE Iter 1: Encode corpus → mine hard negatives → retrain
    ↓
ANCE Iter 2: Re-encode with updated weights → mine harder negatives → retrain
    ↓
Final evaluation vs BM25 baseline
```

Key design choices:
- **Base model**: `naver/splade-cocondenser-ensembledistil` — strongest open SPLADE checkpoint
- **Qdrant in-memory**: `QdrantClient(":memory:")` — no external services, auto-cleans with process
- **FLOPS regularization**: `SpladeLoss` with `lambda=4e-4` controls sparse activation count
- **SageMaker SDK v3**: `ModelTrainer` API with `SourceCode`, `Compute`, `OutputDataConfig`

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install sentence-transformers datasets bm25s scipy numpy tqdm pyyaml boto3 accelerate
```

## Usage

### 1. Prepare data and BM25 baseline

```bash
# Downloads FiQA from HuggingFace (tries Amazon ESCI first, falls back to mteb/fiqa)
python prepare_dataset.py --output-dir data/
```

Output: `data/train.jsonl`, `data/test.jsonl`, `data/corpus.jsonl`, `data/bm25_baseline_results.json`

### 2. Validate locally (no AWS needed)

```bash
python splade-ecommerce/sagemaker_launcher.py --local --data-dir data/
```

Uses truncated dataset (500 train pairs, 600-doc corpus) for fast CPU/MPS iteration. Runs in ~8 minutes on Apple Silicon.

### 3. Submit SageMaker training job

```bash
python splade-ecommerce/sagemaker_launcher.py \
  --s3-bucket <your-bucket> \
  --data-prefix splade-fiqa/data
```

Uploads data to S3, submits `ml.g5.12xlarge` spot instance job (~$2–4 USD estimated).

### 4. Deploy endpoint

```bash
python deploy_endpoint.py \
  --model-artifact s3://<bucket>/splade-training-output/model.tar.gz
```

### 5. Cleanup

```bash
python deploy_endpoint.py --endpoint-name splade-esci-endpoint --delete
```

## Dataset

The pipeline tries dataset sources in order:

1. **Amazon ESCI** (`tasksource/amazon-esci`) — e-commerce product search with graded relevance (E/S/C/I labels). Requires HuggingFace auth or public availability.
2. **FiQA** (`mteb/fiqa`) — financial question answering, 57K corpus, standard BEIR benchmark. Always publicly available.
3. **Synthetic** — structured e-commerce pairs (fast iteration only, no meaningful retrieval signal).

## Hyperparameters

Key settings in `splade-ecommerce/config.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `base` | `naver/splade-cocondenser-ensembledistil` | Pre-trained SPLADE checkpoint |
| `batch_size` | 32 | Training batch size (8 in local mode) |
| `learning_rate` | 2e-5 | AdamW learning rate |
| `flops_weight` | 4e-4 | SPLADE FLOPS regularization weight |
| `ance_iterations` | 2 | Hard negative mining iterations |
| `top_k_mining` | 50 | ANN candidates retrieved per query |

## SageMaker SDK v3

Uses the v3 `ModelTrainer` API exclusively:

```python
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import InputData, Compute, SourceCode, OutputDataConfig

trainer = ModelTrainer(
    role=role,
    training_image=image_uri,
    source_code=SourceCode(source_dir="./splade-ecommerce", entry_script="train.py"),
    compute=Compute(instance_type="ml.g5.12xlarge", instance_count=1),
    output_data_config=OutputDataConfig(s3_output_path=s3_uri),
    sagemaker_session=session,
)
trainer.train(input_data_config=[...], wait=True, logs=True)
```

> SDK v3.4.1 has a known bug where `get_training_code_hash()` crashes when `requirements` is a string. A patch is applied in `sagemaker_launcher.py` before job submission. See [issue #5518](https://github.com/aws/sagemaker-python-sdk/issues/5518).
