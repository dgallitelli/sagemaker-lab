# TabPFN-3 on Amazon SageMaker AI

End-to-end demo for self-hosting [TabPFN-3](https://github.com/PriorLabs/TabPFN)
(Prior Labs' tabular foundation model) on a SageMaker real-time endpoint, with
a benchmark harness, a payload-encoding helper library, and findings from
exhaustive probing of row caps, mixed-data support, ensemble inference, and
GPU-vs-CPU economics.

> Built and validated on 2026-05-14 against `tabpfn==8.0.2` and the
> `pytorch-inference:2.6.0-{gpu,cpu}-py312-ubuntu22.04-sagemaker` DLCs in
> `us-east-1`.

## Approach

- **Custom inference image** extending the AWS PyTorch DLC. Two flavours:
  - `pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker` for GPU
  - `pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker` for CPU
  Both pin `tabpfn==8.0.2`. The Dockerfile uses `--build-arg DEVICE={gpu,cpu}`
  to select the base.
- **Weights pre-staged on S3** as `model.tar.gz` containing `tabpfn_cache/`
  with V3 default checkpoints. `inference.py` loads them with an explicit
  `model_path=`, so the library can never silently fall back to V2.5 or
  attempt a HuggingFace download. The container itself does not need
  `TABPFN_TOKEN`.
- **Inference handler** lives in `src/inference.py` and is shipped via the
  SageMaker SDK's `entry_point` + `source_dir` mechanism. The SDK repacks
  `model.tar.gz` to embed `code/inference.py`, which the inference toolkit
  imports from `/opt/ml/model/code/` at runtime. Baking the script into the
  image at `/opt/ml/code/` does **not** work — the inference toolkit only
  adds `/opt/ml/model/code/` to PYTHONPATH.
- **Real-time endpoint, in-context inference**. Clients send `X_train`,
  `y_train`, `X_test` on every request. TabPFN's "fit" just stores the support
  set; the forward pass on `predict` does both at once.

### Why ship weights via S3 instead of letting the SDK download them?

`tabpfn>=8.0` defaults to V3 (`ModelVersion.V3` in the library settings) and
will fetch the official `v3_default` checkpoints from HuggingFace on first
`.fit()` if they're not in the local cache. So if all you want is the latest
official weights, **you can skip `model.tar.gz` entirely** — just deploy the
image and let the library download at first request, paying a ~30 s one-time
cold-start cost (and accepting that the SageMaker container needs egress to
HuggingFace plus a `TABPFN_TOKEN` for the gated repo).

We pre-stage `model.tar.gz` in S3 anyway because doing so unlocks two things
that "let the SDK download" can't:

1. **Custom / fine-tuned weights.** If you fine-tune TabPFN-3 on your own data
   (Prior Labs publishes a fine-tuning recipe), the resulting `.ckpt` is a
   drop-in replacement for `tabpfn-v3-classifier-v3_default.ckpt`. Re-package
   as `model.tar.gz` containing `tabpfn_cache/<your-ckpt>.ckpt`, point
   `inference.py` at it via the `V3_CLASSIFIER_CKPT` / `V3_REGRESSOR_CKPT`
   constants (or extend the handler to accept a request-time override), and
   the same image serves your custom model.
2. **Air-gapped / no-egress endpoints.** SageMaker endpoints in private VPCs
   without internet egress can't reach HuggingFace. Pre-staging weights in S3
   (and reading them out of `/opt/ml/model/`) is the only path that works.

The handler refuses to start if the V3 default `.ckpt` files are missing from
`tabpfn_cache/`, so failures are loud rather than silent fall-throughs.

## Layout

```
tabpfn3-sagemaker-experiments/
├── README.md
├── NEXT-STEPS.md             # roadmap of P2/P3 follow-ups
├── LICENSE
├── requirements.txt
├── container/
│   ├── Dockerfile             # multi-stage GPU/CPU base via --build-arg DEVICE
│   └── build_and_push.sh      # finch build + push to ECR
├── src/
│   ├── inference.py           # SageMaker contract, V3 explicit model_path
│   └── client_helpers.py      # encode_json / encode_npz / encode_npz_mixed
├── scripts/
│   ├── download_and_package_weights.py   # local cache → model.tar.gz → S3
│   ├── request_quota.py                  # check / request endpoint quotas
│   └── probe_row_capacity.py             # P1.2 driver
└── notebooks/
    ├── 01_deploy_endpoint.py  # deploy + smoke test
    ├── 02_benchmark.py        # cross-instance + 4-section benchmark
    └── 03_cleanup.py          # delete endpoints, optional ECR / S3 cleanup
```

## Inference contract

```jsonc
{
  "task": "classification",            // or "regression"
  "X_train": [[...]] or {"col_name": [...]},   // see "Choosing an encoding"
  "y_train": [...],
  "X_test":  [[...]] or {"col_name": [...]},
  "feature_names": [...],              // optional
  "return_probabilities": false,       // classification only
  "ignore_pretraining_limits": false,  // opt-in past V3 caps
  "inference_config": null             // e.g. {"SUBSAMPLE_SAMPLES": 10000}
}
```

Response:

```jsonc
{
  "predictions": [...],
  "probabilities": [[...]],            // null unless requested
  "metadata": {
    "task": "classification",
    "n_train": 455, "n_test": 114, "n_features": 30,
    "fit_seconds": 0.42,
    "predict_seconds": 1.18,
    "device": "cuda",
    "gpu_memory": {"allocated_gb": ..., "reserved_gb": ..., "peak_gb": ...},
    "cpu_memory": {"peak_rss_gb": ...},
    "model_checkpoint": ".../tabpfn-v3-classifier-v3_default.ckpt",
    "ignore_pretraining_limits": false
  }
}
```

## Choosing an encoding

The handler accepts four wire encodings; the right choice depends on row
count, data type, and how much compute you can spend on the client.

| Encoding | Content-Type | When to use | Approx wire size |
|---|---|---|---|
| Plain JSON list-of-lists | `application/json` | <1k rows, pure numeric, debuggability | ~16 bytes/cell |
| Column-dict JSON | `application/json` (X is a dict of `{col: [...]}`) | Mixed numeric+categorical of any size | ~16 bytes/cell |
| Gzipped JSON | `application/x-json-gzip` | Mid-size payloads, no numpy on client | ~3-6 bytes/cell |
| **NPZ** (`np.savez_compressed`) | `application/x-npz` | **>1k rows, pure numeric — recommended default** | ~2 bytes/cell |
| **Mixed-data NPZ** (object 2-D) | `application/x-npz` | >1k rows, mixed types — best of both | ~2 bytes/cell |

**Hard rule:** never send mixed numeric+text as a row-list of lists.
`np.asarray([[25, 'Private', 50000]])` upcasts everything to a single string
dtype, and TabPFN rejects string ndarrays with a 500. Use the column-dict or
mixed-NPZ paths instead. The `src/client_helpers.py` module ships these:

```python
import boto3
from src.client_helpers import encode_npz, encode_npz_mixed, invoke

rt = boto3.client("sagemaker-runtime", region_name="us-east-1")

# Pure numeric
body, ct, _ = encode_npz(X_train, y_train, X_test, task="classification")
out = invoke(rt, ENDPOINT_NAME, body=body, content_type=ct)

# Mixed (DataFrame goes straight in; categoricals stay as strings)
body, ct, _ = encode_npz_mixed(df_train, y_train, df_test, task="classification")
out = invoke(rt, ENDPOINT_NAME, body=body, content_type=ct)
```

### SageMaker realtime payload caps

The realtime invocation has a hard ~6 MB request body limit. NPZ encoding
gives you ~5-9× more cells per byte than JSON, so the practical NPZ ceiling
on a single realtime request is roughly:

| Features | Max n_train (NPZ, ~6 MB cap) |
|---|---|
| 8 | ~50,000 |
| 16 | ~30,000 |
| 32 | ~15,000 (well under under realtime 60s timeout) |
| 100 | ~5,000 |

Past these limits the path is async inference (1 GB payload, 60 min timeout) —
a planned P2 follow-up.

## Quickstart

```bash
# 0. Set your AWS profile + region (used by every step below)
export AWS_PROFILE=<your-profile>
export AWS_DEFAULT_REGION=us-east-1
BUCKET=$(aws sts get-caller-identity --query Account --output text \
        | xargs -I{} echo "sagemaker-${AWS_DEFAULT_REGION}-{}")

# 1. (Optional) check / request endpoint quotas
python scripts/request_quota.py --region $AWS_DEFAULT_REGION --auto-request

# 2. Download weights locally (uses TABPFN_TOKEN from env) and upload to S3.
#    Use a temp venv so the global Python stays clean.
python -m venv /tmp/tabpfn-venv
source /tmp/tabpfn-venv/bin/activate
pip install tabpfn boto3
python scripts/download_and_package_weights.py \
    --bucket "$BUCKET" --prefix tabpfn3 --region $AWS_DEFAULT_REGION
deactivate

# 3. Build and push the inference container (Finch). Pick GPU or CPU base.
container/build_and_push.sh tabpfn3-sagemaker $AWS_DEFAULT_REGION gpu
# (or)
container/build_and_push.sh tabpfn3-sagemaker $AWS_DEFAULT_REGION cpu

# 4. Deploy on g5.xlarge and run a sklearn smoke test.
ACCT=$(aws sts get-caller-identity --query Account --output text)
ECR_URI=${ACCT}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/tabpfn3-sagemaker:latest
python notebooks/01_deploy_endpoint.py \
    --image-uri "$ECR_URI" \
    --model-data "s3://$BUCKET/tabpfn3/model.tar.gz" \
    --instance-type ml.g5.xlarge \
    --region $AWS_DEFAULT_REGION

# 5. Cross-instance benchmark (auto-deletes endpoints unless --keep-endpoints).
python notebooks/02_benchmark.py \
    --image-uri "$ECR_URI" \
    --model-data "s3://$BUCKET/tabpfn3/model.tar.gz" \
    --instance-types ml.g5.xlarge,ml.g6e.xlarge

# 6. Tear down anything left over.
python notebooks/03_cleanup.py --region $AWS_DEFAULT_REGION
```

## Findings (P1 probe results, 2026-05-14)

Live-tested against `tabpfn==8.0.2` on `ml.g5.xlarge` (A10G, 24 GB) and
`ml.c6i.32xlarge` (128 vCPU, 256 GB).

### Real-world datasets — TabPFN beats XGBoost on 5 of 6

| Dataset | Task | TabPFN | XGBoost | Winner |
|---|---|---|---|---|
| breast_cancer | clf | acc=0.972, f1=0.978 | acc=0.951, f1=0.961 | **TabPFN** |
| adult_income | clf | acc=0.851, f1=0.664 | acc=0.861, f1=0.694 | XGBoost |
| credit_g | clf | acc=0.790, f1=0.860 | acc=0.758, f1=0.840 | **TabPFN** |
| california_housing | reg | RMSE=0.41, R²=0.87 | RMSE=0.49, R²=0.82 | **TabPFN** |
| diabetes | reg | RMSE=53.7, R²=0.49 | RMSE=63.0, R²=0.30 | **TabPFN** |
| bike_sharing | reg | RMSE=37.4, R²=0.96 | RMSE=43.5, R²=0.94 | **TabPFN** |

On creditcardfraud (anomaly): TabPFN ROC-AUC=1.00, PR-AUC=1.00 vs XGBoost
0.9996/0.89, IsolationForest 0.989/0.11.

### Row capacity past the documented 50k cap

V3 default checkpoint has **no row validation cap** (the 10k/50k limits in
the library are V2/V2.5 baggage). VRAM scales linearly:
`peak_gb ≈ 0.55 + 1.4e-5·N`, projecting OOM on a 24 GB A10G around N≈1.3M
rows. The binding constraint is the **SageMaker realtime 60s response
timeout**, not VRAM:

| n_train (32 features) | predict_s (g5.xlarge) | peak GB | Status |
|---|---|---|---|
| 10,000 | 2.9 | 0.62 | ok |
| 30,000 | 12.0 | 0.94 | ok |
| 50,000 | 26.0 | 1.33 | ok (NPZ wire ceiling) |
| 75,000 | 45.1 | 2.03 | ok at 16 features |
| 100,000+ | timeout | — | needs async inference |

### Subsample-ensemble — the LinkedIn 1M-row claim is real

`inference_config={"SUBSAMPLE_SAMPLES": 10000}` makes each of TabPFN's 8
ensemble members process only 10,000 rows, then averages predictions. The
result: predict latency becomes effectively **constant in N**, with no
accuracy hit.

| n_train (32 feat) | mode | predict s | peak GB | accuracy |
|---|---|---|---|---|
| 30k | vanilla | 12.0 | 1.15 | 0.990 |
| 30k | SUBSAMPLE=10k | **2.9** | **0.83** | 0.990 |
| 50k | vanilla | 26.0 | 1.54 | 1.000 |
| 50k | SUBSAMPLE=10k | **2.9** | **0.83** | 1.000 |
| 75k | vanilla | 45.1 | 2.03 | 0.995 |
| 75k | SUBSAMPLE=10k | **2.3** | **0.75** | 0.995 |

This is more than "use less data": at n=45k+SUBSAMPLE=10k, accuracy beat both
n=20k vanilla and n=5k vanilla on a held-out test set. Each estimator sees a
different 10k subsample, so 8 estimators see ~80k rows of effective coverage.

**Recommended default for >15k rows: enable subsampling.**

```python
payload = {
    "task": "classification", "X_train": ..., "y_train": ..., "X_test": ...,
    "ignore_pretraining_limits": True,
    "inference_config": {"SUBSAMPLE_SAMPLES": 10000},
}
```

Caveat: `inference_config` keys are validated server-side with Pydantic
`extra="forbid"`. Typos return 500.

### Mixed numeric+categorical data — works via column-dict or mixed-NPZ

TabPFN-3 handles raw categorical strings internally if the wire format
preserves per-column dtypes. The `_coerce_x` helper in `inference.py`
detects dict payloads and converts them to a pandas DataFrame; NPZ object
2-D arrays are passed through.

On `adult_income` (14 features, mix of numbers and strings), 1k train / 200
test:

| Encoding | Wire bytes | Accuracy | Status |
|---|---|---|---|
| Column-dict JSON | 178,301 | 0.795 | ok |
| Mixed-NPZ | 15,721 | 0.795 | ok (10× smaller) |
| Row-list JSON with strings | 165,863 | n/a | **500** (numpy upcasts to string) |

### Async inference — verified to 1M rows on a single g5.xlarge

Deploying with `--mode async` swaps the realtime 6 MB / 60 s caps for an
S3-staged payload up to 1 GB and a 60 min response budget. Combined with
NPZ encoding and `SUBSAMPLE_SAMPLES=10000`, a single `ml.g5.xlarge` (24 GB
A10G) handles a million-row training set in under 11 seconds end-to-end:

| n_train (32 feat) | NPZ wire MB | raw MB | poll s (E2E) | fit s | predict s | peak GB | accuracy |
|---|---|---|---|---|---|---|---|
| 10,000 | 1.2 | 1.3 | 7.4 | 1.06 | 3.41 | 0.62 | 0.990 |
| 100,000 | 12.0 | 12.8 | 7.2 | 0.71 | 2.96 | 0.60 | 0.995 |
| 500,000 | 59.9 | 64.0 | 7.3 | 1.39 | 2.95 | 0.60 | 0.990 |
| **1,000,000** | **119.6** | **128.0** | **10.8** | **2.62** | **2.95** | **0.60** | **0.995** |

Predict latency is **constant at ~3 s** regardless of N (subsample is
doing the work — each of TabPFN's 8 estimators sees a different random
10k-row subsample). Peak VRAM stays at 0.6 GB across all sizes — N²
attention is bounded by `SUBSAMPLE_SAMPLES`, not the payload.

**Gotcha**: TorchServe inside the DLC defaults to a ~6.5 MB request cap.
Async at the SageMaker frontend supports 1 GB, but the container-side
limit bites first on payloads above ~6 MB. The deploy script sets
`TS_MAX_REQUEST_SIZE=1073741824` (and `_RESPONSE_SIZE`) on async
endpoints to lift this. If you build your own deploy, set both env vars.

### CPU serving — works but uneconomical

Same handler runs on CPU instances with a CPU DLC base. The `model_fn` auto-
detects `torch.cuda.is_available()` and loads on CPU; `TABPFN_ALLOW_CPU_LARGE_DATASET=true` is set to disable the library's >1k-sample CPU guard.

Tested on `ml.c6i.32xlarge` (128 vCPU, 256 GB):

| n_train (32 feat) | predict s | RSS GB | Status |
|---|---|---|---|
| 1,000 | 5.0 | 1.7 | ok |
| 10,000 | 37.4 | 1.8 | ok |
| 13,000 | 52.9 | 4.2 | ok (close to 60s cap) |
| 14,000 | 57.7 | 4.2 | ceiling |
| 15,000+ | timeout | — | exceeds realtime 60s cap |

OOM is a non-issue (peak RSS 4.2 GB on a 256 GB box). The CPU is **~12×
slower** than g5.xlarge GPU and **~49× more expensive per inference** at
N=10k ($0.058 vs $0.0012). A `c6i.4xlarge` (32 GB, ~$0.85/hr) handles the
same workload because RAM is never the bottleneck. **GPU still wins on
every dimension.**

## Cost note

Real-time endpoints bill per-instance-hour while `InService`, even when idle.
`notebooks/03_cleanup.py` deletes anything matching the `tabpfn3-` prefix.

## Roadmap

See [NEXT-STEPS.md](NEXT-STEPS.md) for the deferred priority list (async
endpoint variant, cross-instance benchmark wiring, polish, fine-tuning demo).

## License

MIT for the demo code (see [LICENSE](LICENSE)). TabPFN model weights are
licensed separately by Prior Labs — see https://github.com/PriorLabs/TabPFN.
