# Serving TabPFN-3 on SageMaker AI: a 1M-row tabular foundation model on a single A10G

> **TL;DR**
>
> - **TabPFN-3** is a transformer-based tabular foundation model that runs entirely in-context: hand it `X_train`, `y_train`, `X_test` in one forward pass and get predictions back. No training step, no per-dataset model.
> - **One million training rows in ~11 s end-to-end** on a single `ml.g5.xlarge` (A10G, ~$1.41/hr) via async inference + `SUBSAMPLE_SAMPLES=10000`. Predict latency stays at ~3 s from 10k to 1M rows; peak VRAM stays at 0.6 GB.
> - **Beats XGBoost on 5 of 6** sklearn/OpenML datasets out of the box. Hits **ROC-AUC 1.0000** on `creditcardfraud` anomaly detection.
> - **NPZ over JSON** for any payload past 1k rows — 5–9× more cells per byte under the realtime 6 MB body cap.

If you've spent any time on tabular ML, you know the rhythm: pick a problem, engineer features, sweep XGBoost or LightGBM, repeat. The base assumption — that every dataset gets its own bespoke model trained from scratch — has held for over a decade. Prior Labs' [TabPFN-3](https://github.com/PriorLabs/TabPFN) is the first model we've found that genuinely breaks that assumption, and the results are weird enough to warrant a careful look.

TabPFN-3 is a tabular foundation model. It's a transformer pretrained on millions of synthetic tabular tasks and runs **entirely in-context**: you hand it your training rows and your test rows in the same forward pass, and it returns predictions. There is no `.fit()` step in the gradient-descent sense — what TabPFN calls fit is just stashing the support set so the next predict call can attend over it. In other words, a tabular dataset is treated the way an LLM treats a prompt.

That changes the deployment story entirely. There's no training job, no hyperparameter sweep, no model registry per dataset. Each request is `{X_train, y_train, X_test} → y_pred`, and the same checkpoint serves every customer's tabular problem.

This post walks through what we learned self-hosting TabPFN-3 on Amazon SageMaker AI. We'll cover how we packaged the container and the inference contract we landed on, then dig into the numbers from a thorough probe of row capacity, ensemble subsampling, and GPU-vs-CPU economics.

## Does it actually work?

Before getting into capacity, the obvious question: is the model any good? We ran it on the standard sklearn / OpenML benchmark grid (TabPFN capped at 8 k training rows for fairness, same train/test split as XGBoost):

| Dataset | Task | TabPFN | XGBoost | Winner |
|---|---|---|---|---|
| breast_cancer | clf | acc=0.972, f1=0.978 | acc=0.951, f1=0.961 | **TabPFN** |
| adult_income | clf | acc=0.851, f1=0.664 | acc=0.861, f1=0.694 | XGBoost |
| credit_g | clf | acc=0.790, f1=0.860 | acc=0.758, f1=0.840 | **TabPFN** |
| california_housing | reg | RMSE=0.41, R²=0.87 | RMSE=0.49, R²=0.82 | **TabPFN** |
| diabetes | reg | RMSE=53.7, R²=0.49 | RMSE=63.0, R²=0.30 | **TabPFN** |
| bike_sharing | reg | RMSE=37.4, R²=0.96 | RMSE=43.5, R²=0.94 | **TabPFN** |

Five out of six, no tuning, no feature engineering. The single most striking result is on `creditcardfraud` (OpenML 1597) — a heavily imbalanced anomaly-detection problem where TabPFN-3 hit **ROC-AUC 1.0000 / PR-AUC 1.0000**, versus XGBoost at 0.9996 / 0.8929 and IsolationForest at 0.989 / 0.108. Same out-of-the-box model, same checkpoint, no special-case handling for class imbalance. That's the result that convinced us TabPFN was worth a real deployment story.

## SageMaker AI inference: realtime vs async

Because every TabPFN call ships its training data over the wire, payload size and timeout are first-class design concerns. SageMaker AI gives us two endpoint modes with very different contracts:

| | Real-time | Asynchronous |
|---|---|---|
| API | `InvokeEndpoint` | `InvokeEndpointAsync` |
| Payload | inline body, **6 MB cap** | S3 URI, **1 GB cap** |
| Timeout | **60 s** | **60 min** |
| Latency overhead | ~1–3 s | ~5–10 s (S3 round-trip) |
| Scale-to-zero | not supported | `MinCapacity=0` allowed |

The two modes are mutually exclusive on a given endpoint — `AsyncInferenceConfig` and the realtime variant cannot coexist. A configured-async endpoint cannot serve a sync `InvokeEndpoint` call; the SageMaker frontend rejects with a validation error before the container is ever consulted. If you want "small payload = sync, large payload = async", you deploy two endpoints sharing the same image and the same `model.tar.gz`. We do exactly that in the demo.

For TabPFN, the realtime path covers everything up to roughly 15k training rows at 32 features (NPZ-encoded). Past that, the wire size or the 60-second timeout binds, and async takes over.

## Packaging the container

We extend the AWS Deep Learning Container `pytorch-inference:2.6.0-py312`, with the GPU and CPU variants selected at build time via a Docker build arg:

```dockerfile
ARG DEVICE=gpu
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker AS gpu
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker AS cpu
FROM ${DEVICE}

RUN pip install --no-cache-dir 'tabpfn==8.0.2' pandas scikit-learn pyarrow
```

A single `container/build_and_push.sh tabpfn3-sagemaker us-east-1 gpu` produces a hardened image with `tabpfn` baked in. We pin the version because checkpoint-loading semantics shift across minor releases. Heavy dependencies belong in the image, not in `source_dir/requirements.txt` — the latter is pip-installed on every cold start.

### Why ship weights via S3?

`tabpfn>=8.0` defaults to V3 and will fetch the official `v3_default` checkpoints from HuggingFace on first `.fit()` if it doesn't find them in the local cache. So if you only need the latest official weights and your endpoint has internet egress, you can skip the `model.tar.gz` step entirely — deploy the image and let TabPFN pull weights on first request. We didn't benchmark this path; for the numbers below we always pre-staged weights in S3.

We pre-stage anyway because it unlocks two things "let the SDK download" can't:

1. **Custom or fine-tuned weights.** Prior Labs publishes a fine-tuning recipe; the resulting `.ckpt` is a drop-in replacement for `tabpfn-v3-classifier-v3_default.ckpt`. Re-package as `model.tar.gz` containing `tabpfn_cache/<your-ckpt>.ckpt`, point the handler at it, and the same image serves your custom model.
2. **Air-gapped endpoints.** SageMaker endpoints in private VPCs without internet egress can't reach HuggingFace. Pre-staging weights in S3 is the only path.

The handler loads checkpoints with an explicit `model_path=`, so the library can never silently fall back to V2.5 or attempt a HuggingFace download:

```python
clf_ckpt = Path(model_dir) / "tabpfn_cache" / "tabpfn-v3-classifier-v3_default.ckpt"
classifier = TabPFNClassifier(model_path=str(clf_ckpt), device=device)
```

The handler ships via the SageMaker SDK's `entry_point="inference.py"` + `source_dir="src/"` mechanism — the SDK repacks `model.tar.gz` to embed `code/inference.py`, which the inference toolkit imports at runtime.

### Two things that surprised us

A pair of integration details cost us a deploy cycle each, and neither is obvious from a casual reading of the docs.

The first is that the SageMaker PyTorch inference toolkit only adds `/opt/ml/model/code/` to `PYTHONPATH` at serve time. If you `COPY code/ /opt/ml/code/` in your Dockerfile (the documented *training* pattern), the file is on disk but the toolkit can't import it, and you end up running `default_pytorch_inference_handler.default_model_fn` — which expects a `.pth` file and 500s every healthcheck. The fix is to ship the handler via `entry_point` + `source_dir` so the SDK packs it inside `model.tar.gz`.

The second: TabPFN's `settings` is a pydantic-settings instance, constructed at *import time*. Setting `os.environ["TABPFN_MODEL_CACHE_DIR"]` inside `model_fn` is too late — the library has already snapshotted the environment. We set the env var at module top, before `from tabpfn import …`, and load works deterministically. Both gotchas plus six others are catalogued in [`RESULTS.md → Gotchas`](RESULTS.md#gotchas-each-cost-a-deploy-cycle) for the curious.

## The inference contract

Every request carries the full support set:

```jsonc
{
  "task": "classification",            // or "regression"
  "X_train": [[...]] | {"col": [...]}, // row-list (numeric) or column-dict (mixed)
  "y_train": [...],
  "X_test":  [[...]] | {"col": [...]},
  "feature_names": [...],              // optional
  "return_probabilities": false,       // classification only
  "ignore_pretraining_limits": false,  // opt-in past V3 caps
  "inference_config": null,            // e.g. {"SUBSAMPLE_SAMPLES": 10000}
  "n_estimators": 8,                   // ensemble size; lower = faster
  "softmax_temperature": 0.9           // classifier-only probability scaling
}
```

The handler accepts four wire encodings, picked via `Content-Type`. The right choice depends on row count and data type:

| Encoding | Content-Type | Bytes/cell | When to use |
|---|---|---|---|
| Plain JSON list-of-lists | `application/json` | ~16 | <1 k rows, pure numeric, debuggability |
| Column-dict JSON | `application/json` | ~16 | Mixed numeric+categorical of any size |
| Gzipped JSON | `application/x-json-gzip` | ~3–6 | Mid-size payloads, no numpy on client |
| **NPZ** | `application/x-npz` | **~2** | **>1 k rows, pure numeric — recommended** |
| Mixed-NPZ (object 2-D) | `application/x-npz` | ~2 | >1 k rows, mixed types |

NPZ gives roughly 5–9× more cells per byte than JSON, which directly translates into 5–9× more rows under the realtime 6 MB body cap. The `src/client_helpers.py` module ships these encoders:

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

One hard rule: **never send mixed numeric+text as a row-list of lists.** `np.asarray([[25, "Private", 50000]])` upcasts everything to a single string dtype, and TabPFN rejects string ndarrays. Use the column-dict or mixed-NPZ paths instead — both preserve per-column dtypes end-to-end.

## VRAM and row capacity

Now the interesting part. We probed how TabPFN-3 scales on `ml.g5.xlarge` (A10G, 24 GB VRAM, ~$1.41/hr), pushing past the documented 50k-row limit from the V2.5 era. All runs below used `ignore_pretraining_limits=False` (the safer default):

| n_train (32 features) | predict s | peak VRAM | accuracy | Status |
|---|---|---|---|---|
| 10,000 | 2.9 | 0.62 GB | 0.985 | ok |
| 30,000 | 12.0 | 0.94 GB | 0.990 | ok |
| 50,000 | 26.0 | 1.33 GB | 1.000 | ok (NPZ wire ceiling on realtime) |
| 75,000 (16 feat) | 45.1 | 2.03 GB | 0.995 | ok |
| 100,000 | timeout | — | — | needs async |

The V3 default checkpoint has **no row validation cap** — the 10k/50k limits in the library source apply only to V2 and V2.5. Across the range we tested, accuracy did not degrade, and VRAM scaled cleanly:

```
peak_gb  ≈  0.55 + 1.4e-5 · N      (32 features, vanilla)
```

Projected OOM on a 24 GB A10G: roughly **1.3 million rows**. In practice we never saw VRAM bind first — the realtime 60-second timeout bound first, well before the GPU did. Past 100k rows on realtime, async is the path.

## The subsample-ensemble trick

TabPFN's prediction is an ensemble: by default, eight estimators each see the support set under different feature shifts and preprocessors, and the predictions are averaged. The library exposes a sharp tool here — `inference_config={"SUBSAMPLE_SAMPLES": k}` — that makes each ensemble member process only a random `k`-row subsample of the support set. Each estimator sees a different subsample, so eight estimators cover roughly `8k` rows of effective coverage at the cost of eight `k`-row forward passes.

This is more elegant than it sounds. The N² attention cost is bounded by the subsample size, not by the input. So `SUBSAMPLE_SAMPLES=10000` makes predict latency essentially **constant in N**:

| n_train (32 feat) | mode | predict s | peak VRAM | accuracy |
|---|---|---|---|---|
| 30 k | vanilla | 12.0 | 1.15 GB | 0.990 |
| 30 k | SUBSAMPLE=10k | **2.9** | **0.83 GB** | 0.990 |
| 50 k | vanilla | 26.0 | 1.54 GB | 1.000 |
| 50 k | SUBSAMPLE=10k | **2.9** | **0.83 GB** | 1.000 |
| 75 k | vanilla | 45.1 | 2.03 GB | 0.995 |
| 75 k | SUBSAMPLE=10k | **2.3** | **0.75 GB** | 0.995 |

Crucially, this is not just "throw away data". At n=45k+SUBSAMPLE=10k, accuracy beat both n=20k vanilla and n=5k vanilla on the same held-out test set. The eight-way ensemble does real work; the subsamples genuinely diversify what each estimator sees.

```python
payload = {
    "task": "classification", "X_train": ..., "y_train": ..., "X_test": ...,
    "ignore_pretraining_limits": True,
    "inference_config": {"SUBSAMPLE_SAMPLES": 10000},
}
```

Combining subsample-ensemble with an async endpoint — which lifts the payload to 1 GB and the timeout to 60 minutes — gives us the headline result. On a single `ml.g5.xlarge`:

| n_train (32 feat) | NPZ wire MB | E2E s | fit s | predict s | peak VRAM | accuracy |
|---|---|---|---|---|---|---|
| 10,000 | 1.2 | 7.4 | 1.06 | 3.41 | 0.62 GB | 0.990 |
| 100,000 | 12.0 | 7.2 | 0.71 | 2.96 | 0.60 GB | 0.995 |
| 500,000 | 59.9 | 7.3 | 1.39 | 2.95 | 0.60 GB | 0.990 |
| **1,000,000** | **119.6** | **10.8** | **2.62** | **2.95** | **0.60 GB** | **0.995** |

`predict_seconds` stays at ~3 s across all four sizes. Peak VRAM stays at 0.6 GB. The only thing that grows with N is the S3 upload + ingestion time. We're moving a million-row training set, predicting on it, and returning answers in eleven seconds end-to-end — on a $1.41/hr instance.

One config tip: TorchServe inside the DLC defaults its `max_request_size` to roughly 6.5 MB, even on async endpoints where SageMaker accepts 1 GB at the frontend. Setting `TS_MAX_REQUEST_SIZE=1073741824` (and `TS_MAX_RESPONSE_SIZE`) in the model environment lifts the container-side limit to match. The deploy script does this automatically on async mode.

Async also unlocks scale-to-zero. With `MinCapacity=0` and a step-scaling policy on the `HasBacklogWithoutCapacity` alarm, idle cost drops to $0/hr, with a cold-start penalty of about ten minutes for the first request — perfect for sporadic traffic, not great for interactive demos.

## GPU vs CPU economics

We tested whether TabPFN-3 can run on CPU (it can — `model_fn` checks `torch.cuda.is_available()` and loads accordingly), and what it costs. The CPU container also sets `TABPFN_ALLOW_CPU_LARGE_DATASET=true` to disable the library's >1k-sample CPU guard. We ran on `ml.c6i.32xlarge` (Intel Ice Lake, 128 vCPU, 256 GB RAM, ~$5.44/hr):

| n_train (32 feat) | predict s | RSS GB | Status |
|---|---|---|---|
| 1,000 | 5.0 | 1.7 | ok |
| 10,000 | **37.4** | 1.8 | ok |
| 13,000 | 52.9 | 4.2 | close to 60 s cap |
| 14,000 | 57.7 | 4.2 | ceiling |
| 15,000+ | timeout | — | exceeds realtime 60 s cap |

OOM is a non-issue — RSS topped out at 4.2 GB on a 256 GB instance, so a `c6i.4xlarge` (32 GB, ~$0.85/hr) would handle the same workload because RAM never binds. The realtime 60-second timeout binds first.

At N=10k, the comparison is unflattering:

- **CPU is ~12× slower** than GPU (37.4 s vs 2.9 s)
- **CPU is ~49× more expensive per inference** ($0.058 vs $0.0012)

There is no sweet spot in the range we tested. GPU wins on every dimension at every N. The honest takeaway: **use GPU**. CPU serving is a real option, but only when you have a hard reason — a no-GPU compliance environment, or massively parallel batch jobs where CPU concurrency outweighs GPU per-request cost.

## Try it yourself

Total spend across all the live verification work in this repo — the P1 probes, the async sweep to 1M rows, the CPU comparison — was about **$10 of g5/c6i hours across roughly ten endpoint deployments**. TabPFN's deployment surface really is small. The four-line quickstart below provisions a g5.xlarge realtime endpoint and runs the breast-cancer smoke test:

```bash
export AWS_DEFAULT_REGION=us-east-1
ACCT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=sagemaker-${AWS_DEFAULT_REGION}-${ACCT}
ECR_URI=${ACCT}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/tabpfn3-sagemaker:latest

container/build_and_push.sh tabpfn3-sagemaker $AWS_DEFAULT_REGION gpu
python scripts/download_and_package_weights.py --bucket "$BUCKET" --prefix tabpfn3
python runners/01_deploy_endpoint.py \
    --image-uri "$ECR_URI" \
    --model-data "s3://$BUCKET/tabpfn3/model.tar.gz" \
    --instance-type ml.g5.xlarge
```

Add `--mode async --async-output-bucket "$BUCKET"` for the 1M-row-capable variant, or `--scale-to-zero` to register autoscaling with `MinCapacity=0`.

What we haven't measured yet: a real >1M-row benchmark on a public dataset like Higgs or Criteo, the fine-tuning loop in production, and a head-to-head sweep across A10G / L40S / H100. All on the docket for a follow-up.

For the full forensic record — every gotcha, every encoding ceiling, every cold-start timing — see [`RESULTS.md`](RESULTS.md). For the open backlog, [`NEXT-STEPS.md`](NEXT-STEPS.md).

The most compelling thing we learned probing TabPFN-3 wasn't any single number. It was the shift in mental model: a tabular dataset is a prompt, the model is the same checkpoint for everyone, and the deployment surface area collapses to "one endpoint, many problems." On that frame, eleven seconds for a million rows on a $1.41/hr instance is just the punchline.
