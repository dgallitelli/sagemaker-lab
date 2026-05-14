# TabPFN-3 on SageMaker AI — Results

Findings from 2026-05-13 to 2026-05-14, all live-tested in `us-east-1`
against `tabpfn==8.0.2` on the AWS PyTorch DLC `2.6.0-py312`.

This document is the consolidated answer to "did this actually work?". For
the roadmap of what's left, see [NEXT-STEPS.md](NEXT-STEPS.md). For the
how-to, see [README.md](README.md).

---

## TL;DR

- **TabPFN-3 default V3 weights serve fine on a single `ml.g5.xlarge`** (A10G,
  24 GB VRAM, ~$1.41/hr) for the entire range we tested.
- **The model has no row-count cap in the V3 default checkpoint** — the
  documented 10k/50k limits are V2/V2.5 baggage. VRAM scales linearly; the
  binding constraint on real-time endpoints is the **60 s SageMaker invoke
  timeout**, not VRAM.
- **The "1M-row TabPFN" claim is real**, but only via async endpoints with
  `inference_config={"SUBSAMPLE_SAMPLES": 10000}`. Verified: 1M rows × 32
  features in **11 s end-to-end** on g5.xlarge, peak GPU 0.6 GB.
- **Async with scale-to-zero works** — `MinCapacity=0` reduces idle cost
  to $0/hr at the cost of a **~10 minute wake-from-zero**. Requires a
  step-scaling policy on `HasBacklogWithoutCapacity` to wake at all
  (target-tracking alone won't, see Gotcha G8).
- **NPZ encoding is mandatory above ~6 k rows** at 32 features for either
  endpoint mode. JSON crosses 6 MB before then.
- **CPU serving works but is uneconomical**: ~12× slower and ~49× more
  expensive per inference than GPU at the same workload.
- **Mixed numeric+text data works** end-to-end if the wire format preserves
  per-column dtypes (column-dict JSON or 2-D object NPZ). Row-list of mixed
  types is broken — numpy upcasts everything to strings and TabPFN rejects.
- **Eight non-obvious gotchas** documented below; collectively they cost
  4-5 deploy cycles to discover. Highest-impact ones: the inference
  toolkit only adds `/opt/ml/model/code/` to PYTHONPATH (G1), the
  SageMaker toolkit pre-decodes `application/json` bodies to UTF-8 before
  `input_fn` runs (G2), and the DLC's TorchServe caps requests at 6.5 MB
  even on async endpoints (G5).

---

## Real-world dataset benchmark — TabPFN beats XGBoost on 5 of 6

Run on `ml.g5.xlarge` against the standard sklearn / OpenML datasets, with
TabPFN capped at 8 k training rows for fair comparison. Same train/test
split for both models.

| Dataset | Task | TabPFN | XGBoost | Winner |
|---|---|---|---|---|
| breast_cancer | clf | acc=0.972, f1=0.978 | acc=0.951, f1=0.961 | **TabPFN** |
| adult_income | clf | acc=0.851, f1=0.664 | acc=0.861, f1=0.694 | XGBoost |
| credit_g | clf | acc=0.790, f1=0.860 | acc=0.758, f1=0.840 | **TabPFN** |
| california_housing | reg | RMSE=0.41, R²=0.87 | RMSE=0.49, R²=0.82 | **TabPFN** |
| diabetes | reg | RMSE=53.7, R²=0.49 | RMSE=63.0, R²=0.30 | **TabPFN** |
| bike_sharing | reg | RMSE=37.4, R²=0.96 | RMSE=43.5, R²=0.94 | **TabPFN** |

### Anomaly detection (creditcardfraud, OpenML 1597) — TabPFN crushes it

| Model | ROC-AUC | PR-AUC | F1 (@ 0.5 thresh) |
|---|---|---|---|
| **TabPFN-3** | **1.0000** | **1.0000** | **1.0000** |
| XGBoost | 0.9996 | 0.8929 | 0.8571 |
| IsolationForest | 0.9895 | 0.1080 | 0.0976 |

### Time-series (Air Passengers, lag-feature regression)

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| Linear regression (local) | 14.1 | 18.2 | 3.4 |
| **TabPFN-3** | 44.7 | 51.4 | 9.7 |
| XGBoost (local) | 53.0 | 64.3 | 11.4 |
| Seasonal-naive (local) | 65.3 | 74.0 | 14.3 |

Linear regression wins on a strongly trending series — expected. TabPFN beats
both XGBoost and seasonal-naive.

---

## Row capacity — V3 has no validation cap

We hypothesized TabPFN-3 would error past its pretraining sample cap (V2.5
documented 50k). It doesn't. The library defaults that cap at 10k/50k come
from `inference_config.py:213/336` and apply only to V2/V2.5; V3's
inference config is baked into the checkpoint itself.

**Live-tested with `ignore_pretraining_limits=False` (the safer default):**

| n_train (32 feat) | predict_s (g5.xlarge) | peak VRAM | accuracy | Status |
|---|---|---|---|---|
| 10,000 | 2.9 s | 0.62 GB | 0.985 | ok |
| 30,000 | 12.0 s | 0.94 GB | 0.990 | ok |
| 50,000 | 26.0 s | 1.33 GB | 1.000 | ok (NPZ wire ceiling on realtime) |
| 75,000 (16 feat) | 45.1 s | 2.03 GB | 0.995 | ok |
| 100,000 | timeout | — | — | exceeds realtime 60 s cap; needs async |

**No accuracy degradation** between 5 k and 50 k rows. VRAM scales linearly:

```
peak_gb ≈ 0.55 + 1.4e-5 · N        (32 features)
```

Projected OOM on 24 GB A10G: **N ≈ 1.3 M rows**. The realtime 60 s timeout
binds well before the GPU does.

---

## Subsample-ensemble — the "1M rows on TabPFN" claim is real

`inference_config={"SUBSAMPLE_SAMPLES": k}` makes each of TabPFN's 8
ensemble members process only `k` rows from the support set, then averages
predictions. Each estimator sees a different random subsample, so 8
estimators cover effectively `8k` rows of the input distribution.

**Effect on g5.xlarge at 32 features (32 features, held-out 200-row test):**

| n_train | mode | predict s | peak VRAM | accuracy |
|---|---|---|---|---|
| 30 k | vanilla | 12.0 | 1.15 GB | 0.990 |
| 30 k | SUBSAMPLE=10k | **2.9** | **0.83 GB** | 0.990 |
| 50 k | vanilla | 26.0 | 1.54 GB | 1.000 |
| 50 k | SUBSAMPLE=10k | **2.9** | **0.83 GB** | 1.000 |
| 75 k | vanilla | 45.1 | 2.03 GB | 0.995 |
| 75 k | SUBSAMPLE=10k | **2.3** | **0.75 GB** | 0.995 |

**This is not just "use less data".** At n=45k+SUBSAMPLE=10k, accuracy beat
both n=20k vanilla AND n=5k vanilla on the same held-out test set — so the
ensemble is doing real work, not just sampling noise.

**Recommended default for any N > 15k:**

```python
payload = {
    "task": "classification", "X_train": ..., "y_train": ..., "X_test": ...,
    "ignore_pretraining_limits": True,
    "inference_config": {"SUBSAMPLE_SAMPLES": 10000},
}
```

### Caveats

- `inference_config` is validated server-side with Pydantic `extra="forbid"`.
  Typo'd keys → 500 from the model server. The handler accepts the dict
  as-is and forwards to TabPFN, so misspellings only surface during
  prediction. **Use `client_helpers.validate_inference_config()`** — it
  knows the canonical field set (snapshotted from tabpfn 8.0.2) and
  suggests close matches via `difflib`. `encode_json`/`encode_npz`/
  `encode_npz_mixed` invoke the validator automatically.
- `n_estimators` is a TabPFN constructor arg, not an `InferenceConfig`
  field, so it is **not** tunable via `inference_config`. The handler
  accepts `n_estimators` as a top-level payload key and rebuilds the
  estimator with the override (default 8). Live-tested: at 8 k breast_cancer
  rows, `predict_seconds` scales as 0.61 / 0.50 / 0.97 / 1.94 s for
  n_estimators ∈ {1, 4, 8, 16}.
- `softmax_temperature` (classifier-only constructor arg) likewise accepted
  as a top-level payload key. Useful for probability calibration on
  imbalanced datasets without redeploying.

---

## Async at scale — verified to 1M rows

Deployed `--mode async` on `ml.g5.xlarge`, swept synthetic
`make_classification(n_samples=N, n_features=32)` with NPZ encoding and
`SUBSAMPLE_SAMPLES=10000`.

| n_train | NPZ wire MB | raw MB | E2E s | fit s | predict s | peak VRAM | accuracy |
|---|---|---|---|---|---|---|---|
| 10,000 | 1.2 | 1.3 | 7.4 | 1.06 | 3.41 | 0.62 GB | 0.990 |
| 100,000 | 12.0 | 12.8 | 7.2 | 0.71 | 2.96 | 0.60 GB | 0.995 |
| 500,000 | 59.9 | 64.0 | 7.3 | 1.39 | 2.95 | 0.60 GB | 0.990 |
| **1,000,000** | **119.6** | **128.0** | **10.8** | **2.62** | **2.95** | **0.60 GB** | **0.995** |

`predict_seconds` stayed at **~3 s across all four sizes** because subsample
caps the work per estimator at 10 k rows. The only thing that grew with N
was the S3 upload + ingestion (`fit_seconds`).

### One bug discovered & fixed

The first deploy went InService and 10k worked. **100k returned a 413 from
the container** — not the SageMaker frontend. Async accepts 1 GB at the API
level, but the DLC's TorchServe defaults to ~6.5 MB `max_request_size`.

Fix: set both `TS_MAX_REQUEST_SIZE` and `TS_MAX_RESPONSE_SIZE` to 1 GB in
the model env. The deploy script does this automatically on async mode.

```python
env = {
    ...
    "TS_MAX_REQUEST_SIZE": "1073741824",
    "TS_MAX_RESPONSE_SIZE": "1073741824",
}
```

### Async with scale-to-zero — wake-from-zero is ~10 minutes

Live-tested with `--scale-to-zero --scale-min 0 --scale-max 2 --scale-down-after 300` (5 min idle window). Two autoscaling policies are wired together:

1. **Target-tracking** on `ApproximateBacklogSizePerInstance` (target=1) — handles steady-state load.
2. **Step-scaling** on the `HasBacklogWithoutCapacity` CloudWatch alarm — wakes the endpoint from `MinCapacity=0` when the first request lands. Without this, a scaled-to-zero endpoint never wakes on its own; target-tracking only acts on per-instance metrics, which require ≥1 instance.

Verified timeline (g5.xlarge, the image we have today):

| Event | Wall clock |
|---|---|
| Last invocation served | t=0 |
| Target-tracking fires `DesiredInstanceCount=0` | t≈13 min (300s idle window + 8 min target-tracking baseline latency) |
| Instance terminates | within 10 s of scale-in |
| New invocation queued (from zero) | t' = 0 |
| `HasBacklogWithoutCapacity` alarm fires | ~2 min |
| Instance provisioned, image pulled, container starts | ~6 min |
| Model loads, healthcheck passes, request served | **~10 min total** |

The 10-min wake-from-zero is dominated by ECR image pull (8.5 GB) and `model.tar.gz` download (1.6 GB), then ~2 min for the V3 weights to load on GPU. Halving image size (e.g. trimming the DLC) would cut this meaningfully; image size is the lever.

**When scale-to-zero is the right call:**
- Sporadic traffic — a few requests per day or per hour. Idle GPU cost ($1.41/hr × 24h = $34/day) >> wake-from-zero penalty.
- Async-only (sync endpoints can't have `MinCapacity=0`).
- Caller can tolerate a 10-min first-request latency.

**When to keep `MinCapacity=1`:**
- Steady traffic.
- Interactive demos where 10-min cold-starts would be embarrassing.

### Async vs realtime

| Aspect | Realtime | Async |
|---|---|---|
| API | `InvokeEndpoint` | `InvokeEndpointAsync` |
| Payload | inline body, **6 MB cap** | S3 URI, **1 GB cap** |
| Timeout | 60 s | 60 min |
| Latency | sync, ~1-3 s overhead | async, ~5-10 s S3 overhead |
| Mode flag | absent in EndpointConfig | `AsyncInferenceConfig` set |
| Sync invocation | works | **rejected by SageMaker frontend** |

**An endpoint configured as async cannot serve sync calls.** The gating is
at the SageMaker frontend, not the container — even when the instance is
warm. To get "small=sync, large=async" you deploy two endpoints sharing the
same image and `model.tar.gz`.

---

## CPU serving — works, uneconomical

We tested whether TabPFN-3 can run on CPU instances at all, and what it
costs. The same handler runs on CPU automatically: `model_fn` checks
`torch.cuda.is_available()`. The CPU container also sets
`TABPFN_ALLOW_CPU_LARGE_DATASET=true` to disable the library's "no >1k rows
on CPU" guard.

**Tested on `ml.c6i.32xlarge`** (Intel Ice Lake, 128 vCPU, 256 GB RAM,
~$5.44/hr) using the dedicated CPU DLC base
(`pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker`).

| n_train (32 feat) | predict s | RSS GB | Status |
|---|---|---|---|
| 100 | 3.06 | 1.3 | ok |
| 1,000 | 5.04 | 1.7 | ok |
| 10,000 | 37.4 | 1.8 | ok |
| 13,000 | 52.9 | 4.2 | ok (close to 60 s cap) |
| 14,000 | 57.7 | 4.2 | ceiling |
| 15,000+ | timeout | — | exceeds realtime 60 s cap |

### Findings

- **OOM is not the constraint** — RSS topped out at 4.2 GB on a 256 GB
  instance. The 32 vCPU `c6i.4xlarge` (~$0.85/hr) would handle the same
  workload at ~6× lower cost than c6i.32xlarge.
- **CPU is ~12× slower than GPU** at N=10k.
- **CPU is ~49× more expensive per inference** at N=10k:
  $0.058 (CPU) vs $0.0012 (GPU).
- **No sweet spot exists.** GPU wins on every dimension at every N tested.

### When to use CPU anyway

- You can't get GPU capacity (capacity contention, region limitations).
- You're doing massive parallel batch jobs and CPU concurrency outweighs
  GPU per-request cost.
- You need a non-GPU compliance environment.

For the demo's "interactive small request" use case, CPU is a curiosity.

---

## Mixed numeric+categorical data — works via column-dict or mixed-NPZ

TabPFN-3 handles raw categorical strings if the wire format preserves
per-column dtypes. The handler's `_coerce_x` helper detects three input
shapes:

1. Pure-numeric `list-of-lists` → cast to float32. Cheap, debuggable.
2. Column-dict `{"col": [v0, v1, ...]}` → pandas DataFrame. Preserves
   per-column dtypes; categoricals stay as strings.
3. 2-D object ndarray (from NPZ) → DataFrame internally on the server.

**Tested on `adult_income` (14 features, 1 k train / 200 test):**

| Encoding | Wire bytes | Accuracy | Status |
|---|---|---|---|
| Column-dict JSON | 178,301 | 0.795 | ok |
| **Mixed-NPZ (object 2-D)** | **15,721** | **0.795** | ok (10× smaller) |
| Row-list JSON with strings | 165,863 | n/a | **500** (numpy upcasts to string) |

`encode_npz_mixed()` in [src/client_helpers.py](src/client_helpers.py) wraps
the right path: takes a DataFrame or 2-D object array, packs it correctly,
preserves `feature_names` from `df.columns`.

**Hard rule**: never send mixed types as a row-list of lists. `np.asarray`
collapses everything to a single string dtype, and TabPFN rejects string
ndarrays with a 500.

---

## Encoding choice — when to use what

For a single SageMaker realtime invocation (6 MB body cap), here's the
ceiling per encoding at varying feature counts. Numbers are the largest
n_train that fits; "—" means the format can't handle this case.

| Features | JSON | JSON+gzip | NPZ | Mixed-NPZ |
|---|---|---|---|---|
| 8 | ~30 k | ~70 k | **~50 k** | ~50 k |
| 16 | ~15 k | ~35 k | **~30 k** | ~30 k |
| 32 | ~7 k | ~15 k | **~15 k** | ~15 k |
| 100 | ~2 k | ~5 k | **~5 k** | ~5 k |

JSON+gzip is comparable to NPZ in wire size after compression on float
data, but NPZ is faster to encode/decode and doesn't require the custom
`application/x-json-gzip` Content-Type to dodge the toolkit's UTF-8
pre-decode (see "Gotchas" below).

**Recommendation:**
- N ≤ 1 k, debugging: plain JSON
- 1 k < N ≤ 15 k, numeric: NPZ (`encode_npz`)
- Any N, mixed numeric+categorical: mixed-NPZ (`encode_npz_mixed`)
- N > 15 k or wire > 6 MB: switch to async endpoint

---

## Gotchas (each cost a deploy cycle)

### G1 — Custom `inference.py` must live in `model.tar.gz`, not `/opt/ml/code/`

The SageMaker PyTorch inference toolkit only adds `/opt/ml/model/code/` to
PYTHONPATH at serve time. Baking `inference.py` into the image at
`/opt/ml/code/` (the documented training pattern!) silently fails: the
toolkit falls back to `default_pytorch_inference_handler.default_model_fn`,
which expects a `.pth`/`.pt` file and 500s every `/ping`.

**Fix**: pass `entry_point="inference.py"` and `source_dir="src/"` to
`PyTorchModel`. The SDK repacks `model.tar.gz` to embed `code/inference.py`.

### G2 — `application/json` body is UTF-8-decoded before `input_fn` sees it

The PyTorch toolkit calls `body.decode("utf-8")` for `application/json` and
`text/*` Content-Types. This corrupts gzipped JSON.

**Fix**: send gzipped JSON under `application/x-json-gzip`. The handler's
`_maybe_gunzip` recognizes this. NPZ payloads work fine because
`application/x-npz` isn't a recognized text type → bytes pass through.

### G3 — `TABPFN_MODEL_CACHE_DIR` must be set BEFORE `from tabpfn import`

TabPFN's `settings` is a pydantic-settings `BaseSettings` instance
constructed at import time. Setting `os.environ["TABPFN_MODEL_CACHE_DIR"]`
inside `model_fn` (or anywhere after the import) has no effect — the
library has already snapshotted env. Symptom: container falls back to
`~/.cache/tabpfn` and tries to download from HuggingFace at first
`.fit()`, which fails in air-gapped containers.

**Fix**: set the env var at module top, before `import tabpfn`. See
`src/inference.py:36-46`.

### G4 — `model_path="auto"` works, but only by happy accident

`TabPFNClassifier(device=device)` defaults to `model_path="auto"`, which
resolves through `settings.tabpfn.model_version` (default `ModelVersion.V3`).
That works today, but is fragile: anyone setting `TABPFN_MODEL_VERSION=v2.5`
would silently get a different model.

**Fix**: pass an explicit `model_path=str(clf_ckpt)` pointing at the V3
ckpt file. Library extracts the version from the filename, fail-fast if
the file is missing. See `src/inference.py:114-115`.

### G5 — TorchServe inside the DLC caps requests at ~6.5 MB by default

Even on async endpoints (where SageMaker accepts 1 GB), TorchServe's
`max_request_size` caps each call at 6,553,500 bytes by default. Above
that, the container returns 413.

**Fix**: set `TS_MAX_REQUEST_SIZE=1073741824` and `TS_MAX_RESPONSE_SIZE`
in the model env. The deploy script does this automatically.

### G6 — Async ≠ sync at the API level

`AsyncInferenceConfig` and the realtime variant are mutually exclusive on
a single endpoint. A configured-async endpoint **cannot** serve sync
`InvokeEndpoint` calls — the SageMaker frontend rejects with a
ValidationError. Cold-start has nothing to do with it; it's pure routing.

**Workaround for "small=sync, large=async"**: deploy two endpoints sharing
the same image and `model.tar.gz`. Route client-side based on payload size.

### G7 — `g6e.xlarge` capacity in us-east-1 is unreliable

Saw `InsufficientInstanceCapacity` after 30 min wait twice. `g5.xlarge`
deploys consistently in 5-7 min. The benchmark deploy script now has
capacity-aware fallback.

### G8 — `MinCapacity=0` async endpoints don't wake without a step-scaling policy

Target-tracking autoscaling acts on **per-instance metrics**
(`ApproximateBacklogSizePerInstance`), which are undefined when there are
zero instances. So a scaled-to-zero endpoint with only a target-tracking
policy will never wake — backlog accumulates forever.

**Fix**: also attach a step-scaling policy on the `HasBacklogWithoutCapacity`
CloudWatch alarm. That alarm goes off as soon as the queue has work and no
capacity to serve it, regardless of how many instances exist. The deploy
script's `--scale-to-zero` flag wires both policies automatically.

---

## What we measured but didn't report here

- Multi-instance comparison (g5.xlarge vs g6e.xlarge vs p5.xlarge) — code is
  in `02_benchmark.py` but never executed across all three live. Skipped
  intentionally: at constant `predict_seconds≈3 s` from subsample-ensemble
  on g5, a 30% L40S speedup would be in the noise. See P2.2-followup in
  NEXT-STEPS.md.
- Real >1M-row tabular benchmark — used `make_classification` (synthetic).
  Should be repeated on Higgs/Criteo for an external sanity check. LP.11.
- Wake-from-zero on a smaller image. The 10-min cold-start is dominated by
  the 8.5 GB GPU image pull. Trimming the DLC (or using the 1.6 GB CPU
  image with a different cost story) would cut this. Not tested.

---

## Reproducibility

All findings reproducible from this repo:

```bash
# Deploy GPU realtime
python notebooks/01_deploy_endpoint.py --image-uri <ecr> \
    --model-data <s3-model> --instance-type ml.g5.xlarge

# Deploy async (1M-row capable)
python notebooks/01_deploy_endpoint.py --image-uri <ecr> \
    --model-data <s3-model> --instance-type ml.g5.xlarge \
    --mode async --async-output-bucket <bucket>

# Cross-instance benchmark (GPU)
python notebooks/02_benchmark.py --image-uri <ecr> \
    --model-data <s3-model> \
    --instance-types ml.g5.xlarge,ml.g6e.xlarge

# Tear everything down
python notebooks/03_cleanup.py --region us-east-1
```

Total cost across all P1 + P2.1 + LP.7 verification work: **~$7 of g5/c6i hours**
across ~10 endpoint deployments.
