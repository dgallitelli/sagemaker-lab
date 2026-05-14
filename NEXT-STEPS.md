# TabPFN-3 SageMaker Demo — Roadmap

Last updated 2026-05-14 after the P1 probe round, P2.1/P2.2 verification,
the LP.1/LP.2/LP.7/LP.8 + P3.3 round, and the P3.7/P3.8/P3.9 round.
**Completed work is in [RESULTS.md](RESULTS.md).** This file lists what's
left.

Status legend: `[ ]` open · `[x]` done · `[~]` partial

## Priority 2 — Production shape

- [x] **P2.1** — Async endpoint variant in deploy script. Verified to 1M rows
  on g5.xlarge in 11s end-to-end. See RESULTS.md "Async at scale".
- [x] **P2.2** — Cross-instance benchmark wiring (capacity-aware fallback in
  `02_benchmark.py`). Code merged; not yet exercised live across multiple
  instance families. See P2.2-followup below.
- [x] **P2.3** — Capacity-aware fallback in deploy. Folded into P2.2 (the
  benchmark deploy() is the canonical path). The standalone deploy script
  (`01_deploy_endpoint.py`) does NOT yet have fallback — add it if needed.

### P2.2-followup — actually run the cross-instance benchmark
- Deploy on `ml.g5.xlarge` + `ml.g6e.xlarge` simultaneously and capture the
  real GPU comparison table (A10G 24 GB vs L40S 48 GB).
- Same image, same model.tar.gz; differ only in instance-type.
- Likely needs to run during a non-peak window for g6e capacity in us-east-1.
- Keep the benchmark output (`benchmark_results.json`) as a committed
  artifact under `results/` so people can compare without redeploying.

## Priority 3 — Polish

- [x] **P3.1** — README benchmark section rewritten with P1 results table.
- [x] **P3.2** — `encode_npz_mixed` helper + "Choosing an encoding" docs.
- [x] **P3.3** — Pin requirements.txt with hashes via `pip-compile`. **Done
  2026-05-14** — `requirements.txt` now contains the full pinned closure with
  SHA hashes; source-of-truth is `requirements.in`. Re-run with
  `pip-compile --generate-hashes --allow-unsafe -o requirements.txt requirements.in`.
- [x] **P3.4** — Per-request `ignore_pretraining_limits` flag (done in P1.2).
- [~] **P3.5** — Won't do unless someone asks. RESULTS.md captures the same
  numbers in narrative form; JSON dumps would go stale faster than they'd
  help.

### Polish backlog
- [~] **P3.6** — Notebook walkthrough. **Won't do.** Considered converting
  the `runners/` scripts to `.ipynb`. Decided against: the `.py` files
  already render fine on GitHub, are short and well-commented, and stay
  CLI-runnable. Notebooks would either drift from the scripts (dual
  maintenance) or be empty shells that obscure structure under JSON. The
  `runners/` rename (was `notebooks/`) reflects what they actually are —
  operational entry points, not narrative walkthroughs.
- [x] **P3.7** — CI on GitHub Actions. **Done 2026-05-14**. `.github/workflows/ci.yml`
  AST-parses every Python file then runs `ruff check`. `pyproject.toml` holds
  the lint config (E402 silenced in `inference.py` only, since the
  env-var-before-import pattern is intentional per Gotcha G3).
- [x] **P3.8** — Per-request `n_estimators` and `softmax_temperature` overrides.
  **Done 2026-05-14**. Both flow through `input_fn` and `_maybe_per_request_model`;
  bad values (non-positive int, softmax on regression) raise `ValueError`.
  Live-verified — `predict_seconds` scales sub-linearly with `n_estimators`
  (0.5s @ n=4, 1.0s @ n=8, 1.9s @ n=16).
- [x] **P3.9** — `validate_inference_config()` in `client_helpers.py`. **Done
  2026-05-14**. Frozenset of canonical InferenceConfig keys snapshotted from
  tabpfn==8.0.2; `difflib`-based "did you mean" suggestions for typos. Auto-
  invoked from `encode_json`/`encode_npz`/`encode_npz_mixed` so any caller
  using the helpers gets validation for free.

## Lower priority / future ideas

- [x] **LP.1** — Track `pytorch-inference:2.7` DLC. **Checked 2026-05-14** —
  no 2.7 inference DLC published in us-east-1 yet; still on 2.6.0. Re-check
  in a few weeks.
- [x] **LP.2** — Re-evaluate tabpfn pin. **Checked 2026-05-14** — 8.0.2 is
  still the latest on PyPI. Pin holds.
- [x] **LP.7** — Async with scale-to-zero. **Done 2026-05-14**. Wired
  `--scale-to-zero` in `01_deploy_endpoint.py` with target-tracking +
  step-scaling on `HasBacklogWithoutCapacity`. Live-verified scale-down at
  13 min idle and wake-from-zero at ~10 min total. See RESULTS.md "Async
  with scale-to-zero".
- [x] **LP.8** — Per-request payload validation in `input_fn`. **Done
  2026-05-14**. `_validate_shapes` rejects pathological inputs (shape
  mismatch, empty, oversize, too many features) with clear messages. Limits
  configurable via `TABPFN_MAX_TRAIN_ROWS`/`TEST_ROWS`/`FEATURES`/
  `WIRE_BYTES` env vars on the model. Live-verified.

### Deferred
- [ ] **LP.4** — Fine-tuning recipe demo. The whole reason we ship weights via
  S3 is to support custom checkpoints; show that path. Take a public dataset,
  fine-tune V3, repackage as `model.tar.gz`, deploy on the same image.
- [ ] **LP.6** — Multi-model endpoint. Host multiple fine-tuned variants on
  one endpoint with model selection per-request. Reduces per-variant idle
  cost.
- [ ] **LP.9** — Test on Graviton (`ml.c7g.*`). The CPU image is x86 only;
  building an arm64 variant would let us test the cheaper instance class.
  Likely a non-trivial cost lever for batch inference workloads.
- [ ] **LP.10** — Async path on CPU. We tested CPU only on realtime; async
  + CPU might unlock larger payloads at lower cost than g5 if latency isn't
  critical. Worth a single data point.
- [ ] **LP.11** — Real-world large dataset benchmark. The 1M-row test was
  synthetic. Run the same sweep on a real >1M-row tabular dataset (e.g.
  Higgs Boson, Criteo) to validate the subsample-ensemble accuracy story
  outside `make_classification`.
