# TabPFN-3 SageMaker Demo — Roadmap

Last updated 2026-05-14 after the P1 probe round + P2.1/P2.2 verification.
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
- [ ] **P3.3** — Pin requirements.txt with hashes (`pip-compile`).
- [x] **P3.4** — Per-request `ignore_pretraining_limits` flag (done in P1.2).

### Polish backlog
- [ ] **P3.5** — Add a `results/` directory with the JSON outputs from the P1
  probe runs (subsample-ensemble table, async sweep, CPU sweep). Useful for
  reproducibility / regression tracking.
- [ ] **P3.6** — Notebook walkthrough: convert `notebooks/01_deploy_endpoint.py`
  into an actual `.ipynb` so people skimming GitHub can read the flow without
  cloning. Same for `02_benchmark.py`.
- [ ] **P3.7** — Add CI: `python -c "import ast; ast.parse(...)"` on every
  Python file, plus a black/ruff lint pass. Cheap insurance against typos.
- [ ] **P3.8** — Add per-request `n_estimators` override. P1.3 found that
  `inference_config` is Pydantic `extra="forbid"` so unknown keys crash, and
  `n_estimators` is a constructor arg, not an InferenceConfig field. Wire it
  as a separate top-level payload key.
- [ ] **P3.9** — Client-side `inference_config` schema validation. Pydantic
  rejects typos with a 500; surface that as a 400 (or, better, validate
  client-side before invoke). Could ship a tiny `validate_inference_config()`
  in `client_helpers.py`.

## Lower priority / future ideas

- [ ] **LP.1** — Track `pytorch-inference:2.7` DLC when it ships. Currently
  on `2.6.0-{gpu,cpu}-py312-ubuntu22.04-sagemaker`.
- [ ] **LP.2** — Re-evaluate tabpfn pin as PriorLabs ships fixes (currently
  `8.0.2`).
- [ ] **LP.3** — Cross-region demo on us-west-2 where g7e.xlarge / p5.xlarge
  are listed. Would unlock the original 3-tier benchmark (A10G / L40S /
  Blackwell or H100).
- [ ] **LP.4** — Fine-tuning recipe demo. The whole reason we ship weights via
  S3 is to support custom checkpoints; show that path. Take a public dataset,
  fine-tune V3, repackage as `model.tar.gz`, deploy on the same image.
- [ ] **LP.5** — Streaming response endpoint. `predict_proba` could stream
  per-row probabilities for very large test sets. SageMaker
  `InvokeEndpointWithResponseStream` is the API; needs a different
  `output_fn` shape.
- [ ] **LP.6** — Multi-model endpoint. Host multiple fine-tuned variants on
  one endpoint with model selection per-request. Reduces per-variant idle
  cost.
- [ ] **LP.7** — Async with scale-to-zero (`MinCapacity=0`) + autoscaling
  policy on `HasBacklogWithoutCapacity`. The P2.1 work deployed an async
  endpoint at fixed capacity 1 — extending to true scale-to-zero is a
  separate config change with cold-start implications.
- [ ] **LP.8** — Per-request payload validation in `input_fn` (max rows /
  cells / wire size) so misconfigured clients get 400s instead of OOMs.
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
