# TabPFN-3 SageMaker Demo — Deferred Next Steps

Captured 2026-05-14. Priority 1 items handled in the same session this file was
created in.

## Priority 2 — Make the demo production-shaped

### P2.1 — Async endpoint variant
- Add `--mode {realtime,async}` flag to `notebooks/01_deploy_endpoint.py`.
- Async endpoint with `MinCapacity=0`, S3-staged payloads, 1 GB ceiling.
- Pair with the existing real-time endpoint for "small=sync, large=async" routing.
- Reference: research already done — async mode is mutually exclusive with sync
  invocations, so the routing must happen client-side across two endpoints.

### P2.2 — Cross-instance benchmark
- Actually deploy on `ml.g6e.xlarge` (L40S) and `ml.g5.xlarge` (A10G) and
  `ml.p5.xlarge` (H100, if available) simultaneously.
- Run `notebooks/02_benchmark.py` with `--instance-types`.
- Note: capacity contention on g6e/p5 in us-east-1; may need us-west-2.

### P2.3 — Capacity-aware fallback in deploy script
- Try `g6e.xlarge` first, fall back to `g5.xlarge` on `InsufficientInstanceCapacity`.
- Bonus: `ml.g6.xlarge` (L4, 24 GB) as middle option.
- Could use the May 2026 `AlternateInstanceTypes` feature in `EndpointConfig`.

## Priority 3 — Polish

### P3.1 — Update README benchmark section
- Currently still references the original synthetic/timeseries/anomaly framing
  but the *real* findings (TabPFN beats XGBoost on 5 of 6 real-world datasets,
  perfect ROC-AUC on creditcardfraud, NPZ encoding stats) are buried in chat
  history. Pull them into the README.

### P3.2 — Document the client_helpers.py encoding patterns
- Add a "Choosing an encoding" section to README.
- NPZ wins for >1k rows numeric data (5-9× smaller wire than JSON).
- JSON+gzip works as a middle ground for mixed-type data.
- Plain JSON fine for <1k rows or when interactive payload sizes are small.

### P3.3 — Pin requirements.txt with hashes
- Currently uses `>=` ranges. Use `pip-compile` to generate a fully pinned
  `requirements.txt` with `--hash=sha256:...` for reproducibility.

### P3.4 — Move pretraining-limit override to a per-request flag
- `payload["ignore_pretraining_limits"] = True` so callers can opt into oversized
  data without redeploying. **Already done as part of P1.2 work.**

## Lower priority / open questions

### LP.1 — Newer DLC base image
- Track `pytorch-inference:2.7` when it ships in 2026.
- Currently on `2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker`.
- No urgency.

### LP.2 — Track tabpfn version
- Currently pinned to `tabpfn==8.0.2` in Dockerfile.
- Re-evaluate as PriorLabs ships fixes.

### LP.3 — Quota for ml.g7e.xlarge / ml.p5.xlarge in us-east-1
- Neither exposed in Service Quotas — likely instance not offered in us-east-1.
- A useful cross-region demo would need us-west-2 (where these are listed).

### LP.4 — Fine-tuning recipe demo
- Show how to fine-tune V3 on customer data, package as `model.tar.gz`,
  drop into the same container.
- This is the reason we ship weights via S3 in the first place — demo it.

### LP.5 — Streaming response endpoint
- TabPFN's `predict_proba` could stream class probabilities for large test sets.
- Not currently implemented; SageMaker streaming endpoints are a different mode.

### LP.6 — Multi-model endpoint
- Host multiple fine-tuned variants on one endpoint with model selection
  per-request. Reduces per-variant idle cost.

### LP.7 — Subsample-ensemble path validation
- If P1.3 confirms it works, formalize as a documented pattern in README.
