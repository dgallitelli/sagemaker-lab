# SageMaker Lab

Experiments, reference implementations, and validated findings for Amazon SageMaker AI.

This repository groups self-contained experiments that were previously maintained as
separate repositories. Each experiment keeps its own dependencies, documentation, and
cleanup instructions; there is intentionally no repository-wide Python environment.

## Experiments

| Area | Experiment | Directory |
| --- | --- | --- |
| LLM training | Gemma 4 with Unsloth | [`experiments/llm-training/gemma4-unsloth`](experiments/llm-training/gemma4-unsloth) |
| LLM training | Qwen 3.5 supervised fine-tuning | [`experiments/llm-training/qwen35-sft`](experiments/llm-training/qwen35-sft) |
| AutoML | AutoGluon with SageMaker SDK v3 | [`experiments/automl/autogluon-sdkv3`](experiments/automl/autogluon-sdkv3) |
| Time series | Electricity-theft detection with XGBoost and Chronos | [`experiments/time-series/ntl-xgboost-chronos`](experiments/time-series/ntl-xgboost-chronos) |
| Tabular ML | XGBoost with SageMaker SDK v3 | [`experiments/tabular/xgboost-sdkv3`](experiments/tabular/xgboost-sdkv3) |
| Embeddings | SPLADE sparse-embedding training | [`experiments/embeddings/splade`](experiments/embeddings/splade) |
| Tabular ML | TabPFN-3 inference and benchmarking | [`experiments/tabular/tabpfn3`](experiments/tabular/tabpfn3) |

## Working with an experiment

Start in the experiment's directory and follow its README. Dependencies and commands
are scoped to that directory because the experiments use different frameworks and
runtime assumptions.

Always follow the experiment's cleanup instructions after running SageMaker resources.

## Repository layout

```text
experiments/
├── automl/
├── embeddings/
├── llm-training/
├── tabular/
└── time-series/
```

## History and provenance

The original Git histories are retained in this repository. See
[`MIGRATION.md`](MIGRATION.md) for the source repository mapping and migration policy.

## Licensing

Licensing is scoped per experiment. Existing license files were retained inside their
experiment directories. An experiment without an explicit license file is not covered
by a repository-wide open-source license.
