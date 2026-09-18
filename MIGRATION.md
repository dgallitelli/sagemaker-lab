# Migration record

## Repository mapping

| Original repository | Destination |
| --- | --- |
| `dgallitelli/gemma4-unsloth-sagemaker` | `experiments/llm-training/gemma4-unsloth` |
| `dgallitelli/qwen35-sft-sagemaker` | `experiments/llm-training/qwen35-sft` |
| `dgallitelli/sagemaker-autogluon-sdkv3` | `experiments/automl/autogluon-sdkv3` |
| `dgallitelli/sagemaker-ntl-detection-with-xgboost-and-chronos` | `experiments/time-series/ntl-xgboost-chronos` |
| `dgallitelli/sagemaker-sdk-v3-xgboost-example` | `experiments/tabular/xgboost-sdkv3` |
| `dgallitelli/sagemaker-splade-embeddings` | `experiments/embeddings/splade` |
| `dgallitelli/sagemaker-tabpfn3-experiments` | `experiments/tabular/tabpfn3` |

## GitHub migration policy

`sagemaker-tabpfn3-experiments` is the anchor repository and is intended to be renamed
to `sagemaker-lab`. GitHub redirects the anchor's former repository URL after a rename.

The other source repositories should remain archived with a short README pointing to
their destination above. They should not be deleted if preserving discoverability from
their existing URLs is important: deleting a repository removes the URL instead of
redirecting it.

## History policy

Source histories are imported without squashing. Each imported experiment remains
self-contained under its destination directory.
