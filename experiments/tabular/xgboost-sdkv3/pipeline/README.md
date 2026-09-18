# SageMaker Pipeline for Gas Lift Optimization

Automated ML workflow with quality gates and Model Registry integration using SageMaker SDK v3.

## Pipeline Flow

```
┌──────────────┐   ┌─────────┐   ┌──────────┐   ┌────────────────┐
│  Parameters  │──▶│Preprocess│──▶│  Train   │──▶│    Evaluate    │
└──────────────┘   └─────────┘   └─────────┘   └───────┬────────┘
                                                       │
                                               ┌───────▼────────┐
                                               │ R2 >= threshold?│
                                               └───────┬────────┘
                                        ┌──────────────┼──────────────┐
                                        │ Yes                         │ No
                                ┌───────▼────────┐            ┌───────▼────────┐
                                │ Register Model │            │   Fail Step    │
                                └────────────────┘            └────────────────┘
```

## Quick Start

```bash
# 1. Create the pipeline
python run_pipeline.py create --region us-east-1

# 2. Run the pipeline
python run_pipeline.py run \
    --input-data-uri s3://your-bucket/gas-lift-optimization/data/ \
    --r2-threshold 0.80 \
    --wait

# 3. Check status
python run_pipeline.py describe
```

## Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ProcessingInstanceType` | ml.m5.xlarge | Instance for preprocessing/evaluation |
| `TrainingInstanceType` | ml.m5.xlarge | Instance for training |
| `InputDataUri` | - | S3 path to input data (CSV/Parquet) |
| `MaxDepth` | 6 | XGBoost max_depth |
| `NEstimators` | 100 | XGBoost n_estimators |
| `Eta` | 0.1 | XGBoost learning_rate |
| `Subsample` | 0.8 | XGBoost subsample |
| `R2Threshold` | 0.85 | Quality gate threshold |
| `ModelApprovalStatus` | PendingManualApproval | Model approval status |

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Main pipeline definition |
| `preprocessing.py` | Data loading and train/test split |
| `evaluation.py` | Model metrics calculation |
| `run_pipeline.py` | CLI runner |

## Training Code Reuse

The training step reuses `training/train.py` from the parent directory:

```python
source_code = SourceCode(
    source_dir="../training",
    entry_script="train.py",
    requirements="requirements.txt",
)
```

No code duplication - the pipeline uses the same training script as the notebook workflow.

## Quality Gate

The pipeline includes a conditional step that checks if R2 >= threshold:
- **Pass**: Model is registered to `GasLiftOptimizationModels` package group
- **Fail**: Pipeline fails with clear error message

### Testing the Quality Gate

```bash
# Should fail (threshold too high)
python run_pipeline.py run --r2-threshold 0.99 --wait

# Should pass (reasonable threshold)
python run_pipeline.py run --r2-threshold 0.50 --wait
```

## Model Registry

Registered models appear in SageMaker Model Registry under:
- **Package Group**: `GasLiftOptimizationModels`
- **Status**: `PendingManualApproval` (default)

To approve for production deployment:
```python
import boto3
client = boto3.client("sagemaker")
client.update_model_package(
    ModelPackageArn="arn:aws:sagemaker:...",
    ModelApprovalStatus="Approved"
)
```

## SDK v3 Patterns

This pipeline demonstrates SDK v3 best practices:

1. **ModelTrainer** - Unified training API with `train()` method
2. **SourceCode** - Script packaging (source_dir + entry_script)
3. **PipelineSession** - Deferred execution for pipeline steps
4. **PropertyFile** - Cross-step data passing for conditions
5. **JsonGet** - Extract values from JSON for quality gates
6. **CacheConfig** - 30-day step caching for faster reruns
7. **ModelBuilder** - Model registration with `register()` method

### SDK v3.4.1 Bug Workaround

This pipeline includes a monkey-patch for a bug in SDK v3.4.1 where `get_training_code_hash()`
expects `dependencies: List[str]` but receives `requirements: str`. The patch is applied
automatically in `pipeline.py`.

## Troubleshooting

### Execution Failed

```bash
# Get detailed step status
python run_pipeline.py describe --execution-arn arn:aws:sagemaker:...
```

Check CloudWatch logs at:
```
/aws/sagemaker/ProcessingJobs
/aws/sagemaker/TrainingJobs
```

### Model Not Registered

Verify the R2 score meets the threshold. Run with lower threshold for testing:
```bash
python run_pipeline.py run --r2-threshold 0.50 --wait
```

### IAM Permissions

The execution role needs:
- `sagemaker:*` for pipeline operations
- `s3:*` for data access
- `logs:*` for CloudWatch
- `ecr:*` for container images
