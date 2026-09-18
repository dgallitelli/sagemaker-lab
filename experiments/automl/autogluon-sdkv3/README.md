# AutoGluon on Amazon SageMaker with SDK v3

Train, evaluate, deploy, and orchestrate [AutoGluon](https://auto.gluon.ai/) models on Amazon SageMaker using the **SageMaker Python SDK v3**.

This repository demonstrates three ML use cases end-to-end, plus a custom Docker image example:

| Experiment | Task | Dataset | Key Metric |
|---|---|---|---|
| **Tabular Classification** | Binary classification | UCI Adult Census | ROC AUC |
| **TimeSeries Forecasting** | Multi-item forecasting | UCI Electricity | MASE |
| **Multimodal** | Text + tabular fusion | Synthetic churn (text + numerical + categorical) | ROC AUC |
| **Custom Image** | Custom Docker images | UCI Adult Census | ROC AUC |
| **SageMaker Project** | CI/CD MLOps platform | Pluggable (tabular/timeseries/multimodal) | Same as AutoGluon task |

The first three experiments follow the same four-stage structure and demonstrate SageMaker SDK v3 patterns for processing, training, real-time inference, and ML pipelines. The custom image experiment shows how to build and use your own Docker images with the latest AutoGluon version.

## Repository Structure

```
<experiment>/
  0-data-prep/
    0_upload_raw.ipynb          # Upload raw data to S3
    1_launch_processing.ipynb   # SageMaker Processing job
    preprocess.py               # Container script for preprocessing
  1-training/
    launch_training.ipynb       # SageMaker Training job
    train.py                    # Container training script
    config.yaml                 # (tabular only) AutoGluon config
  2-inference/
    deploy.ipynb                # Deploy real-time endpoint (v3 resource API)
    serve.py                    # Inference handler (packaged as code/inference.py)
  3-pipeline/
    pipeline.ipynb              # SageMaker Pipeline orchestration
    evaluate.py / run_evaluation.py  # Evaluation script for pipeline

4-custom-image/               # Custom Docker images with latest AutoGluon
  ag-dlc-upgrade/             # Upgrade AutoGluon DLC base image
  pytorch-dlc/                # Install AutoGluon on PyTorch DLC base
  (each contains: docker/, 0-build-push/, 1-training/, 2-inference/, 3-pipeline/)

5-sagemaker-project/           # SageMaker Projects CI/CD (Build + Deploy pipelines)
  0-project-setup/
  cfn-templates/
  seed-code/{build,deploy}/
```

## Prerequisites

- An AWS account with SageMaker access
- An IAM role with SageMaker execution permissions
- Python 3.12+ with the SageMaker SDK v3 installed (tested with v3.5.0):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Service quotas for the following instance types:
  - `ml.m5.xlarge` — processing and CPU inference
  - `ml.m5.2xlarge` — tabular/timeseries training
  - `ml.g4dn.xlarge` — multimodal training and inference (GPU)

## Getting Started

### 1. Datasets

No manual download is required. Each experiment's `0_upload_raw.ipynb` notebook copies data from the SageMaker example files S3 bucket to your default SageMaker bucket automatically.

The datasets used are:

- **Tabular Classification** — [UCI Adult Census](https://archive.ics.uci.edu/dataset/2/adult): Binary income prediction from demographic features.
- **TimeSeries Forecasting** — [UCI Electricity Load Diagrams](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014): Hourly electricity consumption across 370 households.
- **Multimodal** — Synthetic churn dataset with text, numerical, and categorical features.

### 2. Run notebooks in order

For each experiment, run the notebooks sequentially:

1. `0-data-prep/0_upload_raw.ipynb` — uploads raw data to your default SageMaker S3 bucket
2. `0-data-prep/1_launch_processing.ipynb` — runs a SageMaker Processing job
3. `1-training/launch_training.ipynb` — launches a SageMaker Training job
4. `2-inference/deploy.ipynb` — deploys a real-time endpoint and tests it
5. `3-pipeline/pipeline.ipynb` — creates and runs a SageMaker Pipeline

> **Note:** Step 4 (deploy) requires a trained model artifact. You can either run step 3 first for a standalone training job, or run step 5 (pipeline) and update the model path in the deploy notebook.

## SageMaker SDK v3 Patterns

This repo showcases several key SDK v3 patterns:

### Processing Jobs (v3 step_args pattern)
```python
from sagemaker.core.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

processor = ScriptProcessor(image_uri=..., role=..., instance_type=...)
step = ProcessingStep(name="...", step_args=processor.run(code=..., inputs=..., outputs=...))
```

### Training with ModelTrainer
```python
from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import Compute, OutputDataConfig, SourceCode

trainer = ModelTrainer(training_image=..., source_code=SourceCode(...), compute=Compute(...))
step = TrainingStep(name="...", step_args=trainer.train(input_data_config=[...]))
```

### Real-Time Inference (v3 Resource API)
```python
from sagemaker.core.resources import Model, EndpointConfig, Endpoint
from sagemaker.core.shapes.shapes import ContainerDefinition, ProductionVariant

model = Model.create(model_name=..., primary_container=ContainerDefinition(...), execution_role_arn=...)
config = EndpointConfig.create(endpoint_config_name=..., production_variants=[ProductionVariant(...)])
endpoint = Endpoint.create(endpoint_name=..., endpoint_config_name=...)
endpoint.wait_for_status("InService")
response = endpoint.invoke(body=payload, content_type="text/csv", accept="application/json")
result = response.body.read().decode("utf-8")
```

### Model Repackaging for AutoGluon DLC
The AutoGluon DLC uses TorchServe and requires `code/inference.py` inside the `model.tar.gz`. Pipeline training outputs only contain model artifacts, so the deploy notebook repackages the archive with the inference script.

## Key Considerations

- **AutoGluon DLC version**: 1.5 with Python 3.12
- **Multimodal GPU requirement**: The multimodal experiment requires GPU instances (`ml.g4dn.xlarge`) for training and inference due to the text transformer model
- **TimeSeries inference**: Requires 48+ historical timestamps per item for reliable predictions
- **Evaluation script naming**: The multimodal evaluation script is named `run_evaluation.py` (not `evaluate.py`) to avoid conflicts with the HuggingFace `evaluate` package
- **Pipeline parameters**: Use `ParameterString` for hyperparameters (not `ParameterInteger`), as SageMaker passes all hyperparameters as strings

## Cleanup

To avoid ongoing charges, delete any deployed endpoints:

```python
endpoint.delete()
endpoint_config.delete()
model.delete()
```

Delete SageMaker Pipelines when no longer needed:

```python
import boto3
sm = boto3.client("sagemaker")
sm.delete_pipeline(PipelineName="AutoGluonTabularPipeline")
sm.delete_pipeline(PipelineName="AutoGluonTimeSeriesPipeline")
sm.delete_pipeline(PipelineName="AutoGluonMultimodalPipeline")
```

If you used the custom image experiment, clean up the ECR repository:

```bash
aws ecr delete-repository --repository-name autogluon-custom --force
```

Also delete S3 artifacts if no longer needed:

```bash
aws s3 rm s3://<your-bucket>/autogluon-tabular/ --recursive
aws s3 rm s3://<your-bucket>/autogluon-timeseries/ --recursive
aws s3 rm s3://<your-bucket>/autogluon-multimodal/ --recursive
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file.
