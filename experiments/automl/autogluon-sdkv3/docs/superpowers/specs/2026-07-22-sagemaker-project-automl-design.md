# 5-sagemaker-project: AutoGluon AutoML on SageMaker Projects (CI/CD)

## Purpose

Add a fifth experiment demonstrating SageMaker Projects as an "MLOps platform" for AutoGluon
AutoML: a CI/CD pipeline (SageMaker Pipelines + CodePipeline + CodeBuild + GitHub) that builds,
evaluates, registers, and deploys an AutoGluon model, gated by manual approval between staging
and production. IaC (the SageMaker Project's CloudFormation template) is stored in Amazon S3 and
registered with SageMaker Studio via the `create_project` "bring your own template" API — not
Service Catalog.

This experiment is meant as a **cookie-cutter starting point for production use**, not a single
fixed dataset demo. The seed code's `train.py`/`evaluate.py` are written to generalize across
AutoGluon's three predictor families (tabular, timeseries, multimodal) via a config file, so a
team adopting this template swaps in their own dataset and config rather than rewriting the
pipeline.

## Precedent in this AWS account

Account `859755744029` already has a working SageMaker Project (`petro-s3-project-test3`)
built exactly this way: a CFN template uploaded to
`s3://sagemaker-us-east-1-859755744029/sagemaker-projects-templates/build-train-deploy-monitor-codepipeline.yaml`,
registered via:

```python
sagemaker_client.create_project(
    ProjectName="...",
    TemplateProviders=[{
        "CfnTemplateProvider": {
            "TemplateName": "...",
            "TemplateURL": "https://<bucket>.s3.<region>.amazonaws.com/<key>",
            "Parameters": [...],
        }
    }],
)
```

wired to two GitHub repos (`dgallitelli/petro-build`, `dgallitelli/petro-deploy`) via a
CodeConnections connection tagged `sagemaker=true`. This experiment follows the same mechanism,
trimmed to Build + Deploy (no Model Monitor stage) and with two Deploy flavors (real-time,
batch) instead of one.

The account's SageMaker Projects service roles already exist and require no setup:
`AmazonSageMakerProjectsExecutionRole`, `AmazonSageMakerProjectsCodeBuildRole`,
`AmazonSageMakerProjectsCodePipelineRole`, `AmazonSageMakerProjectsCloudformationRole`,
`AmazonSageMakerProjectsEventsRole`, `AmazonSageMakerProjectsLambdaRole`.

The caller running the setup notebook needs: `sagemaker:CreateProject`/`DescribeProject`,
`s3:PutObject` on the default SageMaker bucket, `codeconnections:ListConnections` (to check for
an existing `sagemaker=true`-tagged connection), and `iam:PassRole` for the six roles above (the
CFN template assumes them at deploy time, not the caller directly, but CloudFormation requires
the deploying principal to have `PassRole` for any role a stack references). No new IAM roles or
policies are created by this experiment.

## Correcting a bug in this repo's existing pipelines

`1-tabular-classification/3-pipeline/pipeline.ipynb`, `2-timeseries-forecasting/3-pipeline/pipeline.ipynb`,
and `3-multimodal/3-pipeline/pipeline.ipynb` all contain a comment stating model registration was
removed because `Model.register()` is "incompatible with PipelineSession/PipelineVariable."

This is a misdiagnosis. `sagemaker.core.resources.Model.register` is Python's
`ABCMeta.register()` (virtual-subclass registration) — an unrelated method that happens to share
a name, not a SageMaker model-registry API. It was never going to do anything useful.

Verified against the installed SDK (`sagemaker==3.16.0`) that the actual, documented v3
mechanism is:

```python
from sagemaker.serve import ModelBuilder

model_builder = ModelBuilder(
    model=trainer,                 # accepts a ModelTrainer directly
    sagemaker_session=pipeline_session,
    role_arn=role_arn,
)
register_step_args = model_builder.build().register(
    model_package_group_name=...,
    content_types=[...],
    response_types=[...],
    inference_instances=[...],
    transform_instances=[...],
    approval_status="PendingManualApproval",
)
step_register = ModelStep(name="RegisterModel", step_args=register_step_args)
```

`ModelStep` (`sagemaker.mlops.workflow.model_step.ModelStep`) explicitly validates that
`step_args` came from `ModelBuilder.build()`/`.register()` — confirming this is the intended v3
path, not the plain `Model` resource class.

This experiment's `pipeline.py` restores `ConditionStep` + `ModelStep` using the correct
`ModelBuilder` pattern. This is a new pipeline in a new directory — none of the other four
experiments' existing pipelines are modified by this work.

## Architecture

```
GitHub: <project>-build          GitHub: <project>-deploy
       |                                |
       v                                v
ModelBuildPipeline               ModelDeployRealTimePipeline
 Source -> Build                  Source -> Build -> DeployStaging(+Test+Approval) -> DeployProd
 (CodeBuild runs the
  SageMaker Pipeline:             ModelDeployBatchPipeline
  Preprocess -> Train ->           Source -> Build -> DeployStaging(+Test+Approval) -> DeployProd
  Evaluate -> Condition ->
  Register)
       |
       v
  Model Registry (ModelPackageGroup)
       |  EventBridge: Model Package State Change (status != PendingManualApproval)
       +----------------------------------------------+
       v                                               v
ModelDeployRealTimePipeline triggers          ModelDeployBatchPipeline triggers
```

- **Build**: one SageMaker Pipeline, defined in `pipelines/automl/pipeline.py`, run by CodeBuild
  on every push to the build repo. Steps: `PreprocessData` (ScriptProcessor) → `TrainAutoGluon`
  (`ModelTrainer`) → `EvaluateModel` (ScriptProcessor, writes `evaluation.json`) →
  `ConditionStep` (metric vs. threshold) → `RegisterModel` (`ModelStep` via `ModelBuilder`, as
  above).
- **Deploy (real-time)**: staging/prod `Model` + `EndpointConfig` + `Endpoint`, same v3 resource
  API already used in `1-tabular-classification/2-inference/deploy.ipynb`. Reuses that notebook's
  model-repackaging logic (adding `code/inference.py` for the AutoGluon DLC/TorchServe
  requirement) as a build-time step, so the registered model package is already deploy-ready —
  no repackaging needed at deploy time.
- **Deploy (batch)**: staging/prod `Model` + a Lambda (`run_transform.py`) that calls
  `Transformer.transform()` and waits for completion + an EventBridge Scheduled Rule
  (`rate(1 day)` default, parameterized) that invokes the Lambda in prod. Staging's Test stage
  invokes the Lambda synchronously against a small fixture dataset and asserts the transform job
  succeeds with expected output shape.
- Both Deploy pipelines source from the **same** `<project>-deploy` repo (avoids repo sprawl),
  each with its own subdirectory (`realtime/`, `batch/`) and CodeBuild project/buildspec, so they
  build and test independently.
- No Model Monitor stage, no baseline/drift CodeBuild project, no
  `InServiceEndpointEventRule` — out of scope per explicit choice.

## Pluggable `train.py` / `evaluate.py`

One script per role, shared across task types, selected by a `task_type` field in a YAML config
(same `config` channel convention as the existing three experiments):

```yaml
task_type: tabular          # tabular | timeseries | multimodal
label: class                 # tabular/multimodal target column
eval_metric: roc_auc         # or auto
presets: best_quality
# tabular: no extra fields required beyond label/eval_metric
# timeseries: id_column, timestamp_column, prediction_length
# multimodal: numerical_features, categorical_features, textual_features
```

`train.py` branches once, at the top, on `task_type` to construct the right AutoGluon predictor
class (`TabularPredictor` / `TimeSeriesPredictor` / `MultiModalPredictor`), then follows the same
load → `fit()` → evaluate → write-`evaluation.json` → save shape already proven in the three
existing experiments' training scripts. `evaluate.py` (the pipeline's standalone evaluation
step) branches the same way to load the correct predictor class and compute the same
`evaluation.json` format the existing `ConditionStep` pattern expects
(`{"metrics": {<eval_metric>: <value>}}`).

**`preprocess.py` is not made generic.** Dataset shaping is inherently task-specific (wide-to-long
resampling for timeseries, JSONL feature assembly for multimodal, CSV cleaning for tabular) — the
existing three experiments' preprocessors already prove this. Only the tabular preprocessor
(UCI Adult Census, adapted from `1-tabular-classification/0-data-prep/preprocess.py`) ships as a
working default, so the pipeline runs end-to-end out of the box. `config/timeseries.yaml` and
`config/multimodal.yaml` ship as documented reference configs with a README note that adopting
teams supply their own `preprocess.py` matching their data shape for those task types.

Inference handlers (`serve.py` for real-time, and a batch-transform handler adapted from the
existing `1-tabular-classification/2-inference/serve_batch.py`) also branch on the predictor
class the same way, so one seed-code inference script serves all three task types.

## Repo layout

```
5-sagemaker-project/
  README.md
  0-project-setup/
    setup_project.ipynb        # creates GitHub repos, walks through CodeConnections,
                                # uploads CFN template to S3, pushes seed code, calls create_project()
  cfn-templates/
    project-template.yaml      # Build + Deploy(RealTime) + Deploy(Batch), 2 EventBridge rules
  seed-code/
    build/                     # pushed to <project>-build repo
      pipelines/automl/
        pipeline.py            # Preprocess -> Train -> Evaluate -> Condition -> Register
        preprocess.py          # tabular (Adult Census) — working default
        train.py               # pluggable: task_type switch
        evaluate.py            # pluggable: task_type switch
      pipelines/run_pipeline.py
      pipelines/_utils.py
      codebuild-buildspec.yml
      config/tabular.yaml
      config/timeseries.yaml   # reference config; requires custom preprocess.py
      config/multimodal.yaml   # reference config; requires custom preprocess.py
    deploy/                    # pushed to <project>-deploy repo
      realtime/
        build.py               # emits Model+EndpointConfig+Endpoint CFN + staging/prod configs
        buildspec.yml
        serve.py               # pluggable inference handler
        test/test.py
        test/buildspec.yml
      batch/
        build.py                # emits Model + Lambda + EventBridge Schedule CFN + staging/prod configs
        buildspec.yml
        serve_batch.py          # pluggable batch inference handler
        lambda/run_transform.py # create_transform_job + wait for completion
        test/test.py
        test/buildspec.yml
```

## Setup notebook (`0-project-setup/setup_project.ipynb`)

1. `gh repo create` for `<project>-build` and `<project>-deploy` (private, under whichever
   GitHub account/org the user's authenticated `gh` CLI defaults to).
2. Check for an existing CodeConnections connection tagged `sagemaker=true` and status
   `AVAILABLE`; if none exists, print console instructions (OAuth authorization cannot be
   scripted) and poll until the user completes it.
3. Upload `cfn-templates/project-template.yaml` to the SageMaker default bucket.
4. `git init`/push the two seed-code trees into the newly created repos.
5. Call `sagemaker_client.create_project(...)` with `TemplateProviders=[{"CfnTemplateProvider": {...}}]`
   and the repo/branch/connection parameters.
6. Poll `describe_project` until `ProjectStatus == "CreateCompleted"`; print the three
   CodePipeline console URLs (Build, Deploy-RealTime, Deploy-Batch).

## Cleanup

Documented in `5-sagemaker-project/README.md`, following the pattern of the root README's
Cleanup section: delete the SageMaker Project (which deletes its CFN stacks), delete the two
GitHub repos, delete the S3 seed-code/artifact objects, and delete any live staging/prod
endpoints or scheduled batch transform rules left running.

## Out of scope

- Model Monitor stage (no baseline capture, no drift CodeBuild project, no
  `InServiceEndpointEventRule`).
- Modifying any of the four existing experiments' pipeline notebooks — the `Model.register()`
  fix is demonstrated only in this new experiment's `pipeline.py`.
- Service Catalog portfolio/product registration — this uses the direct `create_project`
  `TemplateProviders` API instead.
