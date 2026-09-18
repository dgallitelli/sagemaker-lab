# SageMaker Projects: AutoGluon AutoML CI/CD

Use [SageMaker Projects](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects.html)
as a full MLOps CI/CD platform for AutoGluon AutoML: a custom CloudFormation template (stored in
Amazon S3) provisions a Build pipeline (SageMaker Pipeline: preprocess -> train -> evaluate ->
register) and two Deploy pipelines (real-time endpoint, scheduled batch transform), each gated
staging -> automated test -> manual approval -> production.

This experiment is a **cookie-cutter starting point**: `train.py`/`evaluate.py` are pluggable
across AutoGluon's tabular, timeseries, and multimodal predictors via a `task_type` field in
`config/*.yaml`, so adopting teams swap in their own dataset/config rather than rewriting the
pipeline.

## Repository Structure

```
0-project-setup/
  setup_project.ipynb      # creates GitHub repos, CodeConnections, uploads CFN, creates the Project
cfn-templates/
  project-template.yaml           # Build + Deploy(RealTime) + Deploy(Batch), registered via create_project
  iam-policy-patch.json           # account-setup patch for AmazonSageMakerProjectsCloudformationRole
  iam-policy-patch-codebuild.json # account-setup patch for AmazonSageMakerProjectsCodeBuildRole
seed-code/
  build/                    # pushed to <project>-build — SageMaker Pipeline + CodeBuild
    codebuild-buildspec.yml
    config/{tabular,timeseries,multimodal}.yaml
    pipelines/automl/
      pipeline.py           # get_pipeline(): preprocess -> train -> evaluate -> condition -> register
      preprocess.py
      train.py               # trains, then packages code/serve.py into the model artifact (see below)
      evaluate.py
      serve.py               # inference handler; a build-repo-local copy, baked into model.tar.gz by train.py
    tests/
  deploy/                   # pushed to <project>-deploy — realtime/ and batch/ deploy pipelines, one repo
    realtime/
      build.py, buildspec.yml, endpoint-config-template.yml
      staging-config.json, prod-config.json
      serve.py               # source-of-truth copy for manual/notebook deploys; not what gets baked in
      test/{buildspec.yml,test.py}
      tests/
    batch/
      build.py, buildspec.yml, batch-transform-template.yml
      staging-config.json, prod-config.json
      lambda/run_transform.py
      test/{buildspec.yml,test.py}
      tests/
```

## Prerequisites

- Everything in the root [README](../README.md#prerequisites), plus:
- [GitHub CLI](https://cli.github.com/) (`gh`), authenticated (`gh auth login`)
- A GitHub account/org where new repos can be created
- A [CodeConnections](https://docs.aws.amazon.com/dtconsole/latest/userguide/welcome-connections.html)
  connection to GitHub, tagged `sagemaker=true` (the setup notebook creates one if none exists —
  authorizing it with GitHub is a one-time console step that cannot be scripted)
- This account's `AmazonSageMakerProjectsCloudformationRole` must permit creating a Lambda
  function, IAM role (including `iam:TagRole`), and EventBridge rule matching the `sagemaker-*`
  naming prefix (see `cfn-templates/iam-policy-patch.json`), and this account's
  `AmazonSageMakerProjectsCodeBuildRole` must permit `lambda:InvokeFunction` on `sagemaker-*`
  functions (see `cfn-templates/iam-policy-patch-codebuild.json`) — both are one-time account
  setup steps, already applied in this account as part of building this experiment

## Workflow

1. Run `0-project-setup/setup_project.ipynb` top to bottom. It creates two GitHub repos, an
   S3-hosted CFN template, and the SageMaker Project itself.
2. Upload a raw dataset to `s3://sagemaker-project-<project-id>/<project-name>-<project-id>/raw/`
   before triggering the Build pipeline — nothing in the project setup does this for you.
   `sagemaker-project-<project-id>` is the project's own dedicated artifact bucket, created by
   `project-template.yaml` (distinct from the SDK's shared account-default bucket used elsewhere
   in this repo). For the default tabular config, that means the UCI Adult Census files, the
   same way `1-tabular-classification/0-data-prep/0_upload_raw.ipynb` seeds its own experiment:
   ```bash
   for filename in adult.data adult.test; do
     aws s3 cp "s3://sagemaker-example-files-prod-<region>/datasets/tabular/uci_adult/${filename}" \
       "s3://sagemaker-project-<project-id>/<project-name>-<project-id>/raw/${filename}"
   done
   ```
3. Upload a batch-transform test fixture to `s3://<ARTIFACT_BUCKET>/AutoML/batch-test-input/`
   before the Deploy-batch pipeline's `TestStaging` stage will pass — nothing in the project
   setup does this for you. The fixture must be a small headerless CSV (3 rows recommended) with
   columns matching your model's training features. For the default tabular config, use the same
   Adult Census feature columns as `seed-code/build/config/tabular.yaml`:
   ```bash
   # Create a minimal fixture matching your model's training schema (headerless CSV)
   # Example for Adult Census (matching serve.py's headerless batch-transform handling):
   cat > fixture.csv << 'EOF'
   39,State-gov,77516,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States
   50,Self-emp-inc,83311,Bachelors,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,13,United-States
   38,Private,215646,HS-grad,9,Divorced,Handlers-cleaners,Not-in-family,White,Male,0,0,40,United-States
   EOF
   aws s3 cp fixture.csv s3://sagemaker-project-<project-id>/AutoML/batch-test-input/fixture.csv
   ```
4. Push a change to the `<project>-build` repo (or wait for the initial push) to trigger the
   Build pipeline: preprocess -> train -> evaluate -> register (`PendingManualApproval`).
5. Approve the registered model package in the SageMaker Studio Model Registry UI (or via
   `aws sagemaker update-model-package --model-approval-status Approved`).
6. Both Deploy pipelines trigger automatically off the approval event, deploying to staging,
   running an automated test, then waiting for manual approval before deploying to production.

## How It Works

- `pipeline.py` resolves two distinct AutoGluon DLC images — `autogluon-training` for the
  `Train`/`Evaluate` steps, and `autogluon-inference` for the `RegisterModel` step's container.
  Using the training image to serve inference fails the endpoint's health check.
- The AutoGluon inference container (TorchServe-based) requires a `code/serve.py` entry point
  inside `model.tar.gz`; AutoGluon's own training output is predictor pickle files, not a
  `.pth`/`.pt` checkpoint the container's default handler expects. Inference code is packaged by
  `train.py` itself — after training, it copies `serve.py` (mounted via `ModelTrainer`'s `code`
  input channel) into `{model_dir}/code/serve.py`, so it ends up in the training job's
  `model.tar.gz` automatically. This is deliberate, not a stopgap: `ModelBuilder`/`ModelStep`'s
  SDK-level "runtime repack" mechanism (the natural v3 equivalent of
  `1-tabular-classification/2-inference/deploy.ipynb`'s manual tarball repack) is a silent no-op
  for `ModelBuilder.register()` under a `PipelineSession` in this SDK version — see the seed
  code's inline comments in `pipeline.py`/`train.py` for the full mechanism.
- The same registered model package backs both the real-time endpoint and the batch-transform
  job, so `serve.py`'s `transform_fn` handles both header-CSV (real-time) and headerless-CSV
  (batch) input.

## Bringing Your Own Dataset

- **Tabular** (default, works out of the box): point `InputDataUri` at your own CSV and update
  `label`/`eval_metric` in `seed-code/build/config/tabular.yaml`.
- **Timeseries / Multimodal**: replace `seed-code/build/pipelines/automl/preprocess.py` with a
  script matching your data shape (see `2-timeseries-forecasting/0-data-prep/preprocess.py` and
  `3-multimodal/0-data-prep/preprocess.py` in this repo for worked examples), then use
  `config/timeseries.yaml`/`config/multimodal.yaml` as your starting config. Also set the
  `CONFIG_FILE` environment variable on the ModelBuild CodeBuild project (Console: project
  Settings > Environment > Environment variables, or `aws codebuild start-build
  --environment-variables-override`) to the matching filename, otherwise the pipeline keeps
  using `tabular.yaml`.

## Cleanup

`aws sagemaker delete-project` only deletes the top-level Service Catalog product stack — it
does **not** tear down the Deploy pipelines' own CloudFormation stacks (created by their
`DeployResourcesStaging`/`DeployResourcesProd` CodePipeline actions), their EventBridge batch
schedule rules, or the Model Registry's model package group. Skipping the steps below leaves an
endpoint-adjacent Lambda, an IAM role, and a recurring EventBridge schedule running indefinitely.

```bash
# 1. Delete real-time endpoints
aws sagemaker delete-endpoint --endpoint-name <project-name>-staging
aws sagemaker delete-endpoint --endpoint-name <project-name>-prod

# 2. Delete the SageMaker Project (removes the Project's own CFN stack, its CodePipelines/
#    CodeBuild projects, and the model-approval EventBridge trigger rules)
aws sagemaker delete-project --project-name <project-name>

# 3. Delete the four Deploy-pipeline-managed CloudFormation stacks directly — delete-project does
#    NOT do this. This also removes the batch-transform Lambda, its IAM role, and both
#    (staging/prod) EventBridge schedule rules.
aws cloudformation delete-stack --stack-name sagemaker-<project-name>-<project-id>-deploy-rt-staging
aws cloudformation delete-stack --stack-name sagemaker-<project-name>-<project-id>-deploy-rt-prod
aws cloudformation delete-stack --stack-name sagemaker-<project-name>-<project-id>-deploy-batch-staging
aws cloudformation delete-stack --stack-name sagemaker-<project-name>-<project-id>-deploy-batch-prod

# 4. Delete the Model Registry group — delete-project does NOT do this either
for v in $(aws sagemaker list-model-packages --model-package-group-name <project-name>-<project-id> \
    --query 'ModelPackageSummaryList[].ModelPackageArn' --output text); do
  aws sagemaker delete-model-package --model-package-name "$v"
done
aws sagemaker delete-model-package-group --model-package-group-name <project-name>-<project-id>

# 5. Delete the GitHub repos
gh repo delete <owner>/<project-name>-build --yes
gh repo delete <owner>/<project-name>-deploy --yes

# 6. Delete S3 artifacts — all locations
aws s3 rm s3://sagemaker-project-<project-id>/ --recursive
aws s3 rm s3://<sdk-default-bucket>/sagemaker-projects-templates/<project-name>-project-template.yaml
# NOTE: The artifact bucket (ARTIFACT_BUCKET, e.g. sagemaker-project-<project-id>) is used for
# all pipeline artifacts in this workflow, so the SDK default bucket (`sagemaker-us-east-1-<account-id>`)
# is only used for the CFN template itself. If you manually set default_bucket to a different value,
# also clean: aws s3 rm s3://<custom-default-bucket>/<project-name>-<project-id>/ --recursive
```

Verify nothing is left with a final sweep:
```bash
aws sagemaker list-projects --query "ProjectSummaryList[?ProjectName=='<project-name>']"
aws sagemaker list-endpoints --query "Endpoints[?contains(EndpointName,'<project-name>')]"
aws cloudformation list-stacks --query "StackSummaries[?contains(StackName,'<project-id>') && StackStatus!='DELETE_COMPLETE']"
aws events list-rules --query "Rules[?contains(Name,'<project-name>') || contains(Name,'<project-id>')]"
aws sagemaker list-model-package-groups --query "ModelPackageGroupSummaryList[?contains(ModelPackageGroupName,'<project-id>')]"
```

The account-level IAM policy patches (`iam-policy-patch.json`,
`iam-policy-patch-codebuild.json`) are reusable, account-wide fixes, not per-project resources —
leave them in place for future SageMaker Projects in the same account.
