"""AutoGluon AutoML SageMaker Pipeline (SDK v3) for SageMaker Projects CI/CD.

Preprocess -> Train -> Evaluate -> Condition (metric >= threshold) ->
RegisterModel (else FailStep). Implements get_pipeline(**kwargs) for the
petro-build-style pipelines._utils.get_pipeline_driver integration.

Registration uses ModelBuilder(...).register(...) -> ModelStep. Do NOT call
.register() on a plain sagemaker.core.resources.Model — that resolves to
Python's ABCMeta.register() (virtual-subclass registration), not the Model
Registry API. Verified against sagemaker==3.16.0.
"""
import os

import boto3
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.core.shapes.shapes import ProcessingS3Input, ProcessingS3Output
from sagemaker.core.training.configs import Compute, OutputDataConfig, SourceCode, StoppingCondition
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.core.workflow.functions import JsonGet
from sagemaker.core.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.fail_step import FailStep
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.serve import ModelBuilder
from sagemaker.train import ModelTrainer

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
    return boto3.Session(region_name=region).client("sagemaker")


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        project_arn = sm_client.describe_project(ProjectName=sagemaker_project_name)["ProjectArn"]
        for tag in sm_client.list_tags(ResourceArn=project_arn)["Tags"]:
            new_tags.append(tag)
    except Exception as e:  # noqa: BLE001 — tagging is best-effort
        print(f"Error getting project tags: {e}")
    return new_tags


def _retrieve_autogluon_image(region, ag_version, py_version, instance_type, image_scope):
    """Resolve an AutoGluon training or inference image URI.

    Works around a persistent bug in sagemaker SDK's bundled image config
    (confirmed across versions 3.5.0-3.16.0): it lists only py311 as valid
    for AutoGluon 1.5.0, even though AWS ECR only publishes py312 images
    for that version. For ag_version=1.5/py_version=py312 specifically,
    resolve the registry/repository via a working py_version first, then
    substitute the real, confirmed-existing ECR tag.

    image_scope must be "training" or "inference" — these are two distinct
    ECR repositories (autogluon-training / autogluon-inference). Only the
    inference image has the real serving entrypoint; using the training
    image to serve a real-time endpoint or batch transform job produces a
    container that never passes the ping health check (verified during
    Task 10 e2e testing — the training image's default CMD does not start
    a model server at all).
    """
    if ag_version == "1.5" and py_version == "py312":
        base_uri = image_uris.retrieve(
            "autogluon", region=region, version=ag_version, py_version="py311",
            image_scope=image_scope, instance_type=instance_type,
        )
        registry_and_repo = base_uri.split(":")[0]
        return f"{registry_and_repo}:1.5-cpu-py312-ubuntu22.04-v1"
    return image_uris.retrieve(
        "autogluon", region=region, version=ag_version, py_version=py_version,
        image_scope=image_scope, instance_type=instance_type,
    )


def _resolve_config_path(config_file):
    """Locate a config/*.yaml file across both install modes this pipeline runs under.

    Editable installs (local dev) resolve config/ as a sibling of the
    installed pipelines/ package via __file__. Non-editable installs (what
    CodeBuild's `pip install --force-reinstall .` produces) don't package
    config/ as package data, so __file__-relative resolution fails there;
    CodeBuild always invokes run-pipeline from the checked-out source root,
    where config/ is a real sibling directory on disk, so cwd-relative
    resolution covers that case.
    """
    build_root = os.path.dirname(os.path.dirname(BASE_DIR))
    package_relative = os.path.join(build_root, "config", config_file)
    if os.path.exists(package_relative):
        return package_relative

    cwd_relative = os.path.join(os.getcwd(), "config", config_file)
    if os.path.exists(cwd_relative):
        return cwd_relative

    raise FileNotFoundError(
        f"Could not find config file {config_file!r} at {package_relative!r} "
        f"or {cwd_relative!r}. This can happen after a non-editable `pip "
        "install .` (config/ is not packaged as package data) run from "
        "somewhere other than the repo checkout root — run from the "
        "checkout root, or use an editable install (`pip install -e .`)."
    )


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="AutoMLModels",
    pipeline_name="AutoMLPipeline",
    base_job_prefix="AutoML",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.2xlarge",
    # NOTE: sagemaker==3.16.0's bundled image_uri_config/autogluon.json only
    # validates py311 for AutoGluon training version 1.5.0, while AWS DLC's ECR
    # repository (763104351884.dkr.ecr.<region>.amazonaws.com/autogluon-training)
    # publishes only py312 tags for 1.5.0 (no py311 variant exists). This repo's
    # standard is AutoGluon 1.5/py312 (matches the other four experiments), so
    # _retrieve_autogluon_image() below works around the stale SDK
    # config directly instead of downgrading the target version.
    ag_version="1.5",
    py_version="py312",
    config_file="tabular.yaml",
    sagemaker_project_name=None,
):
    """Build the AutoGluon AutoML pipeline: Preprocess -> Train -> Evaluate -> Condition -> Register."""
    import yaml

    sagemaker_session = Session()
    pipeline_session = PipelineSession()

    if role is None:
        role = get_execution_role()
    if default_bucket is None:
        default_bucket = sagemaker_session.default_bucket()

    # Read eval_metric from the local config file (uploaded to S3 by the buildspec
    # before this pipeline is built) so the ConditionStep's JsonGet knows which key
    # to read out of evaluation.json.
    config_path = _resolve_config_path(config_file)
    with open(config_path) as f:
        eval_metric = yaml.safe_load(f).get("eval_metric", "roc_auc")

    s3_prefix = f"s3://{default_bucket}/{base_job_prefix}/pipeline"
    config_s3_uri = f"{s3_prefix}/config/"

    param_input_data_uri = ParameterString(
        name="InputDataUri", default_value=f"s3://{default_bucket}/{base_job_prefix}/raw/"
    )
    param_training_instance_type = ParameterString(name="TrainingInstanceType", default_value=training_instance_type)
    param_model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    param_metric_threshold = ParameterFloat(name="MetricThreshold", default_value=0.75)

    ag_training_image = _retrieve_autogluon_image(
        region, ag_version, py_version, training_instance_type, image_scope="training")
    # Used only for RegisterModel: the model package's inference container must be the
    # autogluon-inference image (it has the real serving entrypoint), not autogluon-training —
    # using the training image to serve real-time/batch inference silently fails the endpoint's
    # ping health check (see _retrieve_autogluon_image's docstring).
    ag_inference_image = _retrieve_autogluon_image(
        region, ag_version, py_version, "ml.m5.xlarge", image_scope="inference")
    sklearn_image = image_uris.retrieve("sklearn", region=region, version="1.2-1")

    # -- Step 1: Preprocess --
    preprocessor = ScriptProcessor(
        image_uri=sklearn_image, role=role, command=["python3"],
        instance_type=processing_instance_type, instance_count=1,
        base_job_name=f"{base_job_prefix}-preprocess", sagemaker_session=pipeline_session,
    )
    step_preprocess = ProcessingStep(
        name="PreprocessData",
        step_args=preprocessor.run(
            code=os.path.join(BASE_DIR, "preprocess.py"),
            inputs=[ProcessingInput(input_name="input", s3_input=ProcessingS3Input(
                s3_uri=param_input_data_uri, local_path="/opt/ml/processing/input", s3_data_type="S3Prefix"))],
            outputs=[
                ProcessingOutput(output_name="train", s3_output=ProcessingS3Output(
                    s3_uri=f"{s3_prefix}/processed/train/", local_path="/opt/ml/processing/train", s3_upload_mode="EndOfJob")),
                ProcessingOutput(output_name="test", s3_output=ProcessingS3Output(
                    s3_uri=f"{s3_prefix}/processed/test/", local_path="/opt/ml/processing/test", s3_upload_mode="EndOfJob")),
            ],
        ),
    )

    # -- Step 2: Train --
    trainer = ModelTrainer(
        training_image=ag_training_image, role=role,
        source_code=SourceCode(source_dir=BASE_DIR, entry_script="train.py"),
        compute=Compute(instance_type=param_training_instance_type, instance_count=1,
                         volume_size_in_gb=100, keep_alive_period_in_seconds=0),
        output_data_config=OutputDataConfig(s3_output_path=f"{s3_prefix}/model/"),
        base_job_name=f"{base_job_prefix}-train",
        stopping_condition=StoppingCondition(max_runtime_in_seconds=7200),
        sagemaker_session=pipeline_session,
    )
    step_train = TrainingStep(
        name="TrainAutoGluon",
        step_args=trainer.train(input_data_config=[
            {"channel_name": "train", "data_source": {"s3_data_source": {
                "s3_uri": step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                "s3_data_type": "S3Prefix"}}},
            {"channel_name": "test", "data_source": {"s3_data_source": {
                "s3_uri": step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                "s3_data_type": "S3Prefix"}}},
            {"channel_name": "config", "data_source": {"s3_data_source": {
                "s3_uri": config_s3_uri, "s3_data_type": "S3Prefix"}}},
        ]),
    )

    # -- Step 3: Evaluate --
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    evaluator = ScriptProcessor(
        image_uri=ag_training_image, role=role, command=["python3"],
        instance_type=processing_instance_type, instance_count=1,
        base_job_name=f"{base_job_prefix}-evaluate", sagemaker_session=pipeline_session,
    )
    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        step_args=evaluator.run(
            code=os.path.join(BASE_DIR, "evaluate.py"),
            inputs=[
                ProcessingInput(input_name="model", s3_input=ProcessingS3Input(
                    s3_uri=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                    local_path="/opt/ml/processing/model", s3_data_type="S3Prefix")),
                ProcessingInput(input_name="test", s3_input=ProcessingS3Input(
                    s3_uri=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    local_path="/opt/ml/processing/test", s3_data_type="S3Prefix")),
                ProcessingInput(input_name="config", s3_input=ProcessingS3Input(
                    s3_uri=config_s3_uri, local_path="/opt/ml/processing/config", s3_data_type="S3Prefix")),
            ],
            outputs=[ProcessingOutput(output_name="evaluation", s3_output=ProcessingS3Output(
                s3_uri=f"{s3_prefix}/evaluation/", local_path="/opt/ml/processing/evaluation", s3_upload_mode="EndOfJob"))],
        ),
        property_files=[evaluation_report],
    )

    # -- Step 4: Register (via ModelBuilder — see module docstring) --
    # The AutoGluon inference DLC uses TorchServe and expects code/<entry point> inside the
    # model archive (same requirement documented in
    # 1-tabular-classification/2-inference/deploy.ipynb, which repacks the model tarball with
    # serve.py by hand). AutoGluon's own training output only contains predictor artifacts
    # (*.pkl), not a code/ dir, so without this the container falls back to its default
    # PyTorch handler and crashes on load ("Exactly one .pth or .pt file is required for
    # PyTorch models: []") — confirmed directly against a deployed endpoint during Task 10 e2e
    # testing.
    #
    # The actual code/serve.py packaging happens in train.py (it copies serve.py from
    # SM_CHANNEL_CODE, where ModelTrainer mounts this training job's own SourceCode at
    # /opt/ml/input/data/code, into {model_dir}/code/serve.py so it's included in the
    # model.tar.gz SageMaker auto-uploads from SM_MODEL_DIR) — NOT via ModelBuilder/ModelStep's
    # "runtime repack"
    # mechanism, which looks like the natural SDK v3 fit here (pass source_code=SourceCode(...)
    # to ModelBuilder, matching ModelTrainer's own pattern above) but is silently a no-op:
    # ModelStep._append_repack_model_step() only inserts a _RepackModelStep when
    # isinstance(self._model, sagemaker.core.resources.Model) — and self._model is actually the
    # ModelBuilder instance itself (see @runnable_by_pipeline's
    # init_model_step_arguments(self_instance) in sagemaker.core.workflow.pipeline_context),
    # which is not a Model subclass, so the isinstance check fails and the method returns early
    # with "No models to repack" logged (not raised) — confirmed directly by reading
    # sagemaker-serve==1.16.0 / sagemaker-mlops==1.16.0 source and by observing a model package
    # with SAGEMAKER_PROGRAM=serve.py / SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code correctly
    # set in its Environment, yet an unmodified model.tar.gz with no code/ dir at all (verified
    # by downloading and inspecting the actual S3 artifact). source_code is still passed to
    # ModelBuilder here anyway because it's what sets those Environment variables correctly on
    # the registered container — train.py's copy just has to independently place the file where
    # those variables say it will be.
    model_builder = ModelBuilder(
        image_uri=ag_inference_image,
        s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role_arn=role,
        sagemaker_session=pipeline_session,
        source_code=SourceCode(source_dir=BASE_DIR, entry_script="serve.py"),
    )
    step_register = ModelStep(
        name="RegisterModel",
        step_args=model_builder.register(
            content_types=["text/csv", "application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name=model_package_group_name,
            approval_status=param_model_approval_status,
        ),
    )

    # -- Step 5: Condition (metric gate) --
    step_fail = FailStep(
        name="AutoMLQualityGateFailed",
        error_message="Evaluation metric is below MetricThreshold. Model not registered.",
    )
    step_condition = ConditionStep(
        name="CheckEvaluationCondition",
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=step_evaluate.name,
                property_file=evaluation_report,
                json_path=f"metrics.{eval_metric}",
            ),
            right=param_metric_threshold,
        )],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[param_input_data_uri, param_training_instance_type, param_model_approval_status, param_metric_threshold],
        steps=[step_preprocess, step_train, step_evaluate, step_condition],
        sagemaker_session=pipeline_session,
    )
