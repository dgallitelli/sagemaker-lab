"""SageMaker Pipeline for Gas Lift Optimization Model

This pipeline automates the ML workflow:
1. Preprocess - Load and split data
2. Train - Train XGBoost model (reuses training/train.py)
3. Evaluate - Calculate R2/RMSE metrics
4. Quality Gate - Register model only if R2 >= threshold

SDK v3 compatible (Feb 2026).
"""

import os

# Monkey-patch SDK v3.4.1 bug: get_training_code_hash passes requirements (str)
# as dependencies (expects List[str]). This patch wraps the original function.
import sagemaker.core.workflow.utilities as _workflow_utils

_original_get_training_code_hash = _workflow_utils.get_training_code_hash


def _patched_get_training_code_hash(entry_point, source_dir, dependencies):
    """Patched version that handles string/None dependencies."""
    if dependencies is None:
        dependencies = []
    elif isinstance(dependencies, str):
        dependencies = [dependencies] if dependencies else []
    return _original_get_training_code_hash(entry_point, source_dir, dependencies)


_workflow_utils.get_training_code_hash = _patched_get_training_code_hash

# SDK v3 workflow imports (mlops module)
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.fail_step import FailStep
from sagemaker.mlops.workflow.model_step import ModelStep

# SDK v3 core workflow imports
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.core.workflow.parameters import ParameterInteger, ParameterFloat, ParameterString
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.core.workflow.functions import JsonGet

# SDK v3 training imports
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import InputData, Compute, SourceCode

# SDK v3 processing imports
from sagemaker.core.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.core.shapes.shapes import ProcessingS3Input, ProcessingS3Output

# SDK v3 model/session imports
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.image_uris import retrieve


def get_pipeline(
    region: str,
    role: str = None,
    default_bucket: str = None,
    pipeline_name: str = "GasLiftOptimizationPipeline",
    base_job_prefix: str = "gas-lift-opt",
    model_package_group_name: str = "GasLiftOptimizationModels",
) -> Pipeline:
    """Create the SageMaker Pipeline.

    Args:
        region: AWS region
        role: IAM role ARN (if None, uses get_execution_role())
        default_bucket: S3 bucket for artifacts (if None, uses session default)
        pipeline_name: Name of the pipeline
        base_job_prefix: Prefix for job names
        model_package_group_name: Model Registry group name

    Returns:
        Pipeline: Configured SageMaker Pipeline
    """
    # Session setup
    sagemaker_session = Session()
    pipeline_session = PipelineSession()

    if role is None:
        role = get_execution_role()
    if default_bucket is None:
        default_bucket = sagemaker_session.default_bucket()

    # Container image (SKLearn 1.2-1 for all steps)
    sklearn_image = retrieve(
        framework="sklearn",
        version="1.2-1",
        region=region,
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )

    # Pipeline Parameters
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.xlarge",
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.xlarge",
    )
    input_data_uri = ParameterString(
        name="InputDataUri",
        default_value=f"s3://{default_bucket}/{base_job_prefix}/data/",
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",
    )

    # XGBoost hyperparameters
    max_depth = ParameterInteger(name="MaxDepth", default_value=6)
    n_estimators = ParameterInteger(name="NEstimators", default_value=100)
    eta = ParameterFloat(name="Eta", default_value=0.1)
    subsample = ParameterFloat(name="Subsample", default_value=0.8)

    # Quality gate threshold
    r2_threshold = ParameterFloat(name="R2Threshold", default_value=0.85)

    # Cache configuration (30 days)
    cache_config = CacheConfig(enable_caching=True, expire_after="P30D")

    # ==========================================================================
    # Step 1: Preprocessing
    # ==========================================================================
    script_processor = ScriptProcessor(
        image_uri=sklearn_image,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-preprocess",
        role=role,
        sagemaker_session=pipeline_session,
    )

    preprocessing_step = ProcessingStep(
        name="Preprocess",
        step_args=script_processor.run(
            code="preprocessing.py",
            inputs=[
                ProcessingInput(
                    input_name="input",
                    s3_input=ProcessingS3Input(
                        s3_uri=input_data_uri,
                        s3_data_type="S3Prefix",
                        local_path="/opt/ml/processing/input",
                    ),
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{default_bucket}/{base_job_prefix}/processed/train",
                        s3_upload_mode="EndOfJob",
                        local_path="/opt/ml/processing/train",
                    ),
                ),
                ProcessingOutput(
                    output_name="test",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{default_bucket}/{base_job_prefix}/processed/test",
                        s3_upload_mode="EndOfJob",
                        local_path="/opt/ml/processing/test",
                    ),
                ),
            ],
        ),
        cache_config=cache_config,
    )

    # ==========================================================================
    # Step 2: Training (reuses existing training/train.py)
    # ==========================================================================
    # Note: requirements not specified here due to SDK v3.4.1 bug with code hashing
    # The requirements.txt in source_dir is still used at runtime
    source_code = SourceCode(
        source_dir="../training",
        entry_script="train.py",
    )

    compute = Compute(
        instance_type=training_instance_type,
        instance_count=1,
    )

    model_trainer = ModelTrainer(
        role=role,
        training_image=sklearn_image,
        source_code=source_code,
        compute=compute,
        base_job_name=f"{base_job_prefix}-train",
        hyperparameters={
            "n-estimators": n_estimators,
            "max-depth": max_depth,
            "learning-rate": eta,
            "subsample": subsample,
            "target-column": "production",
            "test-size": 0.0,  # No internal split - we already have train/test
        },
        sagemaker_session=pipeline_session,
    )

    training_step = TrainingStep(
        name="Train",
        step_args=model_trainer.train(
            input_data_config=[
                InputData(
                    channel_name="train",
                    data_source=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                        "train"
                    ].S3Output.S3Uri,
                )
            ]
        ),
        cache_config=cache_config,
    )

    # ==========================================================================
    # Step 3: Evaluation
    # ==========================================================================
    evaluation_processor = ScriptProcessor(
        image_uri=sklearn_image,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-evaluate",
        role=role,
        sagemaker_session=pipeline_session,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluation_step = ProcessingStep(
        name="Evaluate",
        step_args=evaluation_processor.run(
            code="evaluation.py",
            inputs=[
                ProcessingInput(
                    input_name="model",
                    s3_input=ProcessingS3Input(
                        s3_uri=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                        s3_data_type="S3Prefix",
                        local_path="/opt/ml/processing/model",
                    ),
                ),
                ProcessingInput(
                    input_name="test",
                    s3_input=ProcessingS3Input(
                        s3_uri=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                            "test"
                        ].S3Output.S3Uri,
                        s3_data_type="S3Prefix",
                        local_path="/opt/ml/processing/test",
                    ),
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{default_bucket}/{base_job_prefix}/evaluation",
                        s3_upload_mode="EndOfJob",
                        local_path="/opt/ml/processing/evaluation",
                    ),
                ),
            ],
        ),
        property_files=[evaluation_report],
        cache_config=cache_config,
    )

    # ==========================================================================
    # Step 4: Model Registration (conditional)
    # ==========================================================================
    model_builder = ModelBuilder(
        image_uri=sklearn_image,
        s3_model_data_url=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role_arn=role,
        sagemaker_session=pipeline_session,
    )

    register_step = ModelStep(
        name="RegisterModel",
        step_args=model_builder.register(
            content_types=["text/csv", "application/json"],
            response_types=["application/json"],
            inference_instances=["ml.t3.medium", "ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status,
        ),
    )

    # ==========================================================================
    # Step 5: Fail Step (if quality gate fails)
    # ==========================================================================
    fail_step = FailStep(
        name="QualityGateFailed",
        error_message="Model R2 score is below the threshold. Model not registered.",
    )

    # ==========================================================================
    # Step 6: Condition Step (quality gate)
    # ==========================================================================
    condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="regression_metrics.r2.value",
        ),
        right=r2_threshold,
    )

    condition_step = ConditionStep(
        name="CheckR2Threshold",
        conditions=[condition],
        if_steps=[register_step],
        else_steps=[fail_step],
    )

    # ==========================================================================
    # Create Pipeline
    # ==========================================================================
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            training_instance_type,
            input_data_uri,
            model_approval_status,
            max_depth,
            n_estimators,
            eta,
            subsample,
            r2_threshold,
        ],
        steps=[preprocessing_step, training_step, evaluation_step, condition_step],
        sagemaker_session=pipeline_session,
    )

    return pipeline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--role", type=str, default=None)
    args = parser.parse_args()

    pipeline = get_pipeline(region=args.region, role=args.role)
    print(pipeline.definition())
