import json
import os

import pytest

REAL_ROLE_ARN = "arn:aws:iam::859755744029:role/service-role/SageMaker-ExecutionRole-20250226T142578"


@pytest.mark.skipif(
    os.environ.get("SKIP_AWS_TESTS") == "1",
    reason="requires AWS credentials for IAM role validation during pipeline construction",
)
def test_get_pipeline_builds_valid_definition():
    from pipelines.automl.pipeline import get_pipeline

    pipeline = get_pipeline(
        region="us-east-1",
        role=REAL_ROLE_ARN,
        default_bucket="sagemaker-us-east-1-859755744029",
        model_package_group_name="TestAutoMLModels",
        pipeline_name="TestAutoMLPipeline",
    )
    definition = json.loads(pipeline.definition())
    step_names = [s["Name"] for s in definition["Steps"]]
    assert "PreprocessData" in step_names
    assert "TrainAutoGluon" in step_names
    assert "EvaluateModel" in step_names
    assert "CheckEvaluationCondition" in step_names
