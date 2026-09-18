import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, "..")


def test_extend_config_adds_required_parameters():
    import build

    args = SimpleNamespace(
        sagemaker_project_name="test-proj",
        sagemaker_project_id="p-abc123",
        model_execution_role="arn:aws:iam::123456789012:role/exec-role",
        s3_bucket="test-bucket",
    )
    stage_config = {"Parameters": {"StageName": "staging", "EndpointInstanceCount": "1"}}

    with patch("build.get_pipeline_custom_tags", return_value={}):
        result = build.extend_config(args, "arn:aws:sagemaker:::model-package/foo/1", stage_config)

    assert result["Parameters"]["ModelPackageName"] == "arn:aws:sagemaker:::model-package/foo/1"
    assert result["Parameters"]["SageMakerProjectName"] == "test-proj"
    assert result["Parameters"]["ModelExecutionRoleArn"] == "arn:aws:iam::123456789012:role/exec-role"
    assert "datacapture-staging" in result["Parameters"]["DataCaptureUploadPath"]


def test_extend_config_missing_stagename_raises():
    import build

    args = SimpleNamespace(sagemaker_project_name="x", sagemaker_project_id="y",
                            model_execution_role="z", s3_bucket="b")
    try:
        build.extend_config(args, "arn", {"Parameters": {}})
        assert False, "expected Exception"
    except Exception as e:
        assert "StageName" in str(e)


def test_get_approved_package_returns_latest_arn():
    import build

    fake_response = {
        "ModelPackageSummaryList": [{"ModelPackageArn": "arn:aws:sagemaker:::model-package/foo/2"}]
    }
    with patch.object(build, "sm_client") as mock_client:
        mock_client.list_model_packages.return_value = fake_response
        arn = build.get_approved_package("foo")
    assert arn == "arn:aws:sagemaker:::model-package/foo/2"
