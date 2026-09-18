import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, "lambda")


def test_handler_creates_transform_job_and_waits():
    import run_transform

    fake_event = {
        "model_name": "test-model",
        "transform_job_name": "test-transform-20260722",
        "input_s3_uri": "s3://bucket/input/",
        "output_s3_uri": "s3://bucket/output/",
        "instance_type": "ml.m5.xlarge",
        "instance_count": 1,
        "content_type": "text/csv",
    }

    mock_client = MagicMock()
    mock_client.describe_transform_job.return_value = {"TransformJobStatus": "Completed"}

    with patch.object(run_transform, "sm_client", mock_client):
        result = run_transform.handler(fake_event, None)

    mock_client.create_transform_job.assert_called_once()
    call_kwargs = mock_client.create_transform_job.call_args.kwargs
    assert call_kwargs["TransformJobName"] == "test-transform-20260722"
    assert call_kwargs["ModelName"] == "test-model"
    assert result["status"] == "Completed"


def test_handler_raises_on_failed_transform_job():
    import run_transform

    fake_event = {
        "model_name": "test-model",
        "transform_job_name": "test-transform-fail",
        "input_s3_uri": "s3://bucket/input/",
        "output_s3_uri": "s3://bucket/output/",
        "instance_type": "ml.m5.xlarge",
        "instance_count": 1,
        "content_type": "text/csv",
    }

    mock_client = MagicMock()
    mock_client.describe_transform_job.return_value = {
        "TransformJobStatus": "Failed",
        "FailureReason": "boom",
    }

    with patch.object(run_transform, "sm_client", mock_client):
        try:
            run_transform.handler(fake_event, None)
            assert False, "expected Exception"
        except Exception as e:
            assert "boom" in str(e)


def test_handler_falls_back_to_model_name_env_var(monkeypatch):
    import run_transform

    monkeypatch.setenv("MODEL_NAME", "env-fallback-model")

    fake_event = {
        "model_name": None,
        "transform_job_name": "test-transform-fallback",
        "input_s3_uri": "s3://bucket/input/",
        "output_s3_uri": "s3://bucket/output/",
        "instance_type": "ml.m5.xlarge",
        "instance_count": 1,
        "content_type": "text/csv",
    }

    mock_client = MagicMock()
    mock_client.describe_transform_job.return_value = {"TransformJobStatus": "Completed"}

    with patch.object(run_transform, "sm_client", mock_client):
        run_transform.handler(fake_event, None)

    call_kwargs = mock_client.create_transform_job.call_args.kwargs
    assert call_kwargs["ModelName"] == "env-fallback-model"
