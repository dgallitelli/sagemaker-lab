import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, "pipelines/automl")


def test_build_predictor_tabular_uses_tabular_predictor():
    with patch.dict(sys.modules, {"autogluon.tabular": MagicMock()}):
        import importlib

        import train as train_module

        importlib.reload(train_module)
        config = {"task_type": "tabular", "label": "class", "eval_metric": "roc_auc"}
        predictor_cls, kwargs = train_module.build_predictor_args(config, model_dir="/tmp/model")
        assert predictor_cls == "TabularPredictor"
        assert kwargs["label"] == "class"
        assert kwargs["eval_metric"] == "roc_auc"


def test_build_predictor_timeseries_uses_timeseries_predictor():
    import train as train_module

    config = {
        "task_type": "timeseries",
        "eval_metric": "MASE",
        "id_column": "item_id",
        "timestamp_column": "timestamp",
        "prediction_length": 84,
    }
    predictor_cls, kwargs = train_module.build_predictor_args(config, model_dir="/tmp/model")
    assert predictor_cls == "TimeSeriesPredictor"
    assert kwargs["prediction_length"] == 84


def test_build_predictor_multimodal_uses_multimodal_predictor():
    import train as train_module

    config = {"task_type": "multimodal", "label": "y", "eval_metric": "roc_auc"}
    predictor_cls, kwargs = train_module.build_predictor_args(config, model_dir="/tmp/model")
    assert predictor_cls == "MultiModalPredictor"
    assert kwargs["label"] == "y"


def test_build_predictor_unknown_task_type_raises():
    import train as train_module

    try:
        train_module.build_predictor_args({"task_type": "unknown"}, model_dir="/tmp/model")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "unknown" in str(e)
