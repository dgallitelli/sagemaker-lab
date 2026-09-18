import sys

sys.path.insert(0, "pipelines/automl")


def test_predictor_class_for_task_type_tabular():
    import evaluate as evaluate_module

    assert evaluate_module.predictor_class_for_task_type("tabular") == "TabularPredictor"


def test_predictor_class_for_task_type_timeseries():
    import evaluate as evaluate_module

    assert evaluate_module.predictor_class_for_task_type("timeseries") == "TimeSeriesPredictor"


def test_predictor_class_for_task_type_multimodal():
    import evaluate as evaluate_module

    assert evaluate_module.predictor_class_for_task_type("multimodal") == "MultiModalPredictor"


def test_predictor_class_for_task_type_unknown_raises():
    import evaluate as evaluate_module

    try:
        evaluate_module.predictor_class_for_task_type("unknown")
        assert False, "expected ValueError"
    except ValueError:
        pass
