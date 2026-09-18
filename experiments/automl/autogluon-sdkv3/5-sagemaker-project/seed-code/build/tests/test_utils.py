from pipelines._utils import convert_struct, get_pipeline_driver


def test_convert_struct_none_returns_empty_dict():
    assert convert_struct(None) == {}


def test_convert_struct_parses_dict_literal():
    assert convert_struct("{'region': 'us-east-1'}") == {"region": "us-east-1"}


def test_get_pipeline_driver_imports_and_calls_get_pipeline(monkeypatch, tmp_path):
    import sys
    import types

    fake_module = types.ModuleType("fake_pipeline_module")
    fake_module.get_pipeline = lambda **kwargs: {"called_with": kwargs}
    sys.modules["fake_pipeline_module"] = fake_module

    result = get_pipeline_driver("fake_pipeline_module", "{'region': 'us-east-1'}")
    assert result == {"called_with": {"region": "us-east-1"}}
