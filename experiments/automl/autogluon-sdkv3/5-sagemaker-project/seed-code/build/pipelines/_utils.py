"""Utilities for the SageMaker Pipeline CLI."""
from __future__ import annotations

import ast


def get_pipeline_driver(module_name: str, passed_args: str | None = None):
    """Import `module_name` and call its `get_pipeline(**kwargs)` function.

    Args:
        module_name: Dotted module path of a pipeline module exposing `get_pipeline`.
        passed_args: Optional Python-literal dict string of keyword arguments.

    Returns:
        The constructed `sagemaker.mlops.workflow.pipeline.Pipeline`.
    """
    imported = __import__(module_name, fromlist=["get_pipeline"])
    kwargs = convert_struct(passed_args)
    return imported.get_pipeline(**kwargs)


def convert_struct(str_struct: str | None = None) -> dict:
    """Parse a Python-literal dict string into a dict, or return {} if None."""
    return ast.literal_eval(str_struct) if str_struct else {}


def get_pipeline_custom_tags(module_name: str, args: str | None, tags: list) -> list:
    """Call the pipeline module's `get_pipeline_custom_tags`, if it defines one."""
    try:
        imported = __import__(module_name, fromlist=["get_pipeline_custom_tags"])
        kwargs = convert_struct(args)
        return imported.get_pipeline_custom_tags(
            tags, kwargs["region"], kwargs["sagemaker_project_name"]
        )
    except Exception as e:  # noqa: BLE001 — tagging is best-effort
        print(f"Error getting project tags: {e}")
    return tags
