#!/usr/bin/env python3
"""Dependency-free structural checks for the SageMaker Lab monorepo."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
FENCED_CODE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
MARKDOWN_LINK = re.compile(r"\[[^\]]*]\(([^)]+)\)")
EXPECTED_IMPORTS = (
    (
        "gemma4-unsloth-sagemaker",
        "experiments/llm-training/gemma4-unsloth",
        "6b1f0af7f4214fa5cc38cae9ca7c3879e025a50c",
    ),
    (
        "qwen35-sft-sagemaker",
        "experiments/llm-training/qwen35-sft",
        "edccd65f3e2ee52f22d0249278dad2b643b2da66",
    ),
    (
        "sagemaker-autogluon-sdkv3",
        "experiments/automl/autogluon-sdkv3",
        "a902956e2201d4d1b09ac04c00d314c9d6082288",
    ),
    (
        "sagemaker-ntl-detection-with-xgboost-and-chronos",
        "experiments/time-series/ntl-xgboost-chronos",
        "b4a61d08bc77d2191e210e668bfae9c1b1424959",
    ),
    (
        "sagemaker-sdk-v3-xgboost-example",
        "experiments/tabular/xgboost-sdkv3",
        "cc03e302418d7d877b64c1649d231282cf1790f9",
    ),
    (
        "sagemaker-splade-embeddings",
        "experiments/embeddings/splade",
        "67be92a1fc5128ec01baca1f716f9a20291732a1",
    ),
    (
        "sagemaker-tabpfn3-experiments",
        "experiments/tabular/tabpfn3",
        "9b717ba43b1dc6367a3dd486f4a689b35711f0ca",
    ),
)
EXPECTED_EXTRA_HISTORY = (
    (
        "qwen35-sft-sagemaker pull request #7",
        "35f23d6f5b54057b4eed8f659732fa3ae0fd64a5",
    ),
)


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    return [ROOT / line for line in output.splitlines() if line]


def check_python(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix != ".py":
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{path.relative_to(ROOT)}: invalid Python: {exc}")


def check_notebooks(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix != ".ipynb":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{path.relative_to(ROOT)}: invalid notebook JSON: {exc}")


def check_markdown_links(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix != ".md":
            continue
        text = FENCED_CODE.sub("", path.read_text(encoding="utf-8"))
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group(1).strip().strip("<>")
            if not target or target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(
                    f"{path.relative_to(ROOT)}: broken relative link: {match.group(1)}"
                )


def is_ancestor(commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def check_migration_history(errors: list[str]) -> None:
    for source, destination, commit in EXPECTED_IMPORTS:
        if not (ROOT / destination).is_dir():
            errors.append(f"{source}: missing destination directory: {destination}")
        if not is_ancestor(commit):
            errors.append(f"{source}: imported history is not reachable from HEAD")

    for source, commit in EXPECTED_EXTRA_HISTORY:
        if not is_ancestor(commit):
            errors.append(f"{source}: preserved history is not reachable from HEAD")


def main() -> int:
    files = tracked_files()
    errors: list[str] = []
    check_python(files, errors)
    check_notebooks(files, errors)
    check_markdown_links(files, errors)
    check_migration_history(errors)

    if errors:
        print("\n".join(errors))
        return 1

    python_count = sum(path.suffix == ".py" for path in files)
    notebook_count = sum(path.suffix == ".ipynb" for path in files)
    markdown_count = sum(path.suffix == ".md" for path in files)
    print(
        "structural checks passed: "
        f"{python_count} Python files, "
        f"{notebook_count} notebooks, "
        f"{markdown_count} Markdown files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
