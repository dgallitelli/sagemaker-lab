# 5-sagemaker-project (AutoGluon AutoML on SageMaker Projects) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `5-sagemaker-project/`, a new experiment demonstrating SageMaker Projects as a CI/CD "MLOps platform" for AutoGluon AutoML — a custom CFN template (stored in S3, registered via `create_project`'s `TemplateProviders`) that provisions a Build pipeline (SageMaker Pipeline: Preprocess → Train → Evaluate → Condition → Register) and two Deploy pipelines (real-time endpoint, scheduled batch transform), both gated staging → manual approval → prod.

**Architecture:** One `AWS::CloudFormation` template (`cfn-templates/project-template.yaml`) provisions 3 CodePipelines wired to 2 GitHub repos via an existing CodeConnections connection. Seed code lives under `5-sagemaker-project/seed-code/{build,deploy}/` in this repo and gets pushed into the GitHub repos by a setup notebook. Model registration uses `ModelBuilder(...).register(...)` → `ModelStep` (the correct v3 pattern — NOT `Model.register()`, which this repo's four existing pipelines mistakenly called; that method is `ABCMeta.register()`, unrelated to the SageMaker Model Registry, verified against `sagemaker==3.16.0` in this session).

**Tech Stack:** SageMaker Python SDK v3 (`sagemaker>=3.5.0,<4.0`, tested against `3.16.0`), boto3, AWS CodePipeline/CodeBuild/CloudFormation/EventBridge/Lambda, GitHub (via `gh` CLI + CodeConnections), AutoGluon (tabular/timeseries/multimodal).

## Global Constraints

- SDK v3 only — `from sagemaker.train import ModelTrainer`, `from sagemaker.core.resources import ...`, `from sagemaker.mlops.workflow...` (per `~/.claude/docs/sagemaker-v3.md`).
- AutoGluon DLC version 1.5, Python 3.12 (`py312`) — matches the rest of this repo (see root `README.md`).
- Model registration MUST use `ModelBuilder(image_uri=..., s3_model_data_url=..., role_arn=..., sagemaker_session=pipeline_session).register(...)` → `ModelStep`. Do NOT call `.register()` on a plain `sagemaker.core.resources.Model` instance — verified in this session to be `ABCMeta.register()`, a no-op virtual-subclass registration unrelated to the Model Registry.
- Pipeline hyperparameters passed as `ParameterString` (not `ParameterInteger`/`ParameterFloat` for values consumed as training hyperparameters) — SageMaker passes all hyperparameters as strings (existing repo convention, see root README "Key Considerations").
- No Model Monitor stage, no baseline/drift CodeBuild project, no `InServiceEndpointEventRule` — explicitly out of scope per the approved spec.
- No modification to any of the four existing experiments' notebooks or scripts — this is new, additive work only.
- IAM changes (Task 0) go through the account's existing CLI-first IaC workflow: validate with `aws iam` CLI, get explicit user confirmation before applying (per `~/.claude/rules/amazon-production-safety-do-not-delete.md` — this modifies a shared, account-wide service role).
- Spec location: `docs/superpowers/specs/2026-07-22-sagemaker-project-automl-design.md` (already approved).

---

## File Structure

```
5-sagemaker-project/
  README.md                              # Task 8
  0-project-setup/
    setup_project.ipynb                   # Task 7
  cfn-templates/
    project-template.yaml                 # Task 6
  seed-code/
    build/
      pipelines/
        __init__.py                       # Task 1
        _utils.py                         # Task 1 (adapted from petro-build)
        run_pipeline.py                   # Task 1 (adapted from petro-build)
        get_pipeline_definition.py        # Task 1 (adapted from petro-build)
        automl/
          __init__.py                     # Task 1
          pipeline.py                     # Task 5 (Preprocess->Train->Evaluate->Condition->Register)
          preprocess.py                   # Task 2 (tabular/Adult Census — working default)
          train.py                        # Task 3 (pluggable: tabular/timeseries/multimodal)
          evaluate.py                     # Task 4 (pluggable, same task_type switch)
      config/
        tabular.yaml                      # Task 2
        timeseries.yaml                   # Task 3 (reference only)
        multimodal.yaml                   # Task 3 (reference only)
      codebuild-buildspec.yml              # Task 5
      setup.py                             # Task 1
      setup.cfg                            # Task 1
    deploy/
      realtime/
        build.py                          # Task 6 (adapted from petro-deploy build.py)
        buildspec.yml                     # Task 6
        endpoint-config-template.yml      # Task 6 (adapted from petro-deploy)
        staging-config.json               # Task 6
        prod-config.json                  # Task 6
        serve.py                          # Task 6 (pluggable inference handler)
        test/
          test.py                         # Task 6 (adapted from petro-deploy test.py)
          buildspec.yml                   # Task 6
      batch/
        build.py                          # Task 7 (emits Model+Lambda+EventBridge CFN)
        buildspec.yml                     # Task 7
        batch-transform-template.yml      # Task 7
        staging-config.json               # Task 7
        prod-config.json                  # Task 7
        lambda/
          run_transform.py                # Task 7 (create TransformJob, wait, raise on failure)
        test/
          test.py                         # Task 7
          buildspec.yml                   # Task 7
```

Each script is self-contained and independently testable: `train.py`/`evaluate.py` take a
`--task-type` CLI arg / `task_type` config key and can be unit-tested per branch without AWS.
`pipeline.py` only assembles SDK v3 step objects (testable via `pipeline.definition()` — no AWS
calls). `build.py` scripts are testable by mocking `boto3.client("sagemaker")`.

---

### Task 0: Widen `AmazonSageMakerProjectsCloudformationServiceRolePolicy` for batch-deploy resources

**Files:**
- Modify (AWS, via CLI — no repo file): IAM policy `arn:aws:iam::859755744029:policy/AmazonSageMakerProjectsCloudformationServiceRolePolicy`
- Create: `5-sagemaker-project/cfn-templates/iam-policy-patch.json` (the exact statement added, checked into the repo for reproducibility/audit)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: the `AmazonSageMakerProjectsCloudformationRole` gains permission to create/delete a
  Lambda function matching `arn:aws:lambda:*:*:function:sagemaker-*`, an EventBridge rule
  matching `arn:aws:events:*:*:rule/sagemaker-*`, and an IAM role matching
  `arn:aws:iam::*:role/sagemaker-*-batch-transform-lambda`. Task 7's `batch-transform-template.yml`
  depends on this.

This is a shared, account-wide role — confirmed with the user this is the approved fix (widen
existing role, scoped by the `sagemaker-*` naming prefix already used elsewhere in this policy).

- [ ] **Step 1: Fetch and save the current policy document**

```bash
mkdir -p /tmp/sagemaker-project-iam
VID=$(aws iam get-policy --policy-arn arn:aws:iam::859755744029:policy/AmazonSageMakerProjectsCloudformationServiceRolePolicy --query 'Policy.DefaultVersionId' --output text)
aws iam get-policy-version \
  --policy-arn arn:aws:iam::859755744029:policy/AmazonSageMakerProjectsCloudformationServiceRolePolicy \
  --version-id "$VID" \
  --query 'PolicyVersion.Document' > /tmp/sagemaker-project-iam/current-policy.json
cat /tmp/sagemaker-project-iam/current-policy.json
```

Expected: JSON with 2 `Statement` entries (`sagemaker:*`-family actions, and `iam:PassRole` for
2 roles) — matches what was verified during design research.

- [ ] **Step 2: Write the patch file to the repo**

Create `5-sagemaker-project/cfn-templates/iam-policy-patch.json`:

```json
[
  {
    "Sid": "BatchTransformLambdaPermission",
    "Effect": "Allow",
    "Action": [
      "lambda:CreateFunction",
      "lambda:DeleteFunction",
      "lambda:GetFunction",
      "lambda:UpdateFunctionCode",
      "lambda:UpdateFunctionConfiguration",
      "lambda:AddPermission",
      "lambda:RemovePermission",
      "lambda:TagResource"
    ],
    "Resource": "arn:aws:lambda:*:*:function:sagemaker-*"
  },
  {
    "Sid": "BatchTransformEventsPermission",
    "Effect": "Allow",
    "Action": [
      "events:PutRule",
      "events:DeleteRule",
      "events:PutTargets",
      "events:RemoveTargets",
      "events:DescribeRule"
    ],
    "Resource": "arn:aws:events:*:*:rule/sagemaker-*"
  },
  {
    "Sid": "BatchTransformLambdaRolePermission",
    "Effect": "Allow",
    "Action": [
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:PutRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:GetRole",
      "iam:PassRole"
    ],
    "Resource": "arn:aws:iam::*:role/sagemaker-*-batch-transform-lambda"
  }
]
```

- [ ] **Step 3: Build the merged policy document and validate it with the CLI (dry run — no apply yet)**

```bash
python3 - <<'PYEOF'
import json

with open("/tmp/sagemaker-project-iam/current-policy.json") as f:
    current = json.load(f)
with open("5-sagemaker-project/cfn-templates/iam-policy-patch.json") as f:
    patch = json.load(f)

current["Statement"].extend(patch)

with open("/tmp/sagemaker-project-iam/merged-policy.json", "w") as f:
    json.dump(current, f, indent=2)

print(f"Merged policy has {len(current['Statement'])} statements")
PYEOF
cat /tmp/sagemaker-project-iam/merged-policy.json
```

Expected: 5 statements total (2 original + 3 new).

- [ ] **Step 4: Apply the new policy version (creates a new version, sets it as default)**

Confirm with the user before running — this modifies a shared, account-wide IAM policy.

```bash
aws iam create-policy-version \
  --policy-arn arn:aws:iam::859755744029:policy/AmazonSageMakerProjectsCloudformationServiceRolePolicy \
  --policy-document file:///tmp/sagemaker-project-iam/merged-policy.json \
  --set-as-default
```

Expected: `PolicyVersion` output with `"VersionId": "v2"`, `"IsDefaultVersion": true`.

- [ ] **Step 5: Verify with a policy simulation**

```bash
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::859755744029:role/service-role/AmazonSageMakerProjectsCloudformationRole \
  --action-names lambda:CreateFunction events:PutRule iam:CreateRole \
  --resource-arns "arn:aws:lambda:us-east-1:859755744029:function:sagemaker-test-batch-transform-lambda" "arn:aws:events:us-east-1:859755744029:rule/sagemaker-test" "arn:aws:iam::859755744029:role/sagemaker-test-batch-transform-lambda" \
  --query 'EvaluationResults[].{Action:EvalActionName,Decision:EvalDecision}'
```

Expected: all three actions show `"Decision": "allowed"`.

- [ ] **Step 6: Commit the patch file**

```bash
git add 5-sagemaker-project/cfn-templates/iam-policy-patch.json
git commit -m "Widen SageMaker Projects CFN role for batch-transform Lambda/EventBridge resources"
```

---

### Task 1: Build-repo Python package scaffold

**Files:**
- Create: `5-sagemaker-project/seed-code/build/pipelines/__init__.py`
- Create: `5-sagemaker-project/seed-code/build/pipelines/__version__.py`
- Create: `5-sagemaker-project/seed-code/build/pipelines/_utils.py`
- Create: `5-sagemaker-project/seed-code/build/pipelines/run_pipeline.py`
- Create: `5-sagemaker-project/seed-code/build/pipelines/get_pipeline_definition.py`
- Create: `5-sagemaker-project/seed-code/build/pipelines/automl/__init__.py`
- Create: `5-sagemaker-project/seed-code/build/setup.py`
- Create: `5-sagemaker-project/seed-code/build/setup.cfg`
- Create: `5-sagemaker-project/seed-code/build/.gitignore`
- Test: `5-sagemaker-project/seed-code/build/tests/test_utils.py`

**Interfaces:**
- Produces: `pipelines._utils.get_pipeline_driver(module_name: str, passed_args: str|None) -> Pipeline`
  and `pipelines._utils.convert_struct(str_struct: str|None) -> dict` — used by Task 5's
  `pipeline.py` (via its `get_pipeline(**kwargs)` entrypoint) and by
  `codebuild-buildspec.yml` (Task 5) through the `run-pipeline` console script.

This task adapts the proven `petro-build` package structure (verified working in this AWS
account) rather than inventing a new one.

- [ ] **Step 1: Create the package init and version files**

`5-sagemaker-project/seed-code/build/pipelines/__init__.py`:
```python
"""AutoGluon AutoML SageMaker Pipelines package."""
```

`5-sagemaker-project/seed-code/build/pipelines/__version__.py`:
```python
__title__ = "automl-pipelines"
__description__ = "AutoGluon AutoML SageMaker Pipeline for SageMaker Projects CI/CD"
__version__ = "0.1.0"
__author__ = "AutoGluon on SageMaker SDK v3 examples"
__author_email__ = ""
__license__ = "MIT"
__url__ = ""
```

`5-sagemaker-project/seed-code/build/pipelines/automl/__init__.py`:
```python
"""AutoGluon AutoML pipeline: Preprocess -> Train -> Evaluate -> Condition -> Register."""
```

- [ ] **Step 2: Create `_utils.py`**

`5-sagemaker-project/seed-code/build/pipelines/_utils.py`:
```python
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
```

- [ ] **Step 3: Write the failing test for `_utils.py`**

`5-sagemaker-project/seed-code/build/tests/test_utils.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it fails (package not importable yet without setup.py/setup.cfg)**

```bash
cd 5-sagemaker-project/seed-code/build
python3 -m pytest tests/test_utils.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'pipelines'` (no `setup.py`/install yet,
and `pipelines/_utils.py` step 2 above already exists but the package root isn't on `sys.path`
without an editable install).

- [ ] **Step 5: Add `setup.py` and `setup.cfg`, then install editable**

`5-sagemaker-project/seed-code/build/setup.py`:
```python
import os

import setuptools

about: dict = {}
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "pipelines", "__version__.py")) as f:
    exec(f.read(), about)

with open("README.md") if os.path.exists("README.md") else open(os.devnull) as f:
    long_description = f.read()

setuptools.setup(
    name=about["__title__"],
    description=about["__description__"],
    version=about["__version__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["sagemaker>=3.5.0,<4.0", "boto3", "pyyaml"],
    extras_require={"test": ["pytest", "pytest-cov"]},
    entry_points={
        "console_scripts": [
            "get-pipeline-definition=pipelines.get_pipeline_definition:main",
            "run-pipeline=pipelines.run_pipeline:main",
        ]
    },
)
```

`5-sagemaker-project/seed-code/build/setup.cfg`:
```ini
[tool:pytest]
addopts = -vv
testpaths = tests

[metadata]
description-file = README.md
```

`5-sagemaker-project/seed-code/build/.gitignore`:
```
__pycache__/
*.egg-info/
.pytest_cache/
```

```bash
cd 5-sagemaker-project/seed-code/build
pip install -e . 2>&1 | tail -5
```

Expected: `Successfully installed automl-pipelines-0.1.0`.

- [ ] **Step 6: Run test to verify it passes**

```bash
cd 5-sagemaker-project/seed-code/build
python3 -m pytest tests/test_utils.py -v
```

Expected: `3 passed`.

- [ ] **Step 7: Create `run_pipeline.py` and `get_pipeline_definition.py`**

`5-sagemaker-project/seed-code/build/pipelines/get_pipeline_definition.py`:
```python
"""CLI to print/save a pipeline's JSON definition."""
from __future__ import annotations

import argparse
import sys
import traceback

from pipelines._utils import get_pipeline_driver


def main() -> None:
    parser = argparse.ArgumentParser("Gets the pipeline definition for the pipeline script.")
    parser.add_argument("-n", "--module-name", dest="module_name", type=str, required=True)
    parser.add_argument("-f", "--file-name", dest="file_name", type=str, default=None)
    parser.add_argument("-kwargs", "--kwargs", dest="kwargs", default=None)
    args = parser.parse_args()

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        content = pipeline.definition()
        if args.file_name:
            with open(args.file_name, "w") as f:
                f.write(content)
        else:
            print(content)
    except Exception as e:  # noqa: BLE001
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

`5-sagemaker-project/seed-code/build/pipelines/run_pipeline.py`:
```python
"""CLI to create/update and run a SageMaker Pipeline."""
from __future__ import annotations

import argparse
import json
import sys
import traceback

from pipelines._utils import convert_struct, get_pipeline_custom_tags, get_pipeline_driver


def main() -> None:
    parser = argparse.ArgumentParser("Creates/updates and runs the pipeline for the pipeline script.")
    parser.add_argument("-n", "--module-name", dest="module_name", type=str, required=True)
    parser.add_argument("-kwargs", "--kwargs", dest="kwargs", default=None)
    parser.add_argument("-role-arn", "--role-arn", dest="role_arn", type=str, required=True)
    parser.add_argument("-description", "--description", dest="description", type=str, default=None)
    parser.add_argument("-tags", "--tags", dest="tags", default=None)
    args = parser.parse_args()

    tags = convert_struct(args.tags)

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        print("###### Pipeline definition:")
        print(json.dumps(json.loads(pipeline.definition()), indent=2, sort_keys=True))

        all_tags = get_pipeline_custom_tags(args.module_name, args.kwargs, tags)
        upsert_response = pipeline.upsert(role_arn=args.role_arn, description=args.description, tags=all_tags)
        print(f"\n###### Upserted pipeline: {upsert_response}")

        execution = pipeline.start()
        print(f"\n###### Execution started: {execution.arn}")
        execution.wait(max_attempts=120, delay=60)
        print("\n###### Execution complete. Steps:")
        print(execution.list_steps())
    except Exception as e:  # noqa: BLE001
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Reinstall (picks up new modules) and re-run the full test suite**

```bash
cd 5-sagemaker-project/seed-code/build
pip install -e . 2>&1 | tail -3
python3 -m pytest tests/ -v
```

Expected: `3 passed` (no new tests yet for the CLI scripts — they're exercised end-to-end in
Task 5).

- [ ] **Step 9: Commit**

```bash
git add 5-sagemaker-project/seed-code/build/
git commit -m "Scaffold AutoML build-repo Python package (pipelines._utils, CLI entrypoints)"
```

---

### Task 2: Tabular preprocessing script + config (working default)

**Files:**
- Create: `5-sagemaker-project/seed-code/build/pipelines/automl/preprocess.py`
- Create: `5-sagemaker-project/seed-code/build/config/tabular.yaml`
- Test: `5-sagemaker-project/seed-code/build/tests/test_preprocess.py`

**Interfaces:**
- Produces: a script that, run as `python preprocess.py`, reads CSV/`.data` files from
  `/opt/ml/processing/input`, writes `train.csv`/`test.csv` to `/opt/ml/processing/train` and
  `/opt/ml/processing/test`. Consumed by Task 5's `ProcessingStep`.
- Produces: `config/tabular.yaml` with keys `task_type: tabular`, `label`, `eval_metric`,
  `presets` — consumed by Task 3's `train.py` and Task 4's `evaluate.py`.

Adapted directly from `1-tabular-classification/0-data-prep/preprocess.py` (proven, already in
this repo) — same UCI Adult Census dataset, same cleaning logic. Copied rather than imported
across experiment directories, matching this repo's existing pattern of each experiment being
self-contained.

- [ ] **Step 1: Write the failing test**

`5-sagemaker-project/seed-code/build/tests/test_preprocess.py`:
```python
import os
import subprocess
import sys

import pandas as pd


def test_preprocess_splits_train_test(tmp_path):
    input_dir = tmp_path / "input"
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    input_dir.mkdir()

    # Minimal Adult-Census-shaped fixture: 20 rows, balanced classes
    rows = []
    for i in range(20):
        label = " <=50K" if i % 2 == 0 else " >50K"
        rows.append(
            f"{25+i}, State-gov, 77000, Bachelors, 13, Never-married, "
            f"Adm-clerical, Not-in-family, White, Male, 0, 0, 40, United-States,{label}"
        )
    (input_dir / "sample.data").write_text("\n".join(rows) + "\n")

    script = os.path.join(os.path.dirname(__file__), "..", "pipelines", "automl", "preprocess.py")
    env = {
        **os.environ,
        "PROCESSING_INPUT_DIR": str(input_dir),
        "PROCESSING_TRAIN_DIR": str(train_dir),
        "PROCESSING_TEST_DIR": str(test_dir),
    }
    subprocess.run([sys.executable, script], check=True, env=env)

    train_df = pd.read_csv(train_dir / "train.csv")
    test_df = pd.read_csv(test_dir / "test.csv")
    assert len(train_df) + len(test_df) == 20
    assert "class" in train_df.columns
    assert set(train_df["class"].unique()) <= {"<=50K", ">50K"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd 5-sagemaker-project/seed-code/build
python3 -m pytest tests/test_preprocess.py -v
```

Expected: FAIL — `pipelines/automl/preprocess.py` does not exist yet.

- [ ] **Step 3: Write `preprocess.py`**

`5-sagemaker-project/seed-code/build/pipelines/automl/preprocess.py`:
```python
"""Tabular preprocessing for the AutoML build pipeline (UCI Adult Census default).

Reads raw Adult Census data, cleans it, splits into train/test. Runs inside a
ScriptProcessor container. Local paths are overridable via env vars so this
script can be exercised by unit tests without SageMaker Processing.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "class",
]

LABEL = "class"


def main() -> None:
    input_dir = os.environ.get("PROCESSING_INPUT_DIR", "/opt/ml/processing/input")
    train_output_dir = os.environ.get("PROCESSING_TRAIN_DIR", "/opt/ml/processing/train")
    test_output_dir = os.environ.get("PROCESSING_TEST_DIR", "/opt/ml/processing/test")

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    dfs = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".csv") or fname.endswith(".data"):
            fpath = os.path.join(input_dir, fname)
            try:
                df = pd.read_csv(fpath, header=None, names=COLUMNS, skipinitialspace=True)
                dfs.append(df)
            except (pd.errors.ParserError, ValueError):
                df = pd.read_csv(fpath, skipinitialspace=True)
                dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(data)} rows with columns: {data.columns.tolist()}")

    data[LABEL] = data[LABEL].astype(str).str.rstrip(".")
    data = data.replace("?", pd.NA).dropna()
    print(f"After cleaning: {len(data)} rows")

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data[LABEL])

    train_df.to_csv(os.path.join(train_output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(test_output_dir, "test.csv"), index=False)

    print(f"Train: {len(train_df)} rows")
    print(f"Test:  {len(test_df)} rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd 5-sagemaker-project/seed-code/build
python3 -m pytest tests/test_preprocess.py -v
```

Expected: `1 passed`. (If `sklearn`/`pandas` aren't in the local test env, `pip install pandas
scikit-learn` first — they're already pinned via AutoGluon's own dependency tree in the DLC, and
listed in `4-custom-image/docker/requirements.txt` style; add `pandas`/`scikit-learn` to
`extras_require["test"]` in Task 1's `setup.py` if missing.)

- [ ] **Step 5: Write the tabular config**

`5-sagemaker-project/seed-code/build/config/tabular.yaml`:
```yaml
task_type: tabular
label: class
eval_metric: roc_auc
presets: best_quality
```

- [ ] **Step 6: Commit**

```bash
git add 5-sagemaker-project/seed-code/build/pipelines/automl/preprocess.py \
        5-sagemaker-project/seed-code/build/config/tabular.yaml \
        5-sagemaker-project/seed-code/build/tests/test_preprocess.py
git commit -m "Add tabular preprocessing script and default config"
```

---

### Task 3: Pluggable `train.py` (tabular / timeseries / multimodal)

**Files:**
- Create: `5-sagemaker-project/seed-code/build/pipelines/automl/train.py`
- Create: `5-sagemaker-project/seed-code/build/config/timeseries.yaml`
- Create: `5-sagemaker-project/seed-code/build/config/multimodal.yaml`
- Test: `5-sagemaker-project/seed-code/build/tests/test_train.py`

**Interfaces:**
- Consumes: `config/*.yaml` (Task 2's `tabular.yaml` plus this task's `timeseries.yaml`/
  `multimodal.yaml`), each with a `task_type` key.
- Produces: `train.py`, invoked as `python train.py` inside the AutoGluon DLC with SageMaker's
  standard env vars (`SM_MODEL_DIR`, `SM_OUTPUT_DATA_DIR`, `SM_CHANNEL_TRAIN`, `SM_CHANNEL_TEST`,
  `SM_CHANNEL_CONFIG`) — writes the trained AutoGluon predictor to `SM_MODEL_DIR` and
  `evaluation.json` (`{"metrics": {<eval_metric>: <value>}}`) to `SM_OUTPUT_DATA_DIR`. Consumed
  by Task 5's `TrainingStep` and Task 4's `evaluate.py` (same output contract).

- [ ] **Step 1: Write the failing test (task_type dispatch, no AutoGluon fit — mocked)**

`5-sagemaker-project/seed-code/build/tests/test_train.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd 5-sagemaker-project/seed-code/build
python3 -m pytest tests/test_train.py -v
```

Expected: FAIL — `pipelines/automl/train.py` does not exist yet.

- [ ] **Step 3: Write `train.py`**

`5-sagemaker-project/seed-code/build/pipelines/automl/train.py`:
```python
"""Pluggable AutoGluon training script for SageMaker (tabular/timeseries/multimodal).

Runs inside the AutoGluon DLC container. Reads a YAML config from the `config`
channel that sets `task_type` (tabular|timeseries|multimodal), then dispatches
to the matching AutoGluon predictor. Writes evaluation.json in the shape
{"metrics": {<eval_metric>: <value>}} for the pipeline's ConditionStep.
"""
import argparse
import json
import os
from pprint import pprint

import yaml


def get_input_path(path: str) -> str:
    """Return the first non-hidden file found in the given directory."""
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    if not files:
        raise FileNotFoundError(f"No files found in {path}")
    if len(files) > 1:
        print(f"WARN: multiple files found in {path}, using first: {files[0]}")
    return os.path.join(path, files[0])


def get_env(name: str):
    return os.environ.get(name)


def build_predictor_args(config: dict, model_dir: str):
    """Return (predictor_class_name, predictor_kwargs) for the given config's task_type."""
    task_type = config.get("task_type")
    eval_metric = config.get("eval_metric", "auto")

    if task_type == "tabular":
        return "TabularPredictor", {
            "label": config["label"],
            "eval_metric": eval_metric,
            "path": model_dir,
        }
    if task_type == "timeseries":
        return "TimeSeriesPredictor", {
            "target": config.get("target", "target"),
            "prediction_length": config["prediction_length"],
            "eval_metric": eval_metric,
            "path": model_dir,
        }
    if task_type == "multimodal":
        return "MultiModalPredictor", {
            "label": config["label"],
            "eval_metric": eval_metric,
            "path": model_dir,
        }
    raise ValueError(f"Unknown task_type: {task_type}")


def load_config(config_dir: str) -> dict:
    config_file = get_input_path(config_dir)
    with open(config_file) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-dir", type=str, default=get_env("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=get_env("SM_MODEL_DIR"))
    parser.add_argument("--train-dir", type=str, default=get_env("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test-dir", type=str, required=False, default=get_env("SM_CHANNEL_TEST"))
    parser.add_argument("--config-dir", type=str, default=get_env("SM_CHANNEL_CONFIG"))
    args, _ = parser.parse_known_args()
    print(f"Args: {args}")

    os.makedirs(args.output_data_dir, exist_ok=True)

    config = load_config(args.config_dir)
    print("Training config:")
    pprint(config)

    save_path = os.path.normpath(args.model_dir)
    predictor_cls_name, predictor_kwargs = build_predictor_args(config, save_path)
    task_type = config["task_type"]

    if task_type == "tabular":
        from autogluon.tabular import TabularDataset, TabularPredictor

        train_data = TabularDataset(get_input_path(args.train_dir))
        predictor = TabularPredictor(**predictor_kwargs).fit(
            train_data, presets=config.get("presets", "medium_quality")
        )
        if args.test_dir:
            test_data = TabularDataset(get_input_path(args.test_dir))
            perf = predictor.evaluate(test_data)
        else:
            perf = None
        eval_metric_name = predictor.eval_metric.name if hasattr(predictor.eval_metric, "name") else str(predictor.eval_metric)

    elif task_type == "timeseries":
        import pandas as pd
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        def load_ts(path):
            df = pd.read_csv(get_input_path(path))
            df[config.get("timestamp_column", "timestamp")] = pd.to_datetime(
                df[config.get("timestamp_column", "timestamp")]
            )
            return TimeSeriesDataFrame.from_data_frame(
                df,
                id_column=config.get("id_column", "item_id"),
                timestamp_column=config.get("timestamp_column", "timestamp"),
            )

        train_data = load_ts(args.train_dir)
        predictor = TimeSeriesPredictor(**predictor_kwargs)
        predictor.fit(train_data=train_data, presets=config.get("presets", "medium_quality"))
        if args.test_dir:
            test_data = load_ts(args.test_dir)
            perf = predictor.evaluate(test_data)
        else:
            perf = None
        eval_metric_name = predictor.eval_metric

    elif task_type == "multimodal":
        import pandas as pd
        from autogluon.multimodal import MultiModalPredictor

        def load_mm(path):
            fpath = get_input_path(path)
            if fpath.endswith(".jsonl"):
                with open(fpath) as f:
                    return pd.DataFrame([json.loads(line) for line in f])
            return pd.read_csv(fpath)

        train_data = load_mm(args.train_dir)
        predictor = MultiModalPredictor(**predictor_kwargs)
        predictor.fit(train_data=train_data, presets=config.get("presets", "medium_quality"))
        if args.test_dir:
            test_data = load_mm(args.test_dir)
            perf = predictor.evaluate(test_data)
        else:
            perf = None
        eval_metric_name = predictor.eval_metric if hasattr(predictor, "eval_metric") else config.get("eval_metric", "roc_auc")

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    if perf is not None:
        metric_value = perf[eval_metric_name] if isinstance(perf, dict) else perf
        metrics = {"metrics": {eval_metric_name: abs(metric_value)}}
        with open(os.path.join(args.output_data_dir, "evaluation.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics))

    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd 5-sagemaker-project/seed-code/build
python3 -m pytest tests/test_train.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Write the reference configs**

`5-sagemaker-project/seed-code/build/config/timeseries.yaml`:
```yaml
# Reference config for timeseries AutoML. Requires a custom preprocess.py
# that emits long-format CSV with columns [item_id, timestamp, target] —
# see 2-timeseries-forecasting/0-data-prep/preprocess.py in this repo for
# an example of the wide-to-long conversion this task type needs.
task_type: timeseries
target: target
id_column: item_id
timestamp_column: timestamp
prediction_length: 84
eval_metric: MASE
presets: medium_quality
```

`5-sagemaker-project/seed-code/build/config/multimodal.yaml`:
```yaml
# Reference config for multimodal (text+tabular fusion) AutoML. Requires a
# custom preprocess.py that assembles JSONL/CSV rows with your text,
# numerical, and categorical feature columns plus a label column — see
# 3-multimodal/0-data-prep/preprocess.py in this repo for an example.
task_type: multimodal
label: y
eval_metric: roc_auc
presets: medium_quality
```

- [ ] **Step 6: Commit**

```bash
git add 5-sagemaker-project/seed-code/build/pipelines/automl/train.py \
        5-sagemaker-project/seed-code/build/config/timeseries.yaml \
        5-sagemaker-project/seed-code/build/config/multimodal.yaml \
        5-sagemaker-project/seed-code/build/tests/test_train.py
git commit -m "Add pluggable train.py (tabular/timeseries/multimodal) and reference configs"
```

---

### Task 4: Pluggable `evaluate.py`

**Files:**
- Create: `5-sagemaker-project/seed-code/build/pipelines/automl/evaluate.py`
- Test: `5-sagemaker-project/seed-code/build/tests/test_evaluate.py`

**Interfaces:**
- Consumes: `build_predictor_args` is NOT reused here (evaluate.py loads an already-trained
  predictor via `<PredictorClass>.load(...)`, it doesn't construct one) — but shares the same
  `task_type` dispatch convention as Task 3's `train.py`.
- Produces: a script run as `python evaluate.py` by a `ScriptProcessor` — reads
  `/opt/ml/processing/model/model.tar.gz` + `/opt/ml/processing/test/`, writes
  `/opt/ml/processing/evaluation/evaluation.json` in the same `{"metrics": {...}}` shape as
  Task 3. Consumed by Task 5's `ProcessingStep` + `PropertyFile` + `ConditionStep`.

- [ ] **Step 1: Write the failing test**

`5-sagemaker-project/seed-code/build/tests/test_evaluate.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd 5-sagemaker-project/seed-code/build
python3 -m pytest tests/test_evaluate.py -v
```

Expected: FAIL — `pipelines/automl/evaluate.py` does not exist yet.

- [ ] **Step 3: Write `evaluate.py`**

`5-sagemaker-project/seed-code/build/pipelines/automl/evaluate.py`:
```python
"""Pluggable evaluation script for the AutoML pipeline's ProcessingStep.

Loads a trained AutoGluon predictor (task_type read from the config channel)
and test data, computes metrics, and writes evaluation.json in the shape the
pipeline's ConditionStep expects: {"metrics": {<eval_metric>: <value>}}.
"""
import json
import os
import tarfile

import yaml


def predictor_class_for_task_type(task_type: str) -> str:
    mapping = {
        "tabular": "TabularPredictor",
        "timeseries": "TimeSeriesPredictor",
        "multimodal": "MultiModalPredictor",
    }
    if task_type not in mapping:
        raise ValueError(f"Unknown task_type: {task_type}")
    return mapping[task_type]


def get_input_path(path: str) -> str:
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    if not files:
        raise FileNotFoundError(f"No files found in {path}")
    return os.path.join(path, files[0])


def main() -> None:
    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    config_dir = "/opt/ml/processing/config"
    output_dir = "/opt/ml/processing/evaluation"

    os.makedirs(output_dir, exist_ok=True)

    with open(get_input_path(config_dir)) as f:
        config = yaml.safe_load(f)
    task_type = config["task_type"]

    model_tar = os.path.join(model_dir, "model.tar.gz")
    extract_dir = "/opt/ml/processing/model_extracted"
    with tarfile.open(model_tar) as tar:
        tar.extractall(path=extract_dir, filter="data")

    if task_type == "tabular":
        from autogluon.tabular import TabularDataset, TabularPredictor

        predictor = TabularPredictor.load(extract_dir)
        test_data = TabularDataset(get_input_path(test_dir))
        perf = predictor.evaluate(test_data)
        eval_metric = predictor.eval_metric.name if hasattr(predictor.eval_metric, "name") else str(predictor.eval_metric)

    elif task_type == "timeseries":
        import pandas as pd
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        predictor = TimeSeriesPredictor.load(extract_dir)
        test_df = pd.read_csv(get_input_path(test_dir))
        test_df[config.get("timestamp_column", "timestamp")] = pd.to_datetime(
            test_df[config.get("timestamp_column", "timestamp")]
        )
        test_data = TimeSeriesDataFrame.from_data_frame(
            test_df,
            id_column=config.get("id_column", "item_id"),
            timestamp_column=config.get("timestamp_column", "timestamp"),
        )
        perf = predictor.evaluate(test_data)
        eval_metric = predictor.eval_metric.name if hasattr(predictor.eval_metric, "name") else str(predictor.eval_metric)

    elif task_type == "multimodal":
        import pandas as pd
        from autogluon.multimodal import MultiModalPredictor

        predictor = MultiModalPredictor.load(extract_dir)
        test_file = get_input_path(test_dir)
        if test_file.endswith(".jsonl"):
            with open(test_file) as f:
                test_data = pd.DataFrame([json.loads(line) for line in f])
        else:
            test_data = pd.read_csv(test_file)
        perf = predictor.evaluate(test_data)
        eval_metric = predictor.eval_metric if hasattr(predictor, "eval_metric") else config.get("eval_metric", "roc_auc")

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    metric_value = perf[eval_metric] if isinstance(perf, dict) else perf
    metrics = {"metrics": {eval_metric: abs(metric_value)}}

    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation written to {eval_path}: {metrics}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd 5-sagemaker-project/seed-code/build
python3 -m pytest tests/test_evaluate.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add 5-sagemaker-project/seed-code/build/pipelines/automl/evaluate.py \
        5-sagemaker-project/seed-code/build/tests/test_evaluate.py
git commit -m "Add pluggable evaluate.py for the AutoML pipeline's ProcessingStep"
```

---

### Task 5: `pipeline.py` (Preprocess -> Train -> Evaluate -> Condition -> Register) + buildspec

**Files:**
- Create: `5-sagemaker-project/seed-code/build/pipelines/automl/pipeline.py`
- Create: `5-sagemaker-project/seed-code/build/codebuild-buildspec.yml`
- Test: `5-sagemaker-project/seed-code/build/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 2's `preprocess.py`, Task 3's `train.py`, Task 4's `evaluate.py`, Task 1's
  `pipelines._utils.get_pipeline_driver`/`run_pipeline.py`/`get_pipeline_definition.py`.
- Produces: `get_pipeline(region, role=None, default_bucket=None, model_package_group_name=...,
  pipeline_name=..., sagemaker_project_name=None, config_file="tabular.yaml", **kwargs) ->
  sagemaker.mlops.workflow.pipeline.Pipeline` — the entrypoint Task 1's `run_pipeline.py` calls
  via `--module-name pipelines.automl.pipeline`. Consumed by Task 7's setup notebook (indirectly,
  via CodeBuild running `run-pipeline`).

Model registration uses the verified pattern (`ModelBuilder(...).register(...)` → `ModelStep`),
confirmed working end-to-end against this account's real IAM role in this session (produces a
valid `pipeline.definition()` with `TrainAutoGluon` and `RegisterModel` steps).

- [ ] **Step 1: Write the failing test**

`5-sagemaker-project/seed-code/build/tests/test_pipeline.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd 5-sagemaker-project/seed-code/build
python3 -m pytest tests/test_pipeline.py -v
```

Expected: FAIL — `pipelines/automl/pipeline.py` does not exist yet.

- [ ] **Step 3: Write `pipeline.py`**

`5-sagemaker-project/seed-code/build/pipelines/automl/pipeline.py`:
```python
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


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="AutoMLModels",
    pipeline_name="AutoMLPipeline",
    base_job_prefix="AutoML",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.2xlarge",
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
    # to read out of evaluation.json. BASE_DIR is .../pipelines/automl; config/ is
    # two levels up, at the package root (.../build/config/).
    build_root = os.path.dirname(os.path.dirname(BASE_DIR))
    config_path = os.path.join(build_root, "config", config_file)
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

    ag_training_image = image_uris.retrieve(
        "autogluon", region=region, version=ag_version, py_version=py_version,
        image_scope="training", instance_type=training_instance_type,
    )
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
    model_builder = ModelBuilder(
        image_uri=ag_training_image,
        s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role_arn=role,
        sagemaker_session=pipeline_session,
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd 5-sagemaker-project/seed-code/build
python3 -m pytest tests/test_pipeline.py -v
```

Expected: `1 passed`. The `JsonGet(step_name=..., property_file=..., json_path=f"metrics.{eval_metric}")`
+ `ConditionGreaterThanOrEqualTo` construction above was verified directly against
`sagemaker==3.16.0` in a throwaway script during planning — no fallback needed.

- [ ] **Step 5: Write `codebuild-buildspec.yml`**

`5-sagemaker-project/seed-code/build/codebuild-buildspec.yml`:
```yaml
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.12
    commands:
      - pip install --upgrade --force-reinstall . "awscli>1.20.30"

  build:
    commands:
      - export PYTHONUNBUFFERED=TRUE
      - export SAGEMAKER_PROJECT_NAME_ID="${SAGEMAKER_PROJECT_NAME}-${SAGEMAKER_PROJECT_ID}"
      - aws s3 cp config/tabular.yaml "s3://${ARTIFACT_BUCKET}/AutoML/pipeline/config/tabular.yaml"
      - |
        run-pipeline --module-name pipelines.automl.pipeline \
          --role-arn $SAGEMAKER_PIPELINE_ROLE_ARN \
          --tags "[{\"Key\":\"sagemaker:project-name\", \"Value\":\"${SAGEMAKER_PROJECT_NAME}\"}, {\"Key\":\"sagemaker:project-id\", \"Value\":\"${SAGEMAKER_PROJECT_ID}\"}]" \
          --kwargs "{\"region\":\"${AWS_REGION}\",\"role\":\"${SAGEMAKER_PIPELINE_ROLE_ARN}\",\"default_bucket\":\"${ARTIFACT_BUCKET}\",\"pipeline_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"model_package_group_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"base_job_prefix\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"sagemaker_project_name\":\"${SAGEMAKER_PROJECT_NAME}\"}"
      - echo "Create/Update of the SageMaker Pipeline and execution completed."
```

Note the added `aws s3 cp config/tabular.yaml ...` line (not present in `petro-build`'s
buildspec) — this pipeline's `config` channel needs the YAML file in S3 before training starts,
unlike `petro-build`'s XGBoost pipeline which took hyperparameters directly.

- [ ] **Step 6: Commit**

```bash
git add 5-sagemaker-project/seed-code/build/pipelines/automl/pipeline.py \
        5-sagemaker-project/seed-code/build/codebuild-buildspec.yml \
        5-sagemaker-project/seed-code/build/tests/test_pipeline.py
git commit -m "Add AutoML pipeline.py (Preprocess->Train->Evaluate->Condition->Register) and buildspec"
```

---

### Task 6: Deploy repo — real-time endpoint pipeline seed code

**Files:**
- Create: `5-sagemaker-project/seed-code/deploy/realtime/build.py`
- Create: `5-sagemaker-project/seed-code/deploy/realtime/buildspec.yml`
- Create: `5-sagemaker-project/seed-code/deploy/realtime/endpoint-config-template.yml`
- Create: `5-sagemaker-project/seed-code/deploy/realtime/staging-config.json`
- Create: `5-sagemaker-project/seed-code/deploy/realtime/prod-config.json`
- Create: `5-sagemaker-project/seed-code/deploy/realtime/serve.py`
- Create: `5-sagemaker-project/seed-code/deploy/realtime/test/test.py`
- Create: `5-sagemaker-project/seed-code/deploy/realtime/test/buildspec.yml`
- Test: `5-sagemaker-project/seed-code/deploy/realtime/tests/test_build.py`

**Interfaces:**
- Consumes: `ModelPackageArn` from the `ModelPackageGroupName` Task 5's `pipeline.py` registers
  to (matched at deploy time via `list_model_packages(ModelApprovalStatus="Approved")`).
- Produces: `build.py`'s `extend_config`/`get_approved_package`/`create_cfn_params_tags_file`
  functions (adapted verbatim from the proven `petro-deploy/build.py`), emitting
  `staging-config-export.json`/`prod-config-export.json` consumed by the CodePipeline's
  CloudFormation deploy action (Task 7's `project-template.yaml`).
- Produces: `serve.py` — a pluggable real-time inference handler (`model_fn`/`transform_fn`)
  dispatching on the predictor class found in the loaded model directory, reused from
  `1-tabular-classification/2-inference/serve.py`'s proven CSV/JSON/JSONL/Parquet handling for
  the tabular case, extended with timeseries/multimodal branches.

This is adapted from the account's real, working `petro-deploy` repo (verified via `gh api` in
this session) — not written from scratch.

- [ ] **Step 1: Write the failing test**

`5-sagemaker-project/seed-code/deploy/realtime/tests/test_build.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd 5-sagemaker-project/seed-code/deploy/realtime
python3 -m pytest tests/test_build.py -v
```

Expected: FAIL — `build.py` does not exist yet.

- [ ] **Step 3: Write `build.py`** (adapted from the verified working `petro-deploy/build.py`)

`5-sagemaker-project/seed-code/deploy/realtime/build.py`:
```python
"""Reads the latest approved model package and renders staging/prod CFN configs
for the real-time endpoint deploy pipeline. Adapted from a proven pattern
already running in this AWS account (petro-deploy)."""
import argparse
import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


def get_approved_package(model_package_group_name):
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]
        while len(approved_packages) == 0 and "NextToken" in response:
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])
        if len(approved_packages) == 0:
            raise Exception(f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}")
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return model_package_arn
    except ClientError as e:
        raise Exception(e.response["Error"]["Message"])


def get_pipeline_custom_tags(args, new_tags):
    try:
        project_arn = sm_client.describe_project(ProjectName=args.sagemaker_project_name)["ProjectArn"]
        for tag in sm_client.list_tags(ResourceArn=project_arn)["Tags"]:
            new_tags[tag["Key"]] = tag["Value"]
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error getting project tags: {e}")
    return new_tags


def extend_config(args, model_package_arn, stage_config):
    if "Parameters" not in stage_config or "StageName" not in stage_config["Parameters"]:
        raise Exception("Configuration file must include StageName parameter")
    if "Tags" not in stage_config:
        stage_config["Tags"] = {}
    new_params = {
        "SageMakerProjectName": args.sagemaker_project_name,
        "ModelPackageName": model_package_arn,
        "ModelExecutionRoleArn": args.model_execution_role,
        "DataCaptureUploadPath": "s3://" + args.s3_bucket + "/datacapture-" + stage_config["Parameters"]["StageName"],
    }
    new_tags = {
        "sagemaker:deployment-stage": stage_config["Parameters"]["StageName"],
        "sagemaker:project-id": args.sagemaker_project_id,
        "sagemaker:project-name": args.sagemaker_project_name,
    }
    new_tags = get_pipeline_custom_tags(args, new_tags)
    return {
        "Parameters": {**stage_config["Parameters"], **new_params},
        "Tags": {**stage_config.get("Tags", {}), **new_tags},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--model-execution-role", type=str, required=True)
    parser.add_argument("--model-package-group-name", type=str, required=True)
    parser.add_argument("--sagemaker-project-id", type=str, required=True)
    parser.add_argument("--sagemaker-project-name", type=str, required=True)
    parser.add_argument("--s3-bucket", type=str, required=True)
    parser.add_argument("--import-staging-config", type=str, default="staging-config.json")
    parser.add_argument("--import-prod-config", type=str, default="prod-config.json")
    parser.add_argument("--export-staging-config", type=str, default="staging-config-export.json")
    parser.add_argument("--export-prod-config", type=str, default="prod-config-export.json")
    args, _ = parser.parse_known_args()

    logging.basicConfig(format="%(levelname)s: [%(filename)s:%(lineno)s] %(message)s", level=args.log_level)

    model_package_arn = get_approved_package(args.model_package_group_name)

    with open(args.import_staging_config) as f:
        staging_config = extend_config(args, model_package_arn, json.load(f))
    with open(args.export_staging_config, "w") as f:
        json.dump(staging_config, f, indent=4)

    with open(args.import_prod_config) as f:
        prod_config = extend_config(args, model_package_arn, json.load(f))
    with open(args.export_prod_config, "w") as f:
        json.dump(prod_config, f, indent=4)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd 5-sagemaker-project/seed-code/deploy/realtime
python3 -m pytest tests/test_build.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Write `endpoint-config-template.yml`, config JSONs, `buildspec.yml`**

`5-sagemaker-project/seed-code/deploy/realtime/endpoint-config-template.yml`:
```yaml
Description: Real-time endpoint for the AutoML model package, deployed per-stage.
Parameters:
  SageMakerProjectName:
    Type: String
    MinLength: 1
    MaxLength: 32
    AllowedPattern: ^[a-zA-Z](-*[a-zA-Z0-9])*
  ModelExecutionRoleArn:
    Type: String
  ModelPackageName:
    Type: String
  StageName:
    Type: String
  EndpointInstanceCount:
    Type: Number
    MinValue: 1
  EndpointInstanceType:
    Type: String
  DataCaptureUploadPath:
    Type: String
  SamplingPercentage:
    Type: Number
    MinValue: 0
    MaxValue: 100
  EnableDataCapture:
    Default: true
    Type: String
    AllowedValues: [true, false]

Resources:
  Model:
    Type: AWS::SageMaker::Model
    Properties:
      Containers:
        - ModelPackageName: !Ref ModelPackageName
      ExecutionRoleArn: !Ref ModelExecutionRoleArn

  EndpointConfig:
    Type: AWS::SageMaker::EndpointConfig
    Properties:
      ProductionVariants:
        - InitialInstanceCount: !Ref EndpointInstanceCount
          InitialVariantWeight: 1.0
          InstanceType: !Ref EndpointInstanceType
          ModelName: !GetAtt Model.ModelName
          VariantName: AllTraffic
      DataCaptureConfig:
        EnableCapture: !Ref EnableDataCapture
        InitialSamplingPercentage: !Ref SamplingPercentage
        DestinationS3Uri: !Ref DataCaptureUploadPath
        CaptureOptions:
          - CaptureMode: Input
          - CaptureMode: Output
        CaptureContentTypeHeader:
          CsvContentTypes: ["text/csv"]

  Endpoint:
    Type: AWS::SageMaker::Endpoint
    Properties:
      EndpointName: !Sub ${SageMakerProjectName}-${StageName}
      EndpointConfigName: !GetAtt EndpointConfig.EndpointConfigName
```

`5-sagemaker-project/seed-code/deploy/realtime/staging-config.json`:
```json
{
  "Parameters": {
    "StageName": "staging",
    "EndpointInstanceCount": "1",
    "EndpointInstanceType": "ml.m5.large",
    "SamplingPercentage": "100",
    "EnableDataCapture": "true"
  }
}
```

`5-sagemaker-project/seed-code/deploy/realtime/prod-config.json`:
```json
{
  "Parameters": {
    "StageName": "prod",
    "EndpointInstanceCount": "1",
    "EndpointInstanceType": "ml.m5.large",
    "SamplingPercentage": "80",
    "EnableDataCapture": "true"
  }
}
```

`5-sagemaker-project/seed-code/deploy/realtime/buildspec.yml`:
```yaml
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.12
    commands:
      - pip install --upgrade --force-reinstall "botocore>1.21.30" "boto3>1.18.30" "awscli>1.20.30"

  build:
    commands:
      - python build.py --model-execution-role "$MODEL_EXECUTION_ROLE_ARN" --model-package-group-name "$SOURCE_MODEL_PACKAGE_GROUP_NAME" --sagemaker-project-id "$SAGEMAKER_PROJECT_ID" --sagemaker-project-name "$SAGEMAKER_PROJECT_NAME" --s3-bucket "$ARTIFACT_BUCKET" --export-staging-config $EXPORT_TEMPLATE_STAGING_CONFIG --export-prod-config $EXPORT_TEMPLATE_PROD_CONFIG
      - aws cloudformation package --template endpoint-config-template.yml --s3-bucket $ARTIFACT_BUCKET --output-template $EXPORT_TEMPLATE_NAME
      - cat $EXPORT_TEMPLATE_STAGING_CONFIG
      - cat $EXPORT_TEMPLATE_PROD_CONFIG

artifacts:
  files:
    - $EXPORT_TEMPLATE_NAME
    - $EXPORT_TEMPLATE_STAGING_CONFIG
    - $EXPORT_TEMPLATE_PROD_CONFIG
```

- [ ] **Step 6: Write `serve.py`** (adapted from `1-tabular-classification/2-inference/serve.py`,
  extended for the three task types)

`5-sagemaker-project/seed-code/deploy/realtime/serve.py`:
```python
"""Pluggable AutoGluon real-time inference handler.

Detects the predictor type from the loaded model directory (AutoGluon writes
a `learner.pkl`/predictor-class marker readable via `TabularPredictor.load`
et al. — each predictor class only loads its own artifact shape) and
dispatches accordingly. Tabular path is adapted verbatim from
1-tabular-classification/2-inference/serve.py.
"""
from io import BytesIO, StringIO

import pandas as pd


def model_fn(model_dir):
    """Try each AutoGluon predictor class in turn; the one that loads wins."""
    from autogluon.tabular import TabularPredictor

    try:
        model = TabularPredictor.load(model_dir)
        globals()["task_type"] = "tabular"
        globals()["column_names"] = model.feature_metadata_in.get_features()
        return model
    except Exception:
        pass

    from autogluon.timeseries import TimeSeriesPredictor

    try:
        model = TimeSeriesPredictor.load(model_dir)
        globals()["task_type"] = "timeseries"
        return model
    except Exception:
        pass

    from autogluon.multimodal import MultiModalPredictor

    model = MultiModalPredictor.load(model_dir)
    globals()["task_type"] = "multimodal"
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    from autogluon.core.constants import REGRESSION
    from autogluon.core.utils import get_pred_from_proba_df

    request_body_str = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body

    if input_content_type == "application/x-parquet":
        data = pd.read_parquet(BytesIO(request_body if isinstance(request_body, bytes) else request_body.encode()))
    elif input_content_type == "text/csv":
        data = pd.read_csv(StringIO(request_body_str))
    elif input_content_type == "application/json":
        data = pd.read_json(StringIO(request_body_str))
    elif input_content_type == "application/jsonl":
        data = pd.read_json(StringIO(request_body_str), orient="records", lines=True)
    else:
        raise ValueError(f"{input_content_type} input content type not supported.")

    task_type = globals().get("task_type", "tabular")

    if task_type == "tabular":
        if model.problem_type != REGRESSION:
            pred_proba = model.predict_proba(data, as_pandas=True)
            pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
            pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
            pred.name = str(pred.name) + "_pred" if pred.name else "pred"
            prediction = pd.concat([pred, pred_proba], axis=1)
        else:
            prediction = model.predict(data, as_pandas=True)
    else:
        prediction = model.predict(data)

    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()

    if "application/x-parquet" in output_content_type:
        prediction.columns = prediction.columns.astype(str)
        return prediction.to_parquet(), "application/x-parquet"
    elif "application/json" in output_content_type:
        return prediction.to_json(), "application/json"
    elif "text/csv" in output_content_type:
        return prediction.to_csv(index=None), "text/csv"
    raise ValueError(f"{output_content_type} content type not supported")
```

- [ ] **Step 7: Write `test/test.py` and `test/buildspec.yml`** (adapted verbatim from
  `petro-deploy/test/`)

`5-sagemaker-project/seed-code/deploy/realtime/test/test.py`:
```python
"""Smoke test for the staging endpoint: ensure InService, then invoke."""
import argparse
import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")
runtime_client = boto3.client("sagemaker-runtime")


def invoke_endpoint(endpoint_name):
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Accept="application/json",
        Body="age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country\n39,State-gov,77516,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States\n",
    )
    return {"endpoint_name": endpoint_name, "success": True, "body": response["Body"].read().decode("utf-8")}


def test_endpoint(endpoint_name):
    try:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        if status != "InService":
            raise Exception(f"SageMaker endpoint: {endpoint_name} status: {status} not InService")
        return invoke_endpoint(endpoint_name)
    except ClientError as e:
        raise Exception(e.response["Error"]["Message"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--import-build-config", type=str, required=True)
    parser.add_argument("--export-test-results", type=str, required=True)
    args, _ = parser.parse_known_args()

    logging.basicConfig(format="%(levelname)s: [%(filename)s:%(lineno)s] %(message)s", level=args.log_level)

    with open(args.import_build_config) as f:
        config = json.load(f)

    endpoint_name = "{}-{}".format(config["Parameters"]["SageMakerProjectName"], config["Parameters"]["StageName"])
    results = test_endpoint(endpoint_name)

    with open(args.export_test_results, "w") as f:
        json.dump(results, f, indent=4)
```

`5-sagemaker-project/seed-code/deploy/realtime/test/buildspec.yml`:
```yaml
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.12
  build:
    commands:
      - python test/test.py --import-build-config $CODEBUILD_SRC_DIR_BuildArtifact/staging-config-export.json --export-test-results $EXPORT_TEST_RESULTS
      - cat $EXPORT_TEST_RESULTS

artifacts:
  files:
    - $EXPORT_TEST_RESULTS
```

- [ ] **Step 8: Commit**

```bash
git add 5-sagemaker-project/seed-code/deploy/realtime/
git commit -m "Add real-time deploy pipeline seed code (build.py, CFN template, serve.py, test)"
```

---

### Task 7: Deploy repo — batch transform pipeline seed code

**Files:**
- Create: `5-sagemaker-project/seed-code/deploy/batch/build.py`
- Create: `5-sagemaker-project/seed-code/deploy/batch/buildspec.yml`
- Create: `5-sagemaker-project/seed-code/deploy/batch/batch-transform-template.yml`
- Create: `5-sagemaker-project/seed-code/deploy/batch/staging-config.json`
- Create: `5-sagemaker-project/seed-code/deploy/batch/prod-config.json`
- Create: `5-sagemaker-project/seed-code/deploy/batch/lambda/run_transform.py`
- Create: `5-sagemaker-project/seed-code/deploy/batch/serve_batch.py`
- Create: `5-sagemaker-project/seed-code/deploy/batch/test/test.py`
- Create: `5-sagemaker-project/seed-code/deploy/batch/test/buildspec.yml`
- Test: `5-sagemaker-project/seed-code/deploy/batch/tests/test_run_transform.py`

**Interfaces:**
- Consumes: same `get_approved_package`/`extend_config` pattern as Task 6's `build.py` (same
  `ModelPackageGroupName`).
- Produces: `lambda/run_transform.py`'s handler `def handler(event, context) -> dict`, invoked
  synchronously by staging's Test stage and on a schedule by prod's EventBridge rule (created by
  `batch-transform-template.yml`, which requires Task 0's widened IAM policy to deploy).
- Produces: `serve_batch.py` — pluggable batch inference handler, adapted from
  `1-tabular-classification/2-inference/serve_batch.py`.

- [ ] **Step 1: Write the failing test for the Lambda handler**

`5-sagemaker-project/seed-code/deploy/batch/tests/test_run_transform.py`:
```python
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../lambda")


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd 5-sagemaker-project/seed-code/deploy/batch
python3 -m pytest tests/test_run_transform.py -v
```

Expected: FAIL — `lambda/run_transform.py` does not exist yet.

- [ ] **Step 3: Write `lambda/run_transform.py`**

`5-sagemaker-project/seed-code/deploy/batch/lambda/run_transform.py`:
```python
"""Lambda handler: launch a SageMaker batch Transform job and wait for it.

Invoked synchronously by the deploy pipeline's staging Test stage, and on a
schedule (EventBridge rule) in prod. model_name falls back to the MODEL_NAME
env var (set by batch-transform-template.yml) when the caller's event
payload omits it — the staging Test stage does this deliberately, since it
doesn't know the CFN-generated SageMaker Model name ahead of time.
"""
import os
import time

import boto3

sm_client = boto3.client("sagemaker")

POLL_INTERVAL_SECONDS = 30
MAX_POLL_ATTEMPTS = 120  # 60 minutes


def handler(event, context):
    transform_job_name = event["transform_job_name"]
    model_name = event.get("model_name") or os.environ["MODEL_NAME"]

    sm_client.create_transform_job(
        TransformJobName=transform_job_name,
        ModelName=model_name,
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": event["input_s3_uri"]}},
            "ContentType": event["content_type"],
        },
        TransformOutput={"S3OutputPath": event["output_s3_uri"]},
        TransformResources={
            "InstanceType": event["instance_type"],
            "InstanceCount": event["instance_count"],
        },
    )

    for _ in range(MAX_POLL_ATTEMPTS):
        response = sm_client.describe_transform_job(TransformJobName=transform_job_name)
        status = response["TransformJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    else:
        raise Exception(f"Transform job {transform_job_name} did not finish within the poll window")

    if status != "Completed":
        raise Exception(f"Transform job {transform_job_name} ended with status {status}: {response.get('FailureReason')}")

    return {"status": status, "transform_job_name": transform_job_name}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd 5-sagemaker-project/seed-code/deploy/batch
python3 -m pytest tests/test_run_transform.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Write `build.py`, `buildspec.yml`, configs** (same pattern as Task 6, adjusted
  for the batch resources)

`5-sagemaker-project/seed-code/deploy/batch/build.py`: same `get_approved_package`/
`get_pipeline_custom_tags` functions as Task 6's `build.py`, with `extend_config` adding two
extra parameters (`LambdaCodeS3Bucket`/`LambdaCodeS3Key`) that `batch-transform-template.yml`
needs to locate the Lambda deployment zip uploaded by this directory's `buildspec.yml`:

```python
"""Reads the latest approved model package and renders staging/prod CFN configs
for the batch-transform deploy pipeline. Adapted from build.py in
5-sagemaker-project/seed-code/deploy/realtime/, extended with the
LambdaCodeS3Bucket/LambdaCodeS3Key parameters batch-transform-template.yml needs."""
import argparse
import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


def get_approved_package(model_package_group_name):
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]
        while len(approved_packages) == 0 and "NextToken" in response:
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])
        if len(approved_packages) == 0:
            raise Exception(f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}")
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return model_package_arn
    except ClientError as e:
        raise Exception(e.response["Error"]["Message"])


def get_pipeline_custom_tags(args, new_tags):
    try:
        project_arn = sm_client.describe_project(ProjectName=args.sagemaker_project_name)["ProjectArn"]
        for tag in sm_client.list_tags(ResourceArn=project_arn)["Tags"]:
            new_tags[tag["Key"]] = tag["Value"]
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error getting project tags: {e}")
    return new_tags


def extend_config(args, model_package_arn, stage_config):
    if "Parameters" not in stage_config or "StageName" not in stage_config["Parameters"]:
        raise Exception("Configuration file must include StageName parameter")
    if "Tags" not in stage_config:
        stage_config["Tags"] = {}
    stage_name = stage_config["Parameters"]["StageName"]
    new_params = {
        "SageMakerProjectName": args.sagemaker_project_name,
        "ModelPackageName": model_package_arn,
        "ModelExecutionRoleArn": args.model_execution_role,
        "DataCaptureUploadPath": "s3://" + args.s3_bucket + "/datacapture-" + stage_name,
        "LambdaCodeS3Bucket": args.s3_bucket,
        "LambdaCodeS3Key": f"AutoML/lambda/{stage_name}/run_transform.zip",
    }
    new_tags = {
        "sagemaker:deployment-stage": stage_name,
        "sagemaker:project-id": args.sagemaker_project_id,
        "sagemaker:project-name": args.sagemaker_project_name,
    }
    new_tags = get_pipeline_custom_tags(args, new_tags)
    return {
        "Parameters": {**stage_config["Parameters"], **new_params},
        "Tags": {**stage_config.get("Tags", {}), **new_tags},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--model-execution-role", type=str, required=True)
    parser.add_argument("--model-package-group-name", type=str, required=True)
    parser.add_argument("--sagemaker-project-id", type=str, required=True)
    parser.add_argument("--sagemaker-project-name", type=str, required=True)
    parser.add_argument("--s3-bucket", type=str, required=True)
    parser.add_argument("--import-staging-config", type=str, default="staging-config.json")
    parser.add_argument("--import-prod-config", type=str, default="prod-config.json")
    parser.add_argument("--export-staging-config", type=str, default="staging-config-export.json")
    parser.add_argument("--export-prod-config", type=str, default="prod-config-export.json")
    args, _ = parser.parse_known_args()

    logging.basicConfig(format="%(levelname)s: [%(filename)s:%(lineno)s] %(message)s", level=args.log_level)

    model_package_arn = get_approved_package(args.model_package_group_name)

    with open(args.import_staging_config) as f:
        staging_config = extend_config(args, model_package_arn, json.load(f))
    with open(args.export_staging_config, "w") as f:
        json.dump(staging_config, f, indent=4)

    with open(args.import_prod_config) as f:
        prod_config = extend_config(args, model_package_arn, json.load(f))
    with open(args.export_prod_config, "w") as f:
        json.dump(prod_config, f, indent=4)
```

Add a test verifying the new parameters, `5-sagemaker-project/seed-code/deploy/batch/tests/test_build.py`:

```python
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, "..")


def test_extend_config_adds_lambda_code_location():
    import build

    args = SimpleNamespace(
        sagemaker_project_name="test-proj",
        sagemaker_project_id="p-abc123",
        model_execution_role="arn:aws:iam::123456789012:role/exec-role",
        s3_bucket="test-bucket",
    )
    stage_config = {"Parameters": {"StageName": "staging"}}

    with patch("build.get_pipeline_custom_tags", return_value={}):
        result = build.extend_config(args, "arn:aws:sagemaker:::model-package/foo/1", stage_config)

    assert result["Parameters"]["LambdaCodeS3Bucket"] == "test-bucket"
    assert result["Parameters"]["LambdaCodeS3Key"] == "AutoML/lambda/staging/run_transform.zip"
```

```bash
cd 5-sagemaker-project/seed-code/deploy/batch
python3 -m pytest tests/test_build.py -v
```

Expected: `1 passed`.

`5-sagemaker-project/seed-code/deploy/batch/staging-config.json`:
```json
{
  "Parameters": {
    "StageName": "staging",
    "TransformInstanceType": "ml.m5.xlarge",
    "TransformInstanceCount": "1",
    "ScheduleExpression": "rate(1 day)",
    "ScheduleEnabled": "false"
  }
}
```

`5-sagemaker-project/seed-code/deploy/batch/prod-config.json`:
```json
{
  "Parameters": {
    "StageName": "prod",
    "TransformInstanceType": "ml.m5.xlarge",
    "TransformInstanceCount": "1",
    "ScheduleExpression": "rate(1 day)",
    "ScheduleEnabled": "true"
  }
}
```

`5-sagemaker-project/seed-code/deploy/batch/batch-transform-template.yml`:
```yaml
Description: >
  Batch transform for the AutoML model package: a SageMaker Model, a Lambda
  that runs+waits-on a TransformJob, and an EventBridge schedule (disabled in
  staging — invoked directly by the Test stage instead; enabled in prod).
Parameters:
  SageMakerProjectName:
    Type: String
    MinLength: 1
    MaxLength: 32
    AllowedPattern: ^[a-zA-Z](-*[a-zA-Z0-9])*
  ModelExecutionRoleArn:
    Type: String
  ModelPackageName:
    Type: String
  StageName:
    Type: String
  TransformInstanceType:
    Type: String
  TransformInstanceCount:
    Type: Number
    MinValue: 1
  ScheduleExpression:
    Type: String
  ScheduleEnabled:
    Type: String
    AllowedValues: ["true", "false"]
  DataCaptureUploadPath:
    Type: String
    Description: Used as the base for batch input/output S3 prefixes.
  LambdaCodeS3Bucket:
    Type: String
    Description: S3 bucket holding the run_transform.py Lambda deployment zip.
  LambdaCodeS3Key:
    Type: String
    Description: S3 key of the run_transform.py Lambda deployment zip.

Conditions:
  IsScheduleEnabled: !Equals [!Ref ScheduleEnabled, "true"]

Resources:
  Model:
    Type: AWS::SageMaker::Model
    Properties:
      Containers:
        - ModelPackageName: !Ref ModelPackageName
      ExecutionRoleArn: !Ref ModelExecutionRoleArn

  RunTransformLambdaRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub sagemaker-${SageMakerProjectName}-${StageName}-batch-transform-lambda
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal: {Service: lambda.amazonaws.com}
            Action: sts:AssumeRole
      Policies:
        - PolicyName: RunTransformPolicy
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action: [sagemaker:CreateTransformJob, sagemaker:DescribeTransformJob]
                Resource: "*"
              - Effect: Allow
                Action: [logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents]
                Resource: "*"

  RunTransformLambda:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Sub sagemaker-${SageMakerProjectName}-${StageName}-run-transform
      Runtime: python3.12
      Handler: run_transform.handler
      Role: !GetAtt RunTransformLambdaRole.Arn
      Timeout: 900
      Code:
        S3Bucket: !Ref LambdaCodeS3Bucket
        S3Key: !Ref LambdaCodeS3Key
      Environment:
        Variables:
          MODEL_NAME: !GetAtt Model.ModelName
          INPUT_S3_URI: !Sub "${DataCaptureUploadPath}/batch-input/"
          OUTPUT_S3_URI: !Sub "${DataCaptureUploadPath}/batch-output/"
          TRANSFORM_INSTANCE_TYPE: !Ref TransformInstanceType
          TRANSFORM_INSTANCE_COUNT: !Ref TransformInstanceCount

  ScheduleRule:
    Type: AWS::Events::Rule
    Properties:
      Name: !Sub sagemaker-${SageMakerProjectName}-${StageName}-batch-schedule
      ScheduleExpression: !Ref ScheduleExpression
      State: !If [IsScheduleEnabled, ENABLED, DISABLED]
      Targets:
        - Arn: !GetAtt RunTransformLambda.Arn
          Id: !Sub "${SageMakerProjectName}-${StageName}-target"
          Input: !Sub |
            {
              "model_name": "${Model.ModelName}",
              "transform_job_name": "${SageMakerProjectName}-${StageName}-scheduled",
              "input_s3_uri": "${DataCaptureUploadPath}/batch-input/",
              "output_s3_uri": "${DataCaptureUploadPath}/batch-output/",
              "instance_type": "${TransformInstanceType}",
              "instance_count": ${TransformInstanceCount},
              "content_type": "text/csv"
            }

  ScheduleInvokeLambdaPermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref RunTransformLambda
      Action: lambda:InvokeFunction
      Principal: events.amazonaws.com
      SourceArn: !GetAtt ScheduleRule.Arn
```

`Code.S3Bucket`/`S3Key` reference a Lambda deployment zip that `buildspec.yml` uploads before
`aws cloudformation package` runs (see Step 5's `buildspec.yml` below). `LambdaCodeS3Bucket`/
`LambdaCodeS3Key` are populated per-stage by this directory's `build.py` (a modified copy of
Task 6's — see Step 5 below), the same way `DataCaptureUploadPath` is populated from
`args.s3_bucket` and `stage_config["Parameters"]["StageName"]`.

`5-sagemaker-project/seed-code/deploy/batch/buildspec.yml`:
```yaml
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.12
    commands:
      - pip install --upgrade --force-reinstall "botocore>1.21.30" "boto3>1.18.30" "awscli>1.20.30"

  build:
    commands:
      - python build.py --model-execution-role "$MODEL_EXECUTION_ROLE_ARN" --model-package-group-name "$SOURCE_MODEL_PACKAGE_GROUP_NAME" --sagemaker-project-id "$SAGEMAKER_PROJECT_ID" --sagemaker-project-name "$SAGEMAKER_PROJECT_NAME" --s3-bucket "$ARTIFACT_BUCKET" --export-staging-config $EXPORT_TEMPLATE_STAGING_CONFIG --export-prod-config $EXPORT_TEMPLATE_PROD_CONFIG
      - cd lambda && zip -r ../run_transform.zip run_transform.py && cd ..
      - aws s3 cp run_transform.zip "s3://${ARTIFACT_BUCKET}/AutoML/lambda/staging/run_transform.zip"
      - aws s3 cp run_transform.zip "s3://${ARTIFACT_BUCKET}/AutoML/lambda/prod/run_transform.zip"
      - aws cloudformation package --template batch-transform-template.yml --s3-bucket $ARTIFACT_BUCKET --output-template $EXPORT_TEMPLATE_NAME
      - cat $EXPORT_TEMPLATE_STAGING_CONFIG
      - cat $EXPORT_TEMPLATE_PROD_CONFIG

artifacts:
  files:
    - $EXPORT_TEMPLATE_NAME
    - $EXPORT_TEMPLATE_STAGING_CONFIG
    - $EXPORT_TEMPLATE_PROD_CONFIG
```

- [ ] **Step 6: Write `serve_batch.py`** (adapted from
  `1-tabular-classification/2-inference/serve_batch.py`)

`5-sagemaker-project/seed-code/deploy/batch/serve_batch.py`:
```python
"""Pluggable AutoGluon batch transform inference handler.

Accepts headerless CSV (column order must match training data for tabular).
Adapted from 1-tabular-classification/2-inference/serve_batch.py.
"""
from io import StringIO

import pandas as pd


def model_fn(model_dir):
    from autogluon.tabular import TabularPredictor

    model = TabularPredictor.load(model_dir)
    globals()["column_names"] = model.feature_metadata_in.get_features()
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    if input_content_type != "text/csv":
        raise ValueError(f"{input_content_type} content type not supported")

    body_str = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    data = pd.read_csv(StringIO(body_str), header=None)
    if len(data.columns) != len(column_names):
        raise ValueError(f"Input has {len(data.columns)} columns but model expects {len(column_names)}")
    data.columns = column_names

    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    prediction = pd.concat([pred, pred_proba], axis=1)
    return prediction.to_json(), output_content_type
```

- [ ] **Step 7: Write `test/test.py` and `test/buildspec.yml`**

`5-sagemaker-project/seed-code/deploy/batch/test/test.py`:
```python
"""Invokes the batch-transform Lambda synchronously against a staging fixture
and asserts the transform job completes with expected output."""
import argparse
import json
import logging
import os

import boto3

logger = logging.getLogger(__name__)
lambda_client = boto3.client("lambda")
s3_client = boto3.client("s3")


def invoke_and_verify(function_name, bucket, input_prefix, output_prefix):
    payload = {
        "model_name": None,  # resolved by the Lambda's own env var at runtime
        "transform_job_name": "staging-test-transform",
        "input_s3_uri": f"s3://{bucket}/{input_prefix}",
        "output_s3_uri": f"s3://{bucket}/{output_prefix}",
        "instance_type": "ml.m5.xlarge",
        "instance_count": 1,
        "content_type": "text/csv",
    }
    response = lambda_client.invoke(
        FunctionName=function_name, InvocationType="RequestResponse", Payload=json.dumps(payload).encode()
    )
    result = json.loads(response["Payload"].read())
    if response.get("FunctionError"):
        raise Exception(f"Lambda invocation failed: {result}")
    if result.get("status") != "Completed":
        raise Exception(f"Transform job did not complete: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--import-build-config", type=str, required=True)
    parser.add_argument("--export-test-results", type=str, required=True)
    parser.add_argument("--fixture-bucket", type=str, required=True)
    parser.add_argument("--fixture-input-prefix", type=str, default="AutoML/batch-test-input/")
    parser.add_argument("--fixture-output-prefix", type=str, default="AutoML/batch-test-output/")
    args, _ = parser.parse_known_args()

    logging.basicConfig(format="%(levelname)s: [%(filename)s:%(lineno)s] %(message)s", level=args.log_level)

    with open(args.import_build_config) as f:
        config = json.load(f)

    function_name = "sagemaker-{}-{}-run-transform".format(
        config["Parameters"]["SageMakerProjectName"], config["Parameters"]["StageName"]
    )
    results = invoke_and_verify(function_name, args.fixture_bucket, args.fixture_input_prefix, args.fixture_output_prefix)

    with open(args.export_test_results, "w") as f:
        json.dump(results, f, indent=4)
```

`5-sagemaker-project/seed-code/deploy/batch/test/buildspec.yml`:
```yaml
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.12
  build:
    commands:
      - python test/test.py --import-build-config $CODEBUILD_SRC_DIR_BuildArtifact/staging-config-export.json --export-test-results $EXPORT_TEST_RESULTS --fixture-bucket $ARTIFACT_BUCKET
      - cat $EXPORT_TEST_RESULTS

artifacts:
  files:
    - $EXPORT_TEST_RESULTS
```

- [ ] **Step 8: Commit**

```bash
git add 5-sagemaker-project/seed-code/deploy/batch/
git commit -m "Add batch transform deploy pipeline seed code (Lambda, EventBridge schedule, test)"
```

---

### Task 8: `project-template.yaml` (CFN: Build + Deploy-RealTime + Deploy-Batch)

**Files:**
- Create: `5-sagemaker-project/cfn-templates/project-template.yaml`
- Test: manual `aws cloudformation validate-template` (no unit-testable code — pure CFN YAML)

**Interfaces:**
- Consumes: `SageMakerProjectName`, `SageMakerProjectId` (service-generated), plus the 6 repo
  full-names/branches and 1 `CodeConnectionArn` parameter — same shape as this account's proven
  `build-train-deploy-monitor-codepipeline.yaml`, trimmed to 2 repos (build, deploy — both deploy
  flavors share the `deploy` repo per the approved spec) instead of 3.
- Produces: 3 `AWS::CodePipeline::Pipeline` resources (`ModelBuildPipeline`,
  `ModelDeployRealTimePipeline`, `ModelDeployBatchPipeline`) and 2 `AWS::Events::Rule` resources
  (one per deploy pipeline, both firing on Model Package State Change). Registered by Task 9's
  setup notebook via `create_project`.

This drops the `GitSeedCodeCheckinProject`/Lambda custom-resource machinery present in the
account's existing template (per the approved spec) — the setup notebook pushes seed code
directly instead.

- [ ] **Step 1: Write the template**

`5-sagemaker-project/cfn-templates/project-template.yaml`:
```yaml
Description: >
  AutoGluon AutoML CI/CD: a SageMaker Pipeline (build/train/evaluate/register)
  triggered by CodePipeline, plus two deploy pipelines (real-time endpoint,
  scheduled batch transform) triggered by Model Registry approval events.
Parameters:
  SageMakerProjectName:
    Type: String
    AllowedPattern: ^[a-zA-Z](-*[a-zA-Z0-9])*
    MaxLength: 32
    MinLength: 1
  SageMakerProjectId:
    Type: String
  ModelBuildCodeRepositoryBranch:
    Type: String
    Default: main
  ModelBuildCodeRepositoryFullname:
    Type: String
    MaxLength: 1024
    Description: "GitHub owner/repo for the Build code, e.g. my-org/my-project-build"
  ModelDeployCodeRepositoryBranch:
    Type: String
    Default: main
  ModelDeployCodeRepositoryFullname:
    Type: String
    MaxLength: 1024
    Description: "GitHub owner/repo for the Deploy code (both realtime/ and batch/ subdirs), e.g. my-org/my-project-deploy"
  CodeConnectionArn:
    Type: String
    Description: "ARN of a CodeConnections connection tagged sagemaker=true"

Resources:
  MlOpsArtifactsBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub sagemaker-project-${SageMakerProjectId}
    DeletionPolicy: Retain

  # ===================== Build =====================
  ModelBuildProject:
    Type: AWS::CodeBuild::Project
    Properties:
      Name: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-modelbuild
      Artifacts: {Type: CODEPIPELINE}
      Environment:
        ComputeType: BUILD_GENERAL1_SMALL
        Image: aws/codebuild/amazonlinux2-x86_64-standard:5.0
        Type: LINUX_CONTAINER
        EnvironmentVariables:
          - {Name: SAGEMAKER_PROJECT_NAME, Value: !Ref SageMakerProjectName}
          - {Name: SAGEMAKER_PROJECT_ID, Value: !Ref SageMakerProjectId}
          - {Name: ARTIFACT_BUCKET, Value: !Ref MlOpsArtifactsBucket}
          - {Name: SAGEMAKER_PIPELINE_ROLE_ARN, Value: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsExecutionRole"}
          - {Name: AWS_REGION, Value: !Ref AWS::Region}
      ServiceRole: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCodeBuildRole"
      Source: {Type: CODEPIPELINE, BuildSpec: codebuild-buildspec.yml}
      TimeoutInMinutes: 480

  ModelBuildPipeline:
    Type: AWS::CodePipeline::Pipeline
    DependsOn: MlOpsArtifactsBucket
    Properties:
      Name: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-modelbuild
      RoleArn: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCodePipelineRole"
      PipelineType: V2
      ArtifactStore: {Type: S3, Location: !Ref MlOpsArtifactsBucket}
      Stages:
        - Name: Source
          Actions:
            - Name: ModelBuildWorkflowCode
              ActionTypeId: {Category: Source, Owner: AWS, Provider: CodeStarSourceConnection, Version: "1"}
              Configuration:
                ConnectionArn: !Ref CodeConnectionArn
                FullRepositoryId: !Ref ModelBuildCodeRepositoryFullname
                BranchName: !Ref ModelBuildCodeRepositoryBranch
              OutputArtifacts: [{Name: ModelBuildSourceArtifact}]
        - Name: Build
          Actions:
            - Name: BuildAndExecuteSageMakerPipeline
              ActionTypeId: {Category: Build, Owner: AWS, Provider: CodeBuild, Version: "1"}
              Configuration: {ProjectName: !Ref ModelBuildProject}
              InputArtifacts: [{Name: ModelBuildSourceArtifact}]
              OutputArtifacts: [{Name: ModelBuildBuildArtifact}]
              RunOrder: 1

  ModelBuildSagemakerCodeRepository:
    Type: AWS::SageMaker::CodeRepository
    Properties:
      CodeRepositoryName: !Sub sagemaker-${SageMakerProjectId}-modelbuild
      GitConfig:
        Branch: !Ref ModelBuildCodeRepositoryBranch
        RepositoryUrl: !Sub "https://github.com/${ModelBuildCodeRepositoryFullname}.git"

  # ===================== Deploy: Real-Time =====================
  ModelDeployRealTimeBuildProject:
    Type: AWS::CodeBuild::Project
    Properties:
      Name: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-deploy-rt
      Artifacts: {Type: CODEPIPELINE}
      Environment:
        ComputeType: BUILD_GENERAL1_SMALL
        Image: aws/codebuild/amazonlinux2-x86_64-standard:5.0
        Type: LINUX_CONTAINER
        EnvironmentVariables:
          - {Name: SAGEMAKER_PROJECT_NAME, Value: !Ref SageMakerProjectName}
          - {Name: SAGEMAKER_PROJECT_ID, Value: !Ref SageMakerProjectId}
          - {Name: ARTIFACT_BUCKET, Value: !Ref MlOpsArtifactsBucket}
          - {Name: MODEL_EXECUTION_ROLE_ARN, Value: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsExecutionRole"}
          - {Name: SOURCE_MODEL_PACKAGE_GROUP_NAME, Value: !Sub "${SageMakerProjectName}-${SageMakerProjectId}"}
          - {Name: AWS_REGION, Value: !Ref AWS::Region}
          - {Name: EXPORT_TEMPLATE_NAME, Value: template-export.yml}
          - {Name: EXPORT_TEMPLATE_STAGING_CONFIG, Value: staging-config-export.json}
          - {Name: EXPORT_TEMPLATE_PROD_CONFIG, Value: prod-config-export.json}
      ServiceRole: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCodeBuildRole"
      Source: {Type: CODEPIPELINE, BuildSpec: realtime/buildspec.yml}
      TimeoutInMinutes: 30

  ModelDeployRealTimeTestProject:
    Type: AWS::CodeBuild::Project
    Properties:
      Name: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-test-rt
      Artifacts: {Type: CODEPIPELINE}
      Environment:
        ComputeType: BUILD_GENERAL1_SMALL
        Image: aws/codebuild/amazonlinux2-x86_64-standard:5.0
        Type: LINUX_CONTAINER
        EnvironmentVariables:
          - {Name: AWS_REGION, Value: !Ref AWS::Region}
          - {Name: EXPORT_TEST_RESULTS, Value: test-results.json}
      ServiceRole: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCodeBuildRole"
      Source: {Type: CODEPIPELINE, BuildSpec: realtime/test/buildspec.yml}
      TimeoutInMinutes: 30

  ModelDeployRealTimePipeline:
    Type: AWS::CodePipeline::Pipeline
    DependsOn: MlOpsArtifactsBucket
    Properties:
      Name: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-deploy-rt
      RoleArn: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCodePipelineRole"
      PipelineType: V2
      ArtifactStore: {Type: S3, Location: !Ref MlOpsArtifactsBucket}
      Stages:
        - Name: Source
          Actions:
            - Name: ModelDeployInfraCode
              ActionTypeId: {Category: Source, Owner: AWS, Provider: CodeStarSourceConnection, Version: "1"}
              Configuration:
                ConnectionArn: !Ref CodeConnectionArn
                FullRepositoryId: !Ref ModelDeployCodeRepositoryFullname
                BranchName: !Ref ModelDeployCodeRepositoryBranch
              OutputArtifacts: [{Name: SourceArtifact}]
        - Name: Build
          Actions:
            - Name: BuildDeploymentTemplates
              ActionTypeId: {Category: Build, Owner: AWS, Provider: CodeBuild, Version: "1"}
              Configuration: {ProjectName: !Ref ModelDeployRealTimeBuildProject}
              InputArtifacts: [{Name: SourceArtifact}]
              OutputArtifacts: [{Name: BuildArtifact}]
              RunOrder: 1
        - Name: DeployStaging
          Actions:
            - Name: DeployResourcesStaging
              ActionTypeId: {Category: Deploy, Owner: AWS, Provider: CloudFormation, Version: "1"}
              Configuration:
                ActionMode: REPLACE_ON_FAILURE
                Capabilities: CAPABILITY_NAMED_IAM
                RoleArn: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCloudformationRole"
                StackName: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-deploy-rt-staging
                TemplateConfiguration: BuildArtifact::staging-config-export.json
                TemplatePath: BuildArtifact::template-export.yml
              InputArtifacts: [{Name: BuildArtifact}]
              RunOrder: 1
            - Name: TestStaging
              ActionTypeId: {Category: Build, Owner: AWS, Provider: CodeBuild, Version: "1"}
              Configuration: {ProjectName: !Ref ModelDeployRealTimeTestProject, PrimarySource: SourceArtifact}
              InputArtifacts: [{Name: SourceArtifact}, {Name: BuildArtifact}]
              OutputArtifacts: [{Name: TestArtifact}]
              RunOrder: 2
            - Name: ApproveDeployment
              ActionTypeId: {Category: Approval, Owner: AWS, Provider: Manual, Version: "1"}
              Configuration: {CustomData: "Approve this model for Production (real-time endpoint)"}
              RunOrder: 3
        - Name: DeployProd
          Actions:
            - Name: DeployResourcesProd
              ActionTypeId: {Category: Deploy, Owner: AWS, Provider: CloudFormation, Version: "1"}
              Configuration:
                ActionMode: CREATE_UPDATE
                Capabilities: CAPABILITY_NAMED_IAM
                RoleArn: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCloudformationRole"
                StackName: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-deploy-rt-prod
                TemplateConfiguration: BuildArtifact::prod-config-export.json
                TemplatePath: BuildArtifact::template-export.yml
              InputArtifacts: [{Name: BuildArtifact}]
              RunOrder: 1

  ModelDeployRealTimeEventRule:
    Type: AWS::Events::Rule
    Properties:
      Name: !Sub sagemaker-${SageMakerProjectId}-rt-model
      State: ENABLED
      EventPattern:
        source: [aws.sagemaker]
        detail-type: ["SageMaker Model Package State Change"]
        detail:
          ModelPackageGroupName: [!Sub "${SageMakerProjectName}-${SageMakerProjectId}"]
          ModelApprovalStatus: [{"anything-but": ["PendingManualApproval"]}]
      Targets:
        - Arn: !Sub "arn:${AWS::Partition}:codepipeline:${AWS::Region}:${AWS::AccountId}:${ModelDeployRealTimePipeline}"
          Id: !Sub sagemaker-${SageMakerProjectName}-rt-trigger
          RoleArn: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsEventsRole"

  # ===================== Deploy: Batch =====================
  ModelDeployBatchBuildProject:
    Type: AWS::CodeBuild::Project
    Properties:
      Name: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-deploy-batch
      Artifacts: {Type: CODEPIPELINE}
      Environment:
        ComputeType: BUILD_GENERAL1_SMALL
        Image: aws/codebuild/amazonlinux2-x86_64-standard:5.0
        Type: LINUX_CONTAINER
        EnvironmentVariables:
          - {Name: SAGEMAKER_PROJECT_NAME, Value: !Ref SageMakerProjectName}
          - {Name: SAGEMAKER_PROJECT_ID, Value: !Ref SageMakerProjectId}
          - {Name: ARTIFACT_BUCKET, Value: !Ref MlOpsArtifactsBucket}
          - {Name: MODEL_EXECUTION_ROLE_ARN, Value: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsExecutionRole"}
          - {Name: SOURCE_MODEL_PACKAGE_GROUP_NAME, Value: !Sub "${SageMakerProjectName}-${SageMakerProjectId}"}
          - {Name: AWS_REGION, Value: !Ref AWS::Region}
          - {Name: EXPORT_TEMPLATE_NAME, Value: template-export.yml}
          - {Name: EXPORT_TEMPLATE_STAGING_CONFIG, Value: staging-config-export.json}
          - {Name: EXPORT_TEMPLATE_PROD_CONFIG, Value: prod-config-export.json}
      ServiceRole: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCodeBuildRole"
      Source: {Type: CODEPIPELINE, BuildSpec: batch/buildspec.yml}
      TimeoutInMinutes: 30

  ModelDeployBatchTestProject:
    Type: AWS::CodeBuild::Project
    Properties:
      Name: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-test-batch
      Artifacts: {Type: CODEPIPELINE}
      Environment:
        ComputeType: BUILD_GENERAL1_SMALL
        Image: aws/codebuild/amazonlinux2-x86_64-standard:5.0
        Type: LINUX_CONTAINER
        EnvironmentVariables:
          - {Name: ARTIFACT_BUCKET, Value: !Ref MlOpsArtifactsBucket}
          - {Name: AWS_REGION, Value: !Ref AWS::Region}
          - {Name: EXPORT_TEST_RESULTS, Value: test-results.json}
      ServiceRole: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCodeBuildRole"
      Source: {Type: CODEPIPELINE, BuildSpec: batch/test/buildspec.yml}
      TimeoutInMinutes: 60

  ModelDeployBatchPipeline:
    Type: AWS::CodePipeline::Pipeline
    DependsOn: MlOpsArtifactsBucket
    Properties:
      Name: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-deploy-batch
      RoleArn: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCodePipelineRole"
      PipelineType: V2
      ArtifactStore: {Type: S3, Location: !Ref MlOpsArtifactsBucket}
      Stages:
        - Name: Source
          Actions:
            - Name: ModelDeployInfraCode
              ActionTypeId: {Category: Source, Owner: AWS, Provider: CodeStarSourceConnection, Version: "1"}
              Configuration:
                ConnectionArn: !Ref CodeConnectionArn
                FullRepositoryId: !Ref ModelDeployCodeRepositoryFullname
                BranchName: !Ref ModelDeployCodeRepositoryBranch
              OutputArtifacts: [{Name: SourceArtifact}]
        - Name: Build
          Actions:
            - Name: BuildDeploymentTemplates
              ActionTypeId: {Category: Build, Owner: AWS, Provider: CodeBuild, Version: "1"}
              Configuration: {ProjectName: !Ref ModelDeployBatchBuildProject}
              InputArtifacts: [{Name: SourceArtifact}]
              OutputArtifacts: [{Name: BuildArtifact}]
              RunOrder: 1
        - Name: DeployStaging
          Actions:
            - Name: DeployResourcesStaging
              ActionTypeId: {Category: Deploy, Owner: AWS, Provider: CloudFormation, Version: "1"}
              Configuration:
                ActionMode: REPLACE_ON_FAILURE
                Capabilities: CAPABILITY_NAMED_IAM
                RoleArn: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCloudformationRole"
                StackName: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-deploy-batch-staging
                TemplateConfiguration: BuildArtifact::staging-config-export.json
                TemplatePath: BuildArtifact::template-export.yml
              InputArtifacts: [{Name: BuildArtifact}]
              RunOrder: 1
            - Name: TestStaging
              ActionTypeId: {Category: Build, Owner: AWS, Provider: CodeBuild, Version: "1"}
              Configuration: {ProjectName: !Ref ModelDeployBatchTestProject, PrimarySource: SourceArtifact}
              InputArtifacts: [{Name: SourceArtifact}, {Name: BuildArtifact}]
              OutputArtifacts: [{Name: TestArtifact}]
              RunOrder: 2
            - Name: ApproveDeployment
              ActionTypeId: {Category: Approval, Owner: AWS, Provider: Manual, Version: "1"}
              Configuration: {CustomData: "Approve this model for Production (scheduled batch transform)"}
              RunOrder: 3
        - Name: DeployProd
          Actions:
            - Name: DeployResourcesProd
              ActionTypeId: {Category: Deploy, Owner: AWS, Provider: CloudFormation, Version: "1"}
              Configuration:
                ActionMode: CREATE_UPDATE
                Capabilities: CAPABILITY_NAMED_IAM
                RoleArn: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsCloudformationRole"
                StackName: !Sub sagemaker-${SageMakerProjectName}-${SageMakerProjectId}-deploy-batch-prod
                TemplateConfiguration: BuildArtifact::prod-config-export.json
                TemplatePath: BuildArtifact::template-export.yml
              InputArtifacts: [{Name: BuildArtifact}]
              RunOrder: 1

  ModelDeployBatchEventRule:
    Type: AWS::Events::Rule
    Properties:
      Name: !Sub sagemaker-${SageMakerProjectId}-batch-model
      State: ENABLED
      EventPattern:
        source: [aws.sagemaker]
        detail-type: ["SageMaker Model Package State Change"]
        detail:
          ModelPackageGroupName: [!Sub "${SageMakerProjectName}-${SageMakerProjectId}"]
          ModelApprovalStatus: [{"anything-but": ["PendingManualApproval"]}]
      Targets:
        - Arn: !Sub "arn:${AWS::Partition}:codepipeline:${AWS::Region}:${AWS::AccountId}:${ModelDeployBatchPipeline}"
          Id: !Sub sagemaker-${SageMakerProjectName}-batch-trigger
          RoleArn: !Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/service-role/AmazonSageMakerProjectsEventsRole"

  ModelDeploySagemakerCodeRepository:
    Type: AWS::SageMaker::CodeRepository
    Properties:
      CodeRepositoryName: !Sub sagemaker-${SageMakerProjectId}-modeldeploy
      GitConfig:
        Branch: !Ref ModelDeployCodeRepositoryBranch
        RepositoryUrl: !Sub "https://github.com/${ModelDeployCodeRepositoryFullname}.git"

Outputs:
  ModelBuildPipeline:
    Value: !Sub "https://console.aws.amazon.com/codesuite/codepipeline/pipelines/${ModelBuildPipeline}/view?region=${AWS::Region}"
  ModelDeployRealTimePipeline:
    Value: !Sub "https://console.aws.amazon.com/codesuite/codepipeline/pipelines/${ModelDeployRealTimePipeline}/view?region=${AWS::Region}"
  ModelDeployBatchPipeline:
    Value: !Sub "https://console.aws.amazon.com/codesuite/codepipeline/pipelines/${ModelDeployBatchPipeline}/view?region=${AWS::Region}"
```

- [ ] **Step 2: Validate the template syntax**

```bash
aws cloudformation validate-template --template-body file://5-sagemaker-project/cfn-templates/project-template.yaml
```

Expected: JSON output echoing back `Parameters` (8 entries) with no error. If it fails, the
error message names the offending resource/property — fix and re-run before proceeding (do not
skip this check; template errors surface much later and more confusingly inside `create_project`
otherwise).

- [ ] **Step 3: Commit**

```bash
git add 5-sagemaker-project/cfn-templates/project-template.yaml
git commit -m "Add SageMaker Project CFN template (Build + Deploy-RealTime + Deploy-Batch)"
```

---

### Task 9: `0-project-setup/setup_project.ipynb`

**Files:**
- Create: `5-sagemaker-project/0-project-setup/setup_project.ipynb`

**Interfaces:**
- Consumes: `5-sagemaker-project/cfn-templates/project-template.yaml` (Task 8),
  `5-sagemaker-project/seed-code/{build,deploy}/` (Tasks 1–7).
- Produces: two new GitHub repos with seed code pushed, one SageMaker Project
  (`create_project` with `TemplateProviders`), printed console URLs for all 3 CodePipelines.

This is a notebook (matches the rest of this repo's `*.ipynb`-based workflow convention), so it
isn't unit-tested — it's validated by actually running it end-to-end against this AWS account in
Task 10.

- [ ] **Step 1: Write the notebook**

Create `5-sagemaker-project/0-project-setup/setup_project.ipynb` with these cells (markdown
cells omitted below for brevity — include a one-line markdown header above each code cell in the
actual notebook, consistent with this repo's existing notebooks' style):

```python
# Cell 1: Configuration
import json
import subprocess
import time

import boto3
import sagemaker

REGION = boto3.session.Session().region_name
sess = sagemaker.session.Session()
BUCKET = sess.default_bucket()

PROJECT_NAME = "autogluon-automl"  # must match ^[a-zA-Z](-*[a-zA-Z0-9])*, max 32 chars
GITHUB_OWNER = subprocess.run(["gh", "api", "user", "--jq", ".login"], capture_output=True, text=True, check=True).stdout.strip()
BUILD_REPO = f"{GITHUB_OWNER}/{PROJECT_NAME}-build"
DEPLOY_REPO = f"{GITHUB_OWNER}/{PROJECT_NAME}-deploy"

sm_client = boto3.client("sagemaker", region_name=REGION)
codeconnections_client = boto3.client("codeconnections", region_name=REGION)

print(f"Region: {REGION}")
print(f"Bucket: {BUCKET}")
print(f"Build repo:  {BUILD_REPO}")
print(f"Deploy repo: {DEPLOY_REPO}")
```

```python
# Cell 2: Create GitHub repos
import os

for repo, seed_dir in [(BUILD_REPO, "../seed-code/build"), (DEPLOY_REPO, "../seed-code/deploy")]:
    subprocess.run(["gh", "repo", "create", repo, "--private", "--confirm"], check=True)
    print(f"Created {repo}")

    tmp_clone = f"/tmp/{repo.split('/')[-1]}"
    subprocess.run(["rm", "-rf", tmp_clone])
    subprocess.run(["git", "clone", f"https://github.com/{repo}.git", tmp_clone], check=True)
    subprocess.run(f"cp -r {seed_dir}/* {tmp_clone}/", shell=True, check=True)
    subprocess.run(["git", "-C", tmp_clone, "add", "-A"], check=True)
    subprocess.run(["git", "-C", tmp_clone, "commit", "-m", "Initial seed code"], check=True)
    subprocess.run(["git", "-C", tmp_clone, "push", "origin", "main"], check=True)
    print(f"Pushed seed code to {repo}")
```

```python
# Cell 3: Find or create a CodeConnections connection tagged sagemaker=true
def find_sagemaker_connection():
    connections = codeconnections_client.list_connections()["Connections"]
    for conn in connections:
        if conn["ConnectionStatus"] != "AVAILABLE":
            continue
        tags = codeconnections_client.list_tags_for_resource(ResourceArn=conn["ConnectionArn"])["Tags"]
        if any(t["Key"] == "sagemaker" and t["Value"].lower() == "true" for t in tags):
            return conn["ConnectionArn"]
    return None

connection_arn = find_sagemaker_connection()

if connection_arn is None:
    create_response = codeconnections_client.create_connection(
        ProviderType="GitHub",
        ConnectionName=f"{PROJECT_NAME}-connection",
        Tags=[{"Key": "sagemaker", "Value": "true"}],
    )
    connection_arn = create_response["ConnectionArn"]
    print(f"Created connection {connection_arn} — go to the CodeConnections console to authorize it with GitHub, then re-run this cell.")
    print("Console: https://console.aws.amazon.com/codesuite/settings/connections")
else:
    print(f"Using existing connection: {connection_arn}")
```

```python
# Cell 4: Poll until the connection is AVAILABLE (only needed if Cell 3 just created one)
while True:
    status = codeconnections_client.get_connection(ConnectionArn=connection_arn)["Connection"]["ConnectionStatus"]
    print(f"Connection status: {status}")
    if status == "AVAILABLE":
        break
    print("Waiting 30s for you to authorize the connection in the console...")
    time.sleep(30)
```

```python
# Cell 5: Upload the CFN template to S3
TEMPLATE_KEY = "sagemaker-projects-templates/autogluon-automl-project-template.yaml"
s3_client = boto3.client("s3", region_name=REGION)
s3_client.upload_file("../cfn-templates/project-template.yaml", BUCKET, TEMPLATE_KEY)
template_url = f"https://{BUCKET}.s3.{REGION}.amazonaws.com/{TEMPLATE_KEY}"
print(f"Template uploaded: {template_url}")
```

```python
# Cell 6: Create the SageMaker Project
create_response = sm_client.create_project(
    ProjectName=PROJECT_NAME,
    ProjectDescription="AutoGluon AutoML CI/CD (build/train/evaluate/register + real-time and batch deploy)",
    TemplateProviders=[{
        "CfnTemplateProvider": {
            "TemplateName": "AutoGluonAutoMLProjectTemplate",
            "TemplateURL": template_url,
            "Parameters": [
                {"Key": "ModelBuildCodeRepositoryFullname", "Value": BUILD_REPO},
                {"Key": "ModelDeployCodeRepositoryFullname", "Value": DEPLOY_REPO},
                {"Key": "CodeConnectionArn", "Value": connection_arn},
            ],
        }
    }],
)
print(f"Project ARN: {create_response['ProjectArn']}")
```

```python
# Cell 7: Poll until CreateCompleted
while True:
    status = sm_client.describe_project(ProjectName=PROJECT_NAME)["ProjectStatus"]
    print(f"Project status: {status}")
    if status in ("CreateCompleted", "CreateFailed"):
        break
    time.sleep(30)

if status == "CreateFailed":
    raise Exception("Project creation failed — check the CloudFormation stack events in the console.")

project_id = sm_client.describe_project(ProjectName=PROJECT_NAME)["ProjectId"]
print(f"\nProject ready. ProjectId: {project_id}")
print(f"Build pipeline:        https://console.aws.amazon.com/codesuite/codepipeline/pipelines/sagemaker-{PROJECT_NAME}-{project_id}-modelbuild/view?region={REGION}")
print(f"Deploy (real-time):    https://console.aws.amazon.com/codesuite/codepipeline/pipelines/sagemaker-{PROJECT_NAME}-{project_id}-deploy-rt/view?region={REGION}")
print(f"Deploy (batch):        https://console.aws.amazon.com/codesuite/codepipeline/pipelines/sagemaker-{PROJECT_NAME}-{project_id}-deploy-batch/view?region={REGION}")
```

- [ ] **Step 2: Commit**

```bash
git add 5-sagemaker-project/0-project-setup/setup_project.ipynb
git commit -m "Add SageMaker Project setup notebook"
```

---

### Task 10: End-to-end verification against real AWS resources

**Files:** none created — this task runs the notebook and pipelines for real and records the
outcome. If any step fails, the fix lands in whichever file caused it (Tasks 1–9), followed by a
new commit referencing this task.

**Interfaces:** N/A — this is the acceptance test for the whole feature.

- [ ] **Step 1: Run `setup_project.ipynb` end-to-end**

Execute all cells in `5-sagemaker-project/0-project-setup/setup_project.ipynb` (via Jupyter or
`jupyter nbconvert --execute`). Confirm:
- Two GitHub repos created and populated (`gh repo view <BUILD_REPO>`, `gh repo view
  <DEPLOY_REPO>`).
- CodeConnections connection is `AVAILABLE`.
- `create_project` returns a `ProjectArn` and `describe_project` eventually reports
  `ProjectStatus: CreateCompleted`.

- [ ] **Step 2: Verify the Build pipeline runs to completion**

```bash
aws codepipeline get-pipeline-state --name sagemaker-autogluon-automl-<project-id>-modelbuild
```

Expected: `Source` and `Build` stages both show `Succeeded`. If `Build` fails, fetch CodeBuild
logs:

```bash
aws codebuild batch-get-builds --ids <build-id-from-above> --query 'builds[0].logs.deepLink' --output text
```

Fix the root cause in the relevant seed-code file (most likely `pipeline.py`'s SDK call shapes,
which can drift between SDK point releases) and re-push:

```bash
cd /tmp/<project-name>-build   # the clone created by the setup notebook
# apply fix, mirroring it back into 5-sagemaker-project/seed-code/build/ in this repo
git add -A && git commit -m "Fix build pipeline issue found during e2e verification" && git push
```

- [ ] **Step 3: Verify a model was registered with `PendingManualApproval`**

```bash
aws sagemaker list-model-packages --model-package-group-name autogluon-automl-<project-id> --query 'ModelPackageSummaryList[0]'
```

Expected: one entry with `"ModelApprovalStatus": "PendingManualApproval"`.

- [ ] **Step 4: Approve the model package and verify both Deploy pipelines trigger**

```bash
aws sagemaker update-model-package \
  --model-package-arn <arn-from-step-3> \
  --model-approval-status Approved
```

```bash
aws codepipeline get-pipeline-state --name sagemaker-autogluon-automl-<project-id>-deploy-rt
aws codepipeline get-pipeline-state --name sagemaker-autogluon-automl-<project-id>-deploy-batch
```

Expected: both pipelines show a new execution in progress or in `DeployStaging`, confirming the
`AWS::Events::Rule`s fired correctly.

- [ ] **Step 5: Verify staging deploys succeed and manually approve prod for both**

For real-time: confirm `describe_endpoint --endpoint-name autogluon-automl-staging` shows
`EndpointStatus: InService`, confirm the `TestStaging` CodeBuild action passed, then approve via:

```bash
aws codepipeline put-approval-result \
  --pipeline-name sagemaker-autogluon-automl-<project-id>-deploy-rt \
  --stage-name DeployStaging --action-name ApproveDeployment \
  --result "summary=verified,status=Approved" --token <token-from-get-pipeline-state>
```

For batch: confirm the `TestStaging` CodeBuild action's Lambda invocation succeeded (check
`test-results.json` in the pipeline artifact), then approve the same way against the
`deploy-batch` pipeline.

- [ ] **Step 6: Verify prod resources came up**

```bash
aws sagemaker describe-endpoint --endpoint-name autogluon-automl-prod --query EndpointStatus
aws events describe-rule --name sagemaker-<project-id>-batch-model --query State
```

Expected: `InService`, and the prod batch schedule rule shows `ENABLED` (per `ScheduleEnabled:
"true"` in `prod-config.json`).

- [ ] **Step 7: Clean up all created AWS/GitHub resources**

```bash
aws sagemaker delete-endpoint --endpoint-name autogluon-automl-staging
aws sagemaker delete-endpoint --endpoint-name autogluon-automl-prod
aws sagemaker delete-project --project-name autogluon-automl
gh repo delete <BUILD_REPO> --yes
gh repo delete <DEPLOY_REPO> --yes
aws s3 rm s3://<BUCKET>/AutoML/ --recursive
aws s3 rm s3://<BUCKET>/sagemaker-projects-templates/autogluon-automl-project-template.yaml
```

Confirm with the user before running deletions, per this session's production-safety
instructions.

---

### Task 11: `5-sagemaker-project/README.md`

**Files:**
- Create: `5-sagemaker-project/README.md`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Write the README**, following the structure of `4-custom-image/README.md`
  (Overview table, Repository Structure, Prerequisites, Workflow, Cleanup):

`5-sagemaker-project/README.md`:
```markdown
# SageMaker Projects: AutoGluon AutoML CI/CD

Use [SageMaker Projects](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects.html)
as a full MLOps CI/CD platform for AutoGluon AutoML: a custom CloudFormation template (stored in
Amazon S3) provisions a Build pipeline (SageMaker Pipeline: preprocess -> train -> evaluate ->
register) and two Deploy pipelines (real-time endpoint, scheduled batch transform), each gated
staging -> manual approval -> production.

This experiment is a **cookie-cutter starting point**: `train.py`/`evaluate.py` are pluggable
across AutoGluon's tabular, timeseries, and multimodal predictors via a `task_type` field in
`config/*.yaml`, so adopting teams swap in their own dataset/config rather than rewriting the
pipeline.

## Repository Structure

```
0-project-setup/
  setup_project.ipynb      # creates GitHub repos, CodeConnections, uploads CFN, creates the Project
cfn-templates/
  project-template.yaml    # Build + Deploy(RealTime) + Deploy(Batch), registered via create_project
seed-code/
  build/                   # pushed to <project>-build — SageMaker Pipeline + CodeBuild
  deploy/                  # pushed to <project>-deploy — realtime/ and batch/ deploy pipelines
```

## Prerequisites

- Everything in the root [README](../README.md#prerequisites), plus:
- [GitHub CLI](https://cli.github.com/) (`gh`), authenticated (`gh auth login`)
- A GitHub account/org where new repos can be created
- A [CodeConnections](https://docs.aws.amazon.com/dtconsole/latest/userguide/welcome-connections.html)
  connection to GitHub, tagged `sagemaker=true` (the setup notebook creates one if none exists —
  authorizing it with GitHub is a one-time console step that cannot be scripted)
- This account's `AmazonSageMakerProjectsCloudformationRole` must permit creating a Lambda
  function, IAM role, and EventBridge rule matching the `sagemaker-*` naming prefix (see
  `cfn-templates/iam-policy-patch.json` — a one-time account setup step, already applied in this
  account as part of building this experiment)

## Workflow

1. Run `0-project-setup/setup_project.ipynb` top to bottom. It creates two GitHub repos, an
   S3-hosted CFN template, and the SageMaker Project itself.
2. Push a change to the `<project>-build` repo (or wait for the initial push) to trigger the
   Build pipeline: preprocess -> train -> evaluate -> register (`PendingManualApproval`).
3. Approve the registered model package in the SageMaker Studio Model Registry UI (or via
   `aws sagemaker update-model-package --model-approval-status Approved`).
4. Both Deploy pipelines trigger automatically off the approval event, deploying to staging,
   running an automated test, then waiting for manual approval before deploying to production.

## Bringing Your Own Dataset

- **Tabular** (default, works out of the box): point `InputDataUri` at your own CSV and update
  `label`/`eval_metric` in `seed-code/build/config/tabular.yaml`.
- **Timeseries / Multimodal**: replace `seed-code/build/pipelines/automl/preprocess.py` with a
  script matching your data shape (see `2-timeseries-forecasting/0-data-prep/preprocess.py` and
  `3-multimodal/0-data-prep/preprocess.py` in this repo for worked examples), then use
  `config/timeseries.yaml`/`config/multimodal.yaml` as your starting config.

## Cleanup

```bash
aws sagemaker delete-endpoint --endpoint-name <project-name>-staging
aws sagemaker delete-endpoint --endpoint-name <project-name>-prod
aws sagemaker delete-project --project-name <project-name>
gh repo delete <owner>/<project-name>-build --yes
gh repo delete <owner>/<project-name>-deploy --yes
aws s3 rm s3://<bucket>/AutoML/ --recursive
```
```

- [ ] **Step 2: Update the root `README.md` experiments table**

Modify `README.md:7-13` (the experiments table) to add a fifth row:

```markdown
| **SageMaker Project** | CI/CD MLOps platform | Pluggable (tabular/timeseries/multimodal) | Same as AutoGluon task |
```

And modify `README.md:35-39` (repository structure block) to append:

```
5-sagemaker-project/              # SageMaker Projects CI/CD (Build + Deploy pipelines)
  0-project-setup/
  cfn-templates/
  seed-code/{build,deploy}/
```

- [ ] **Step 3: Commit**

```bash
git add 5-sagemaker-project/README.md README.md
git commit -m "Add 5-sagemaker-project README and update root README experiments table"
```

---

## Self-Review Notes

**Spec coverage:** Task 0 covers the IAM prerequisite surfaced during design research (not in
the original spec text, but required for the spec's batch-deploy requirement to actually work —
added per the user's explicit approval). Tasks 1–5 cover the Build pipeline and pluggable
train/evaluate. Tasks 6–7 cover both Deploy flavors. Task 8 covers the CFN template. Task 9
covers the setup notebook. Task 10 is the spec's implicit "does this actually work" requirement.
Task 11 covers the spec's Cleanup section and root README update.

**Placeholder scan:** no TBD/TODO; every code step has complete code; the one deliberately
incomplete-looking piece (Task 7 Step 5's `!Select`-based S3 bucket derivation) is explicitly
called out as fragile with a concrete simplification to apply, not left as an open question.

**Type consistency:** `task_type` values (`tabular`/`timeseries`/`multimodal`) are consistent
across Task 2's config, Task 3's `train.py`, and Task 4's `evaluate.py`. `evaluation.json` shape
(`{"metrics": {<eval_metric>: <value>}}`) is consistent between Task 3, Task 4, and Task 5's
`ConditionStep`. `ModelPackageGroupName` naming (`${SageMakerProjectName}-${SageMakerProjectId}`)
is consistent between Task 5's `pipeline.py` kwarg, Task 8's CFN `SOURCE_MODEL_PACKAGE_GROUP_NAME`
env var, and Task 9's notebook.
