"""Pluggable AutoGluon training script for SageMaker (tabular/timeseries/multimodal).

Runs inside the AutoGluon DLC container. Reads a YAML config from the `config`
channel that sets `task_type` (tabular|timeseries|multimodal), then dispatches
to the matching AutoGluon predictor. Writes evaluation.json in the shape
{"metrics": {<eval_metric>: <value>}} for the pipeline's ConditionStep.
"""
import argparse
import json
import os
import shutil
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
        eval_metric_name = predictor.eval_metric.name if hasattr(predictor.eval_metric, "name") else str(predictor.eval_metric)

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

    # Package inference code into the model artifact so the registered model package's
    # container actually knows how to serve it. The AutoGluon inference DLC (TorchServe-based)
    # requires code/<entry point> inside the model.tar.gz, matching the SAGEMAKER_PROGRAM /
    # SAGEMAKER_SUBMIT_DIRECTORY environment variables set on the model package's container
    # (see pipeline.py's ModelBuilder(source_code=SourceCode(..., entry_script="serve.py"))) --
    # without it, the container falls back to a default PyTorch handler that crashes on load
    # (AutoGluon's output is predictor pickle files, not a .pth/.pt checkpoint).
    #
    # ModelTrainer (SDK v3) mounts its SourceCode as a distinct "code" input *channel* at
    # /opt/ml/input/data/code (env var SM_CHANNEL_CODE) -- NOT at /opt/ml/code, which is the
    # legacy Estimator/sagemaker-training-toolkit convention. Confirmed directly from a real
    # training job's own logs: SM_CHANNEL_CODE=/opt/ml/input/data/code, and a first attempt at
    # this fix using /opt/ml/code found nothing there and logged the "not found" warning below
    # every time. Copy serve.py from the SM_CHANNEL_CODE directory into {model_dir}/code/serve.py
    # (same filename, so it matches SAGEMAKER_PROGRAM=serve.py exactly) so it ends up inside the
    # model.tar.gz that SageMaker automatically uploads from SM_MODEL_DIR. This is deliberately
    # done here (at training time) rather than relying on ModelBuilder/ModelStep's "runtime
    # repack" mechanism: that mechanism silently never triggers for ModelBuilder.register()
    # under a PipelineSession, because ModelStep's _append_repack_model_step() only recognizes
    # plain sagemaker.core.resources.Model/PipelineModel instances via isinstance(), not
    # ModelBuilder (confirmed by reading sagemaker-serve==1.16.0's and sagemaker-mlops==1.16.0's
    # source directly) -- so SAGEMAKER_PROGRAM/SAGEMAKER_SUBMIT_DIRECTORY end up correctly set
    # on the registered model package's container environment, but the actual code/ directory
    # never gets added to the model artifact, and the container still crashes on load. Doing it
    # here avoids depending on that broken code path entirely.
    submit_dir = os.environ.get("SM_CHANNEL_CODE", "/opt/ml/input/data/code")
    serve_script_src = os.path.join(submit_dir, "serve.py")
    if os.path.exists(serve_script_src):
        code_dir = os.path.join(save_path, "code")
        os.makedirs(code_dir, exist_ok=True)
        shutil.copy2(serve_script_src, os.path.join(code_dir, "serve.py"))
        print(f"Copied {serve_script_src} to {os.path.join(code_dir, 'serve.py')}")
    else:
        print(f"WARN: {serve_script_src} not found; model package will not have inference code")


if __name__ == "__main__":
    main()
