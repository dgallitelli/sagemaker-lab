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
