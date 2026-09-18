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
