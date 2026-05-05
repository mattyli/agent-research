"""
CLI entrypoint for the native benchmark harness.

Usage:
    /Users/02matt/.hermes/hermes-agent/venv/bin/python3 -m src.native \
        --config configs/native/hermes_disabled_memory.yaml \
        --task-id task1_1
"""
import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.native.experiments.config_schema import NativeRunConfig
from src.native.driver import NativeBenchDriver


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MedAgentBench tasks through the native Hermes harness.",
    )
    parser.add_argument(
        "--config",
        default="configs/native/hermes_disabled_memory.yaml",
        help="Path to NativeRunConfig YAML (default: configs/native/hermes_disabled_memory.yaml).",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        metavar="ID",
        help="Run a single task by ID, e.g. task1_1. Omit to run all tasks in the config.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = NativeRunConfig.from_yaml(args.config)

    if args.task_id:
        config.benchmark.task_ids = [args.task_id]

    driver = NativeBenchDriver(config)
    summary = driver.run()

    print(
        f"\nRun complete: {summary['success_rate']:.1%} "
        f"({summary['successes']}/{summary['total']}) tasks succeeded"
    )
    print(f"Output directory: {driver._output_root}")


if __name__ == "__main__":
    main()
