"""CLI entry: python -m langgraph.checkpoint.validation <file> [options]."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from langgraph.checkpoint.validation.initializer import get_registry
from langgraph.checkpoint.validation.report import ProgressCallbacks
from langgraph.checkpoint.validation.validate import validate


def _load_module(path: str) -> None:
    """Load a Python file to trigger @checkpointer_test registration."""
    p = Path(path).resolve()
    if not p.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_user_checkpointer", str(p))
    if spec is None or spec.loader is None:
        print(f"Error: cannot load module from {path}", file=sys.stderr)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


async def _run(args: argparse.Namespace) -> int:
    """Run validation and return exit code."""
    _load_module(args.file)

    registry = get_registry()
    if not registry:
        print(
            "Error: no @checkpointer_test decorators found in the file.",
            file=sys.stderr,
        )
        return 1

    capabilities: set[str] | None = None
    if args.capabilities:
        capabilities = {c.strip() for c in args.capabilities.split(",")}

    # Select progress style based on flags.
    if args.json or args.quiet:
        progress = None
    elif args.verbose:
        progress = ProgressCallbacks.verbose()
    else:
        progress = ProgressCallbacks.default()

    exit_code = 0
    for _name, registered in registry.items():
        report = await validate(
            registered, capabilities=capabilities, progress=progress
        )

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            report.print_report()

        if not report.passed_all_base():
            exit_code = 1

    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="langgraph.checkpoint.validation",
        description="Run checkpointer conformance tests.",
    )
    parser.add_argument("file", help="Python file with @checkpointer_test decorator(s)")
    parser.add_argument(
        "--capabilities",
        help="Comma-separated list of capabilities to test (default: all detected)",
        default=None,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (implies --quiet)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show each test result with name",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="No progress output (summary only)",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
