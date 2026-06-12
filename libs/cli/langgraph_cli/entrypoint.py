"""User-facing entrypoint for the LangGraph CLI."""

from __future__ import annotations

import os
import pathlib
import sys
from collections.abc import Sequence

import click

from .cli import cli

_GO_CLI_FLAG = "LANGGRAPH_USE_GO_CLI"
_GO_CLI_PATH_ENV = "LANGGRAPH_GO_CLI_PATH"
_CALLING_PYTHON_ENV = "LANGGRAPH_CALLING_PYTHON"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _legacy_cli(argv: Sequence[str] | None = None) -> None:
    cli.main(args=list(argv) if argv is not None else None, prog_name="langgraph")


def _should_use_go_cli() -> bool:
    value = os.environ.get(_GO_CLI_FLAG, "")
    return value.strip().lower() in _TRUE_VALUES


def _bundled_go_cli_path() -> pathlib.Path:
    binary_name = "langgraph.exe" if os.name == "nt" else "langgraph"
    return pathlib.Path(__file__).resolve().parent / "bin" / binary_name


def _resolve_go_cli_path() -> pathlib.Path | None:
    override = os.environ.get(_GO_CLI_PATH_ENV)
    if override:
        path = pathlib.Path(override).expanduser()
        return path.resolve()

    bundled = _bundled_go_cli_path()
    if bundled.is_file():
        return bundled

    return None


def _exec_go_cli(argv: Sequence[str]) -> None:
    path = _resolve_go_cli_path()
    if path is None:
        raise click.ClickException(
            "Go CLI requested via LANGGRAPH_USE_GO_CLI, but no langgraph binary was "
            "found. Set LANGGRAPH_GO_CLI_PATH or install a wheel that bundles the "
            "binary."
        )
    if not path.is_file():
        raise click.ClickException(
            f"LANGGRAPH_GO_CLI_PATH points to a missing file: {path}"
        )

    env = os.environ.copy()
    env.setdefault(_CALLING_PYTHON_ENV, sys.executable)
    os.execvpe(str(path), [str(path), *argv], env)


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if _should_use_go_cli():
            _exec_go_cli(args)
        _legacy_cli(args)
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc
