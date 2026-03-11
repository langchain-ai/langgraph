"""Resolve the LangGraph engine runtime mode."""

import json
import pathlib

import click

_DISTRIBUTED_MIN_VERSION = (0, 7, 68)


def resolve_engine_runtime_mode(
    config_path: pathlib.Path,
    api_version: str,
    engine_runtime_mode_cli_param: str | None,
) -> str:
    """Resolve the engine runtime mode.

    *api_version* must already be resolved to a patch-level semver string.

    Returns ``"distributed"`` or ``"combined_queue_worker"``.

    Raises `click.ClickException` when distributed mode is requested but
    not supported (JavaScript project or api_version <= 0.7.67).
    """
    requires_combined = _requires_combined(config_path, api_version)

    if engine_runtime_mode_cli_param == "distributed":
        if requires_combined:
            reasons = _constraint_reasons(config_path, api_version)
            raise click.ClickException(
                f"Distributed runtime is not supported for {' and '.join(reasons)}."
            )
        return "distributed"

    if engine_runtime_mode_cli_param == "combined_queue_worker":
        return "combined_queue_worker"

    # No explicit choice → default to distributed
    return "distributed"


def _requires_combined(config_path: pathlib.Path, api_version: str) -> bool:
    return _is_javascript_project(config_path) or _version_too_old(api_version)


def _version_too_old(api_version: str) -> bool:
    try:
        parts = tuple(int(x) for x in api_version.split("."))
    except (ValueError, AttributeError):
        return True
    return parts < _DISTRIBUTED_MIN_VERSION


def _is_javascript_project(config_path: pathlib.Path) -> bool:
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(cfg.get("node_version")) and not cfg.get("python_version")


def _constraint_reasons(config_path: pathlib.Path, api_version: str) -> list[str]:
    reasons: list[str] = []
    if _is_javascript_project(config_path):
        reasons.append("JavaScript projects")
    if _version_too_old(api_version):
        reasons.append(f"API version {api_version} (<= 0.7.67)")
    return reasons
