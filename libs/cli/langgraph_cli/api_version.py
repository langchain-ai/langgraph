"""Resolve the LangGraph API version from CLI flags and langgraph.json."""

import json
import pathlib
import re
import urllib.request

import click

VERSION_MARKER_REPO = "langchain/langgraph-published-version-marker"
_PATCH_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+")


def resolve_langgraph_api_version(
    config_path: pathlib.Path,
    api_version_cli_param: str | None,
) -> str:
    """Resolve the API version from the CLI flag and/or langgraph.json.

    Returns the resolved patch-level version string. When neither source
    provides a version, the latest published version is fetched from Docker Hub.

    Raises `click.ClickException` when both sources specify a version, or
    when a version cannot be resolved via Docker Hub.
    """
    api_version_langgraph_json = _read_api_version_from_config(config_path)

    if api_version_cli_param and api_version_langgraph_json:
        raise click.ClickException(
            "API version specified in both --api-version CLI flag "
            f"({api_version_cli_param!r}) and langgraph.json "
            f"({api_version_langgraph_json!r}). Please use only one."
        )

    preferred_api_version = api_version_cli_param or api_version_langgraph_json

    if preferred_api_version and _PATCH_VERSION_RE.match(preferred_api_version):
        return preferred_api_version

    version_prefix = preferred_api_version or ""
    if version_prefix:
        click.secho(
            f"Resolving API version matching {version_prefix!r} from Docker Hub...",
            fg="cyan",
        )
    else:
        click.secho(
            "Resolving latest API version from Docker Hub...",
            fg="cyan",
        )
    resolved = _fetch_matching_version(version_prefix)
    click.secho(f"Resolved API version: {resolved}", fg="cyan")
    return resolved


def _read_api_version_from_config(config_path: pathlib.Path) -> str | None:
    """Read the `api_version` field from langgraph.json (if present)."""
    try:
        with open(config_path) as f:
            raw_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return raw_config.get("api_version")


def _fetch_matching_version(version_prefix: str = "") -> str:
    """Query Docker Hub for the latest patch version matching *version_prefix*.

    When *version_prefix* is empty, returns the latest published version.
    """
    url = f"https://hub.docker.com/v2/repositories/{VERSION_MARKER_REPO}/tags/?page_size=10"
    if version_prefix:
        url += f"&name={version_prefix}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        raise click.ClickException(
            f"Failed to fetch API version from {VERSION_MARKER_REPO}: {exc}\n"
            "You can specify an exact version with --api-version (e.g. 0.7.67)."
        ) from exc

    for tag in data.get("results", []):
        name = tag.get("name", "")
        if _SEMVER_RE.match(name):
            return name

    if version_prefix:
        msg = f"Could not find a version matching {version_prefix!r} in {VERSION_MARKER_REPO}."
    else:
        msg = f"Could not find a published version in {VERSION_MARKER_REPO}."
    raise click.ClickException(
        f"{msg}\nYou can specify an exact version with --api-version (e.g. 0.7.67)."
    )
