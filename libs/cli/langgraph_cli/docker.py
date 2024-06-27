import json
import pathlib
import shutil
from typing import NamedTuple

import click.exceptions

from langgraph_cli.exec import subp_exec

ROOT = pathlib.Path(__file__).parent.resolve()


class Version(NamedTuple):
    major: int
    minor: int
    patch: int


class DockerCapabilities(NamedTuple):
    version_docker: Version
    healthcheck_start_interval: bool


def _parse_version(version: str) -> Version:
    parts = version.split(".", 2)
    if len(parts) == 1:
        major = parts[0]
        minor = "0"
        patch = "0"
    elif len(parts) == 2:
        major, minor = parts
        patch = "0"
    else:
        major, minor, patch = parts
    return Version(int(major.lstrip("v")), int(minor), int(patch.split("-")[0]))


def check_capabilities(runner) -> DockerCapabilities:
    # check docker available
    if shutil.which("docker") is None:
        raise click.UsageError("Docker not installed") from None

    try:
        stdout, _ = runner.run(subp_exec("docker", "info", "-f", "json", collect=True))
        info = json.loads(stdout)
    except (click.exceptions.Exit, json.JSONDecodeError):
        raise click.UsageError("Docker not installed or not running") from None

    if not info["ServerVersion"]:
        raise click.UsageError("Docker not running") from None

    # parse versions
    docker_version = _parse_version(info["ServerVersion"])

    # check capabilities
    return DockerCapabilities(
        version_docker=docker_version,
        healthcheck_start_interval=docker_version >= Version(25, 0, 0),
    )
