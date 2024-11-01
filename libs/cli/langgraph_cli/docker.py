import json
import pathlib
import shutil
from typing import Literal, NamedTuple, Optional

import click.exceptions

from langgraph_cli.exec import subp_exec

ROOT = pathlib.Path(__file__).parent.resolve()
DEFAULT_POSTGRES_URI = (
    "postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable"
)

REDIS = """
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
"""

DB = """
    langgraph-postgres:
        image: postgres:16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
"""


class Version(NamedTuple):
    major: int
    minor: int
    patch: int


DockerComposeType = Literal["plugin", "standalone"]


class DockerCapabilities(NamedTuple):
    version_docker: Version
    version_compose: Version
    healthcheck_start_interval: bool
    compose_type: DockerComposeType = "plugin"


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

    compose_type: DockerComposeType
    try:
        compose = next(
            p for p in info["ClientInfo"]["Plugins"] if p["Name"] == "compose"
        )
        compose_version_str = compose["Version"]
        compose_type = "plugin"
    except (KeyError, StopIteration):
        if shutil.which("docker-compose") is None:
            raise click.UsageError("Docker Compose not installed") from None

        compose_version_str, _ = runner.run(
            subp_exec("docker-compose", "--version", "--short", collect=True)
        )
        compose_type = "standalone"

    # parse versions
    docker_version = _parse_version(info["ServerVersion"])
    compose_version = _parse_version(compose_version_str)

    # check capabilities
    return DockerCapabilities(
        version_docker=docker_version,
        version_compose=compose_version,
        healthcheck_start_interval=docker_version >= Version(25, 0, 0),
        compose_type=compose_type,
    )


def debugger_compose(
    *, port: Optional[int] = None, base_url: Optional[str] = None
) -> str:
    if port is None:
        return ""

    compose_str = """
    langgraph-debugger:
        image: langchain/langgraph-debugger
        restart: on-failure
        depends_on:
            langgraph-postgres:
                condition: service_healthy
        ports:
            - "{port}:3968"
"""

    if base_url:
        compose_str += """
        environment:
            VITE_STUDIO_LOCAL_GRAPH_URL: {base_url}
"""

    return compose_str.format(port=port, base_url=base_url)


def compose(
    capabilities: DockerCapabilities,
    *,
    port: int,
    debugger_port: Optional[int] = None,
    debugger_base_url: Optional[str] = None,
    # postgres://user:password@host:port/database?option=value
    postgres_uri: Optional[str] = None,
) -> str:
    if postgres_uri is None:
        include_db = True
        postgres_uri = DEFAULT_POSTGRES_URI
    else:
        include_db = False

    db = DB.format() if include_db else ""
    volumes = (
        """volumes:
    langgraph-data:
        driver: local
"""
        if include_db
        else ""
    )
    if db:
        if capabilities.healthcheck_start_interval:
            db += """
            interval: 60s
            start_interval: 1s"""
        else:
            db += """
            interval: 5s"""

    compose_str = f"""{volumes}services:
{REDIS}
{db}
{debugger_compose(port=debugger_port, base_url=debugger_base_url)}
    langgraph-api:
        ports:
            - "{port}:8000\"
        depends_on:
            langgraph-redis:
                condition: service_healthy"""
    if include_db:
        compose_str += """
            langgraph-postgres:
                condition: service_healthy"""
    compose_str += f"""
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: {postgres_uri}
"""
    if capabilities.healthcheck_start_interval:
        compose_str += """        healthcheck:
            test: python /api/healthcheck.py
            interval: 60s
            start_interval: 1s
            start_period: 10s"""

    return compose_str
