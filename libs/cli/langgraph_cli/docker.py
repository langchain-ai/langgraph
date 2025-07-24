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
        stdout, _ = runner.run(
            subp_exec("docker", "info", "-f", "{{json .}}", collect=True)
        )
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
) -> dict:
    if port is None:
        return ""

    config = {
        "langgraph-debugger": {
            "image": "langchain/langgraph-debugger",
            "restart": "on-failure",
            "depends_on": {
                "langgraph-postgres": {"condition": "service_healthy"},
            },
            "ports": [f'"{port}:3968"'],
        }
    }

    if base_url:
        config["langgraph-debugger"]["environment"] = {
            "VITE_STUDIO_LOCAL_GRAPH_URL": base_url
        }

    return config


# Function to convert dictionary to YAML
def dict_to_yaml(d: dict, *, indent: int = 0) -> str:
    """Convert a dictionary to a YAML string."""
    yaml_str = ""

    for idx, (key, value) in enumerate(d.items()):
        # Format things in a visually appealing way
        # Use an extra newline for top-level keys only
        if idx >= 1 and indent < 2:
            yaml_str += "\n"
        space = "    " * indent
        if isinstance(value, dict):
            yaml_str += f"{space}{key}:\n" + dict_to_yaml(value, indent=indent + 1)
        elif isinstance(value, list):
            yaml_str += f"{space}{key}:\n"
            for item in value:
                yaml_str += f"{space}    - {item}\n"
        else:
            yaml_str += f"{space}{key}: {value}\n"
    return yaml_str


def compose_as_dict(
    capabilities: DockerCapabilities,
    *,
    port: int,
    debugger_port: Optional[int] = None,
    debugger_base_url: Optional[str] = None,
    # postgres://user:password@host:port/database?option=value
    postgres_uri: Optional[str] = None,
    # If you are running against an already-built image, you can pass it here
    image: Optional[str] = None,
    # Base image to use for the LangGraph API server
    base_image: Optional[str] = None,
    # API version of the base image
    api_version: Optional[str] = None,
) -> dict:
    """Create a docker compose file as a dictionary in YML style."""
    if postgres_uri is None:
        include_db = True
        postgres_uri = DEFAULT_POSTGRES_URI
    else:
        include_db = False

    # The services below are defined in a non-intuitive order to match
    # the existing unit tests for this function.
    # It's fine to re-order just requires updating the unit tests, so it should
    # be done with caution.

    # Define the Redis service first as per the test order
    services = {
        "langgraph-redis": {
            "image": "redis:6",
            "healthcheck": {
                "test": "redis-cli ping",
                "interval": "5s",
                "timeout": "1s",
                "retries": 5,
            },
        }
    }

    # Add Postgres service before langgraph-api if it is needed
    if include_db:
        services["langgraph-postgres"] = {
            "image": "pgvector/pgvector:pg16",
            "ports": ['"5433:5432"'],
            "environment": {
                "POSTGRES_DB": "postgres",
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": "postgres",
            },
            "command": ["postgres", "-c", "shared_preload_libraries=vector"],
            "volumes": ["langgraph-data:/var/lib/postgresql/data"],
            "healthcheck": {
                "test": "pg_isready -U postgres",
                "start_period": "10s",
                "timeout": "1s",
                "retries": 5,
            },
        }
        if capabilities.healthcheck_start_interval:
            services["langgraph-postgres"]["healthcheck"]["interval"] = "60s"
            services["langgraph-postgres"]["healthcheck"]["start_interval"] = "1s"
        else:
            services["langgraph-postgres"]["healthcheck"]["interval"] = "5s"

    # Add optional debugger service if debugger_port is specified
    if debugger_port:
        services["langgraph-debugger"] = debugger_compose(
            port=debugger_port, base_url=debugger_base_url
        )["langgraph-debugger"]

    # Add langgraph-api service
    services["langgraph-api"] = {
        "ports": [f'"{port}:8000"'],
        "depends_on": {
            "langgraph-redis": {"condition": "service_healthy"},
        },
        "environment": {
            "REDIS_URI": "redis://langgraph-redis:6379",
            "POSTGRES_URI": postgres_uri,
        },
    }
    if image:
        services["langgraph-api"]["image"] = image

    # If Postgres is included, add it to the dependencies of langgraph-api
    if include_db:
        services["langgraph-api"]["depends_on"]["langgraph-postgres"] = {
            "condition": "service_healthy"
        }

    # Additional healthcheck for langgraph-api if required
    if capabilities.healthcheck_start_interval:
        services["langgraph-api"]["healthcheck"] = {
            "test": "python /api/healthcheck.py",
            "interval": "60s",
            "start_interval": "1s",
            "start_period": "10s",
        }

    # Final compose dictionary with volumes included if needed
    compose_dict = {}
    if include_db:
        compose_dict["volumes"] = {"langgraph-data": {"driver": "local"}}
    compose_dict["services"] = services

    return compose_dict


def compose(
    capabilities: DockerCapabilities,
    *,
    port: int,
    debugger_port: Optional[int] = None,
    debugger_base_url: Optional[str] = None,
    # postgres://user:password@host:port/database?option=value
    postgres_uri: Optional[str] = None,
    image: Optional[str] = None,
    base_image: Optional[str] = None,
    api_version: Optional[str] = None,
) -> str:
    """Create a docker compose file as a string."""
    compose_content = compose_as_dict(
        capabilities,
        port=port,
        debugger_port=debugger_port,
        debugger_base_url=debugger_base_url,
        postgres_uri=postgres_uri,
        image=image,
        base_image=base_image,
        api_version=api_version,
    )
    compose_str = dict_to_yaml(compose_content)
    return compose_str
