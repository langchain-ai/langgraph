import json
import pathlib
from typing import Callable, Optional

import click
import click.exceptions

import langgraph_cli.config
import langgraph_cli.docker
from langgraph_cli.analytics import log_command
from langgraph_cli.constants import DEFAULT_CONFIG, DEFAULT_PORT
from langgraph_cli.exec import Runner, subp_exec
from langgraph_cli.progress import Progress

OPT_CONFIG = click.option(
    "--config",
    "-c",
    help="""Path to configuration file declaring dependencies, graphs and environment variables.

    \b
    Config file must be a JSON file that has the following keys:
    - "dependencies": array of dependencies for langgraph API server. Dependencies can be one of the following:
      - ".", which would look for local python packages, as well as pyproject.toml, setup.py or requirements.txt in the app directory
      - "./local_package"
      - "<package_name>
    - "graphs": mapping from graph ID to path where the compiled graph is defined, i.e. ./your_package/your_file.py:variable, where
        "variable" is an instance of langgraph.graph.graph.CompiledGraph
    - "env": (optional) path to .env file or a mapping from environment variable to its value
    - "python_version": (optional) 3.11 or 3.12. Defaults to 3.11
    - "pip_config_file": (optional) path to pip config file
    - "dockerfile_lines": (optional) array of additional lines to add to Dockerfile following the import from parent image

    \b
    Example:
        langgraph up -c langgraph.json

    \b
    Example:
    {
        "dependencies": [
            "langchain_openai",
            "./your_package"
        ],
        "graphs": {
            "my_graph_id": "./your_package/your_file.py:variable"
        },
        "env": "./.env"
    }

    \b
    Example:
    {
        "python_version": "3.11",
        "dependencies": [
            "langchain_openai",
            "."
        ],
        "graphs": {
            "my_graph_id": "./your_package/your_file.py:variable"
        },
        "env": {
            "OPENAI_API_KEY": "secret-key"
        }
    }

    Defaults to looking for langgraph.json in the current directory.""",
    default=DEFAULT_CONFIG,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
)
OPT_PORT = click.option(
    "--port",
    "-p",
    type=int,
    default=DEFAULT_PORT,
    show_default=True,
    help="""
    Port to expose.

    \b
    Example:
        langgraph up --port 8000
    \b
    """,
)
OPT_PULL = click.option(
    "--pull/--no-pull",
    default=True,
    show_default=True,
    help="""
    Pull latest images. Use --no-pull for running the server with locally-built images.

    \b
    Example:
        langgraph up --no-pull
    \b
    """,
)
OPT_VERBOSE = click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show more output from the server logs",
)


@click.group()
def cli():
    pass


@OPT_PULL
@OPT_PORT
@OPT_CONFIG
@OPT_VERBOSE
@cli.command(
    help="Start langgraph test server. This command enables you to confirm your graph will work inside the langgraph API server, before using LangGraph Cloud."
)
@log_command
def test(
    config: pathlib.Path,
    port: int,
    pull: bool,
    # stop_when_ready: bool,
    verbose: bool,
):
    with Runner() as runner, Progress(message="Pulling...") as set:
        # check docker available
        capabilities = langgraph_cli.docker.check_capabilities(runner)
        # open config
        with open(config) as f:
            config_json = langgraph_cli.config.validate_config(json.load(f))
        # build
        base_image = "langchain/langgraph-trial"
        tag = f"langgraph-test-{config.parent.name}"
        _build(
            runner,
            set,
            config,
            config_json,
            None,
            base_image,
            pull,
            tag,
        )
        # run
        set("Running...")
        args = [
            "run",
            "--rm",
            "-p",
            f"{port}:8000",
        ]
        if isinstance(config_json["env"], str):
            args.extend(
                [
                    "--env-file",
                    str(config.parent / config_json["env"]),
                ]
            )
        else:
            for k, v in config_json["env"].items():
                args.extend(
                    [
                        "-e",
                        f"{k}={v}",
                    ]
                )
        if capabilities.healthcheck_start_interval:
            args.extend(
                [
                    "--health-interval",
                    "5s",
                    "--health-retries",
                    "1",
                    "--health-start-period",
                    "10s",
                    "--health-start-interval",
                    "1s",
                ]
            )
        else:
            args.extend(
                [
                    "--health-interval",
                    "5s",
                    "--health-retries",
                    "2",
                ]
            )
        runner.run(
            subp_exec(
                "docker",
                *args,
                tag,
                verbose=verbose,
            )
        )


def _build(
    runner,
    set: Callable[[str], None],
    config: pathlib.Path,
    config_json: dict,
    platform: Optional[str],
    base_image: Optional[str],
    pull: bool,
    tag: str,
):
    base_image = base_image or "langchain/langgraph-api"

    # pull latest images
    if pull:
        runner.run(
            subp_exec(
                "docker",
                "pull",
                f"{base_image}:{config_json['python_version']}",
            )
        )
    set("Building...")
    # apply options
    args = [
        "-f",
        "-",  # stdin
        "-t",
        tag,
    ]
    if platform:
        args.extend(["--platform", platform])
    # apply config
    stdin = langgraph_cli.config.config_to_docker(config, config_json, base_image)
    # run docker build
    runner.run(
        subp_exec(
            "docker", "build", *args, str(config.parent), input=stdin, verbose=True
        )
    )


@OPT_CONFIG
@OPT_PULL
@click.option(
    "--tag",
    "-t",
    help="""Tag for the docker image.

    \b
    Example:
        langgraph build -t my-image

    \b
    """,
    required=True,
)
@click.option(
    "--platform",
    help="""Target platform(s) to build the docker image for.

    \b
    Example:
        langgraph build --platform linux/amd64,linux/arm64
    \b
    """,
)
@click.option(
    "--base-image",
    hidden=True,
)
@cli.command(help="Build langgraph API server docker image")
@log_command
def build(
    config: pathlib.Path,
    platform: Optional[str],
    base_image: Optional[str],
    pull: bool,
    tag: str,
):
    with Runner() as runner, Progress(message="Pulling...") as set:
        # check docker available
        langgraph_cli.docker.check_capabilities(runner)
        # open config
        with open(config) as f:
            config_json = langgraph_cli.config.validate_config(json.load(f))
        # build
        _build(runner, set, config, config_json, platform, base_image, pull, tag)
