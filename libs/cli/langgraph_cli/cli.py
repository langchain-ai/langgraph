import json
import pathlib
import sys
from typing import Callable, Optional

import click
import click.exceptions

import langgraph_cli.config
import langgraph_cli.docker
from langgraph_cli.analytics import log_command
from langgraph_cli.config import Config
from langgraph_cli.constants import DEFAULT_CONFIG, DEFAULT_PORT
from langgraph_cli.docker import DockerCapabilities
from langgraph_cli.exec import Runner, subp_exec
from langgraph_cli.progress import Progress

OPT_DOCKER_COMPOSE = click.option(
    "--docker-compose",
    "-d",
    help="Advanced: Path to docker-compose.yml file with additional services to launch.",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
)
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
OPT_RECREATE = click.option(
    "--recreate/--no-recreate",
    default=False,
    show_default=True,
    help="Recreate containers even if their configuration and image haven't changed",
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
OPT_WATCH = click.option("--watch", is_flag=True, help="Restart on file changes")
OPT_DEBUGGER_PORT = click.option(
    "--debugger-port",
    type=int,
    help="Pull the debugger image locally and serve the UI on specified port",
)
OPT_POSTGRES_URI = click.option(
    "--postgres-uri",
    help="Postgres URI to use for the database. Defaults to launching a local database",
)


@click.group()
def cli():
    pass


@OPT_RECREATE
@OPT_PULL
@OPT_PORT
@OPT_DOCKER_COMPOSE
@OPT_CONFIG
@OPT_VERBOSE
@OPT_DEBUGGER_PORT
@OPT_WATCH
@OPT_POSTGRES_URI
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for services to start before returning. Implies --detach",
)
@cli.command(
    help="Start langgraph API server. For local testing, requires a LangSmith API key with access to LangGraph Cloud closed beta. Requires a license key for production use."
)
@log_command
def up(
    config: pathlib.Path,
    docker_compose: Optional[pathlib.Path],
    port: int,
    recreate: bool,
    pull: bool,
    watch: bool,
    wait: bool,
    verbose: bool,
    debugger_port: Optional[int],
    postgres_uri: Optional[str],
):
    click.secho("Starting LangGraph API server...", fg="green")
    click.secho(
        """For local dev, requires env var LANGSMITH_API_KEY with access to LangGraph Cloud closed beta.
For production use, requires a license key in env var LANGGRAPH_CLOUD_LICENSE_KEY.""",
        fg="red",
    )
    with Runner() as runner, Progress(message="Pulling...") as set:
        capabilities = langgraph_cli.docker.check_capabilities(runner)
        args, stdin = prepare(
            runner,
            capabilities=capabilities,
            config_path=config,
            docker_compose=docker_compose,
            port=port,
            pull=pull,
            watch=watch,
            verbose=verbose,
            debugger_port=debugger_port,
            postgres_uri=postgres_uri,
        )
        # add up + options
        args.extend(["up", "--remove-orphans"])
        if recreate:
            args.extend(["--force-recreate", "--renew-anon-volumes"])
            try:
                runner.run(subp_exec("docker", "volume", "rm", "langgraph-data"))
            except click.exceptions.Exit:
                pass
        if watch:
            args.append("--watch")
        if wait:
            args.append("--wait")
        else:
            args.append("--abort-on-container-exit")
        # run docker compose
        set("Building...")

        def on_stdout(line: str):
            if "unpacking to docker.io" in line:
                set("Starting...")
            elif "GET /ok" in line:
                debugger_origin = (
                    f"http://localhost:{debugger_port}"
                    if debugger_port
                    else "https://smith.langchain.com"
                )
                set("")
                sys.stdout.write(
                    f"""Ready!
- API: http://localhost:{port}
- Docs: http://localhost:{port}/docs
- Debugger: {debugger_origin}/studio/?baseUrl=http://127.0.0.1:{port}
"""
                )
                sys.stdout.flush()
                return True

        if capabilities.compose_type == "plugin":
            compose_cmd = ["docker", "compose"]
        elif capabilities.compose_type == "standalone":
            compose_cmd = ["docker-compose"]

        runner.run(
            subp_exec(
                *compose_cmd,
                *args,
                input=stdin,
                verbose=verbose,
                on_stdout=on_stdout,
            )
        )


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

        def on_stdout(line: str):
            if "GET /ok" in line:
                set("")
                sys.stdout.write(
                    f"""Ready!
- API: http://localhost:{port}
"""
                )
                sys.stdout.flush()
                return True

        runner.run(
            subp_exec(
                "docker",
                *args,
                tag,
                verbose=verbose,
                on_stdout=on_stdout,
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


def prepare_args_and_stdin(
    *,
    capabilities: DockerCapabilities,
    config_path: pathlib.Path,
    config: Config,
    docker_compose: Optional[pathlib.Path],
    port: int,
    watch: bool,
    debugger_port: Optional[int] = None,
    postgres_uri: Optional[str] = None,
):
    # prepare args
    stdin = langgraph_cli.docker.compose(
        capabilities,
        port=port,
        debugger_port=debugger_port,
        postgres_uri=postgres_uri,
    )
    args = [
        "--project-directory",
        str(config_path.parent),
    ]
    # apply options
    if docker_compose:
        args.extend(["-f", str(docker_compose)])
    args.extend(["-f", "-"])  # stdin
    # apply config
    stdin += langgraph_cli.config.config_to_compose(
        config_path,
        config,
        watch=watch,
        base_image="langchain/langgraph-api",
    )
    return args, stdin


def prepare(
    runner,
    *,
    capabilities: DockerCapabilities,
    config_path: pathlib.Path,
    docker_compose: Optional[pathlib.Path],
    port: int,
    pull: bool,
    watch: bool,
    verbose: bool,
    debugger_port: Optional[int] = None,
    postgres_uri: Optional[str] = None,
):
    with open(config_path) as f:
        config = langgraph_cli.config.validate_config(json.load(f))
    # pull latest images
    if pull:
        runner.run(
            subp_exec(
                "docker",
                "pull",
                f"langchain/langgraph-api:{config['python_version']}",
                verbose=verbose,
            )
        )

    args, stdin = prepare_args_and_stdin(
        capabilities=capabilities,
        config_path=config_path,
        config=config,
        docker_compose=docker_compose,
        port=port,
        watch=watch,
        debugger_port=debugger_port,
        postgres_uri=postgres_uri,
    )
    return args, stdin
