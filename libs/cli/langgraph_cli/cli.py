import json
import pathlib
import shutil
import sys
from typing import Optional

import click
import click.exceptions

import langgraph_cli.config
import langgraph_cli.docker
from langgraph_cli.config import Config
from langgraph_cli.docker import DockerCapabilities
from langgraph_cli.exec import Runner, subp_exec
from langgraph_cli.progress import Progress
from langgraph_cli.util import clean_empty_lines

OPT_DOCKER_COMPOSE = click.option(
    "--docker-compose",
    "-d",
    help="Advanced: Path to docker-compose.yml file with additional services to launch",
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
    default="langgraph.json",
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
    default=8123,
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
OPT_LANGGRAPH_API_PATH = click.option(
    "--langgraph-api-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    hidden=True,
)
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
@OPT_LANGGRAPH_API_PATH
@OPT_POSTGRES_URI
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for services to start before returning. Implies --detach",
)
@cli.command(help="Start langgraph API server")
def up(
    config: pathlib.Path,
    docker_compose: Optional[pathlib.Path],
    port: int,
    recreate: bool,
    pull: bool,
    watch: bool,
    langgraph_api_path: Optional[pathlib.Path],
    wait: bool,
    verbose: bool,
    debugger_port: Optional[int],
    postgres_uri: Optional[str],
):
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
            langgraph_api_path=langgraph_api_path,
            verbose=verbose,
            debugger_port=debugger_port,
            postgres_uri=postgres_uri,
        )
        # add up + options
        args.extend(["up", "--remove-orphans"])
        if recreate:
            args.extend(["--force-recreate", "--renew-anon-volumes"])
            shutil.rmtree(config.parent / ".langgraph-data", ignore_errors=True)
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


@OPT_PORT
@OPT_DOCKER_COMPOSE
@OPT_CONFIG
@OPT_VERBOSE
@OPT_DEBUGGER_PORT
@cli.command(help="Stop langgraph API server")
def down(
    config: pathlib.Path,
    docker_compose: Optional[pathlib.Path],
    port: int,
    verbose: bool,
    debugger_port: Optional[int],
):
    with Runner() as runner:
        capabilities = langgraph_cli.docker.check_capabilities(runner)
        args, stdin = prepare(
            runner,
            capabilities=capabilities,
            config_path=config,
            docker_compose=docker_compose,
            port=port,
            pull=False,
            watch=False,
            langgraph_api_path=None,
            verbose=verbose,
            debugger_port=debugger_port,
        )
        # add down + options
        args.append("down")
        # run docker compose
        if capabilities.compose_type == "plugin":
            compose_cmd = ["docker", "compose"]
        elif capabilities.compose_type == "standalone":
            compose_cmd = ["docker-compose"]

        runner.run(subp_exec(*compose_cmd, *args, input=stdin, verbose=verbose))


@OPT_DOCKER_COMPOSE
@OPT_CONFIG
@click.option("--follow", "-f", is_flag=True, help="Follow logs")
@cli.command(help="Show langgraph API server logs")
def logs(
    config: pathlib.Path,
    docker_compose: Optional[pathlib.Path],
    follow: bool,
):
    with Runner() as runner:
        capabilities = langgraph_cli.docker.check_capabilities(runner)
        args, stdin = prepare(
            runner,
            capabilities=capabilities,
            config_path=config,
            docker_compose=docker_compose,
            port=8123,
            pull=False,
            watch=False,
            verbose=False,
            langgraph_api_path=None,
        )
        # add logs + options
        args.append("logs")
        if follow:
            args.extend(["-f"])
        # run docker compose
        if capabilities.compose_type == "plugin":
            compose_cmd = ["docker", "compose"]
        elif capabilities.compose_type == "standalone":
            compose_cmd = ["docker-compose"]

        runner.run(subp_exec(*compose_cmd, *args, input=stdin, verbose=True))


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
@cli.command(help="Build langgraph API server docker image")
def build(
    config: pathlib.Path,
    platform: Optional[str],
    pull: bool,
    tag: str,
):
    with open(config) as f:
        config_json = langgraph_cli.config.validate_config(json.load(f))
    with Runner() as runner:
        # check docker available
        langgraph_cli.docker.check_capabilities(runner)
        # pull latest images
        if pull:
            runner.run(
                subp_exec(
                    "docker",
                    "pull",
                    f"langchain/langgraph-api:{config_json['python_version']}",
                )
            )
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
        stdin = langgraph_cli.config.config_to_docker(config, config_json)
        # run docker build
        runner.run(
            subp_exec(
                "docker", "build", *args, str(config.parent), input=stdin, verbose=True
            )
        )


@cli.group(help="Export langgraph compose files")
def export():
    pass


@click.option(
    "--output",
    "-o",
    help="Output path to write the docker compose file to",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=True,
)
@OPT_CONFIG
@OPT_PORT
@OPT_WATCH
@OPT_LANGGRAPH_API_PATH
@export.command(name="compose", help="Export docker compose file")
def export_compose(
    output: pathlib.Path,
    config: pathlib.Path,
    port: int,
    watch: bool,
    langgraph_api_path: Optional[pathlib.Path],
):
    with Runner() as runner:
        capabilities = langgraph_cli.docker.check_capabilities(runner)
        _, stdin = prepare(
            runner,
            capabilities=capabilities,
            config_path=config,
            docker_compose=None,
            pull=False,
            watch=watch,
            langgraph_api_path=langgraph_api_path,
            port=port,
            verbose=False,
        )

    with open(output, "w") as f:
        f.write(clean_empty_lines(stdin))


@click.option(
    "--output",
    "-o",
    help="Output path (directory) to write the helm chart to",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=True,
)
@OPT_PORT
@OPT_DOCKER_COMPOSE
@OPT_CONFIG
@export.command(
    name="helm",
    help="Build and export a helm chart to deploy to a Kubernetes cluster",
    hidden=True,
)
def export_helm(
    output: pathlib.Path,
    config: pathlib.Path,
    docker_compose: Optional[pathlib.Path],
    port: int,
):
    with open(config) as f:
        config_json = langgraph_cli.config.validate_config(json.load(f))

    with Runner() as runner:
        # check docker available
        capabilities = langgraph_cli.docker.check_capabilities(runner)
        # prepare args
        stdin = langgraph_cli.docker.compose(capabilities, port=port)
        args = [
            "convert",
            "--chart",
            "-o",
            str(output),
            "-v",
        ]
        # apply options
        if docker_compose:
            args.extend(["-f", str(docker_compose)])

        args.extend(["-f", "-"])  # stdin
        # apply config
        stdin += langgraph_cli.config.config_to_compose(config, config_json)
        # run kompose convert
        runner.run(subp_exec("kompose", *args, input=stdin))


def prepare_args_and_stdin(
    *,
    capabilities: DockerCapabilities,
    config_path: pathlib.Path,
    config: Config,
    docker_compose: Optional[pathlib.Path],
    port: int,
    watch: bool,
    langgraph_api_path: Optional[pathlib.Path],
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
        config_path, config, watch=watch, langgraph_api_path=langgraph_api_path
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
    langgraph_api_path: Optional[pathlib.Path],
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
        langgraph_api_path=langgraph_api_path,
        debugger_port=debugger_port,
        postgres_uri=postgres_uri,
    )
    return args, stdin
