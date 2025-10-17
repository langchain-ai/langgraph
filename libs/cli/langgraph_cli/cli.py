"""CLI entrypoint for LangGraph API server."""

import os
import pathlib
import shutil
import sys
from collections.abc import Callable, Sequence

import click
import click.exceptions
from click import secho

import langgraph_cli.config
import langgraph_cli.docker
from langgraph_cli.analytics import log_command
from langgraph_cli.config import Config
from langgraph_cli.constants import DEFAULT_CONFIG, DEFAULT_PORT
from langgraph_cli.docker import DockerCapabilities
from langgraph_cli.exec import Runner, subp_exec
from langgraph_cli.progress import Progress
from langgraph_cli.templates import TEMPLATE_HELP_STRING, create_new
from langgraph_cli.util import warn_non_wolfi_distro
from langgraph_cli.version import __version__

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
    - "python_version": (optional) 3.11, 3.12, or 3.13. Defaults to 3.11
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
OPT_DEBUGGER_BASE_URL = click.option(
    "--debugger-base-url",
    type=str,
    help="URL used by the debugger to access LangGraph API. Defaults to http://127.0.0.1:[PORT]",
)

OPT_POSTGRES_URI = click.option(
    "--postgres-uri",
    help="Postgres URI to use for the database. Defaults to launching a local database",
)

OPT_API_VERSION = click.option(
    "--api-version",
    type=str,
    help="API server version to use for the base image. If unspecified, the latest version will be used.",
)


@click.group()
@click.version_option(version=__version__, prog_name="LangGraph CLI")
def cli():
    pass


@OPT_RECREATE
@OPT_PULL
@OPT_PORT
@OPT_DOCKER_COMPOSE
@OPT_CONFIG
@OPT_VERBOSE
@OPT_DEBUGGER_PORT
@OPT_DEBUGGER_BASE_URL
@OPT_WATCH
@OPT_POSTGRES_URI
@OPT_API_VERSION
@click.option(
    "--image",
    type=str,
    default=None,
    help="Docker image to use for the langgraph-api service. If specified, skips building and uses this image directly."
    " Useful if you want to test against an image already built using `langgraph build`.",
)
@click.option(
    "--base-image",
    default=None,
    help="Base image to use for the LangGraph API server. Pin to specific versions using version tags. Defaults to langchain/langgraph-api or langchain/langgraphjs-api."
    "\n\n    \b\nExamples:\n    --base-image langchain/langgraph-server:0.2.18  # Pin to a specific patch version"
    "\n    --base-image langchain/langgraph-server:0.2  # Pin to a minor version (Python)",
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for services to start before returning. Implies --detach",
)
@cli.command(help="🚀 Launch LangGraph API server.")
@log_command
def up(
    config: pathlib.Path,
    docker_compose: pathlib.Path | None,
    port: int,
    recreate: bool,
    pull: bool,
    watch: bool,
    wait: bool,
    verbose: bool,
    debugger_port: int | None,
    debugger_base_url: str | None,
    postgres_uri: str | None,
    api_version: str | None,
    image: str | None,
    base_image: str | None,
):
    click.secho("Starting LangGraph API server...", fg="green")
    click.secho(
        """For local dev, requires env var LANGSMITH_API_KEY with access to LangSmith Deployment.
For production use, requires a license key in env var LANGGRAPH_CLOUD_LICENSE_KEY.""",
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
            debugger_base_url=debugger_base_url,
            postgres_uri=postgres_uri,
            api_version=api_version,
            image=image,
            base_image=base_image,
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
            elif "Application startup complete" in line:
                debugger_origin = (
                    f"http://localhost:{debugger_port}"
                    if debugger_port
                    else "https://smith.langchain.com"
                )
                debugger_base_url_query = (
                    debugger_base_url or f"http://127.0.0.1:{port}"
                )
                set("")
                sys.stdout.write(
                    f"""Ready!
- API: http://localhost:{port}
- Docs: http://localhost:{port}/docs
- LangGraph Studio: {debugger_origin}/studio/?baseUrl={debugger_base_url_query}
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


def _build(
    runner,
    set: Callable[[str], None],
    config: pathlib.Path,
    config_json: dict,
    base_image: str | None,
    api_version: str | None,
    pull: bool,
    tag: str,
    passthrough: Sequence[str] = (),
    install_command: str | None = None,
    build_command: str | None = None,
):
    # pull latest images
    if pull:
        runner.run(
            subp_exec(
                "docker",
                "pull",
                langgraph_cli.config.docker_tag(config_json, base_image, api_version),
                verbose=True,
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
    # determine build context: use current directory for JS projects, config parent for Python
    is_js_project = config_json.get("node_version") and not config_json.get(
        "python_version"
    )
    # build/install commands only apply to JS projects for now
    # without install/build command, JS projects will follow the old behavior
    if is_js_project and (build_command or install_command):
        build_context = str(pathlib.Path.cwd())
    else:
        build_context = str(config.parent)

    # apply config
    stdin, additional_contexts = langgraph_cli.config.config_to_docker(
        config,
        config_json,
        base_image,
        api_version,
        install_command,
        build_command,
        build_context,
    )
    # add additional_contexts
    if additional_contexts:
        for k, v in additional_contexts.items():
            args.extend(["--build-context", f"{k}={v}"])
    runner.run(
        subp_exec(
            "docker",
            "build",
            *args,
            *passthrough,
            build_context,
            input=stdin,
            verbose=True,
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
    "--base-image",
    help="Base image to use for the LangGraph API server. Pin to specific versions using version tags. Defaults to langchain/langgraph-api or langchain/langgraphjs-api."
    "\n\n    \b\nExamples:\n    --base-image langchain/langgraph-server:0.2.18  # Pin to a specific patch version"
    "\n    --base-image langchain/langgraph-server:0.2  # Pin to a minor version (Python)",
)
@OPT_API_VERSION
@click.option(
    "--install-command",
    help="Custom install command to run from the build context root. If not provided, auto-detects based on package manager files.",
)
@click.option(
    "--build-command",
    help="Custom build command to run from the langgraph.json directory. If not provided, uses default build process.",
)
@click.argument("docker_build_args", nargs=-1, type=click.UNPROCESSED)
@cli.command(
    help="📦 Build LangGraph API server Docker image.",
    context_settings=dict(
        ignore_unknown_options=True,
    ),
)
@log_command
def build(
    config: pathlib.Path,
    docker_build_args: Sequence[str],
    base_image: str | None,
    api_version: str | None,
    pull: bool,
    tag: str,
    install_command: str | None,
    build_command: str | None,
):
    with Runner() as runner, Progress(message="Pulling...") as set:
        if shutil.which("docker") is None:
            raise click.UsageError("Docker not installed") from None
        config_json = langgraph_cli.config.validate_config_file(config)
        warn_non_wolfi_distro(config_json)
        _build(
            runner,
            set,
            config,
            config_json,
            base_image,
            api_version,
            pull,
            tag,
            docker_build_args,
            install_command,
            build_command,
        )


def _get_docker_ignore_content() -> str:
    """Return the content of a .dockerignore file.

    This file is used to exclude files and directories from the Docker build context.

    It may be overly broad, but it's better to be safe than sorry.

    The main goal is to exclude .env files by default.
    """
    return """\
# Ignore node_modules and other dependency directories
node_modules
bower_components
vendor

# Ignore logs and temporary files
*.log
*.tmp
*.swp

# Ignore .env files and other environment files
.env
.env.*
*.local

# Ignore git-related files
.git
.gitignore

# Ignore Docker-related files and configs
.dockerignore
docker-compose.yml

# Ignore build and cache directories
dist
build
.cache
__pycache__

# Ignore IDE and editor configurations
.vscode
.idea
*.sublime-project
*.sublime-workspace
.DS_Store  # macOS-specific

# Ignore test and coverage files
coverage
*.coverage
*.test.js
*.spec.js
tests
"""


@OPT_CONFIG
@click.argument("save_path", type=click.Path(resolve_path=True))
@cli.command(
    help="🐳 Generate a Dockerfile for the LangGraph API server, with Docker Compose options."
)
@click.option(
    # Add a flag for adding a docker-compose.yml file as part of the output
    "--add-docker-compose",
    help=(
        "Add additional files for running the LangGraph API server with "
        "docker-compose. These files include a docker-compose.yml, .env file, "
        "and a .dockerignore file."
    ),
    is_flag=True,
)
@click.option(
    "--base-image",
    help="Base image to use for the LangGraph API server. Pin to specific versions using version tags. Defaults to langchain/langgraph-api or langchain/langgraphjs-api."
    "\n\n    \b\nExamples:\n    --base-image langchain/langgraph-server:0.2.18  # Pin to a specific patch version"
    "\n    --base-image langchain/langgraph-server:0.2  # Pin to a minor version (Python)",
)
@OPT_API_VERSION
@log_command
def dockerfile(
    save_path: str,
    config: pathlib.Path,
    add_docker_compose: bool,
    base_image: str | None = None,
    api_version: str | None = None,
) -> None:
    save_path = pathlib.Path(save_path).absolute()
    secho(f"🔍 Validating configuration at path: {config}", fg="yellow")
    config_json = langgraph_cli.config.validate_config_file(config)
    warn_non_wolfi_distro(config_json)
    secho("✅ Configuration validated!", fg="green")

    secho(f"📝 Generating Dockerfile at {save_path}", fg="yellow")
    dockerfile, additional_contexts = langgraph_cli.config.config_to_docker(
        config,
        config_json,
        base_image=base_image,
        api_version=api_version,
    )
    with open(str(save_path), "w", encoding="utf-8") as f:
        f.write(dockerfile)
    secho("✅ Created: Dockerfile", fg="green")

    if additional_contexts:
        additional_contexts_str = ",".join(
            f"{k}={v}" for k, v in additional_contexts.items()
        )
        secho(
            f"""📝 Run docker build with these additional build contexts `--build-context {additional_contexts_str}`""",
            fg="yellow",
        )

    if add_docker_compose:
        # Add docker compose and related files
        # Add .dockerignore file in the same directory as the Dockerfile
        with open(str(save_path.parent / ".dockerignore"), "w", encoding="utf-8") as f:
            f.write(_get_docker_ignore_content())
        secho("✅ Created: .dockerignore", fg="green")

        # Generate a docker-compose.yml file
        path = str(save_path.parent / "docker-compose.yml")
        with open(path, "w", encoding="utf-8") as f:
            with Runner() as runner:
                capabilities = langgraph_cli.docker.check_capabilities(runner)

            compose_dict = langgraph_cli.docker.compose_as_dict(
                capabilities,
                port=8123,
                base_image=base_image,
            )
            # Add .env file to the docker-compose.yml for the langgraph-api service
            compose_dict["services"]["langgraph-api"]["env_file"] = [".env"]
            # Add the Dockerfile to the build context
            compose_dict["services"]["langgraph-api"]["build"] = {
                "context": ".",
                "dockerfile": save_path.name,
            }
            # Add the base_image as build arg if provided
            if base_image:
                compose_dict["services"]["langgraph-api"]["build"]["args"] = {
                    "BASE_IMAGE": base_image
                }
            f.write(langgraph_cli.docker.dict_to_yaml(compose_dict))
            secho("✅ Created: docker-compose.yml", fg="green")

        # Check if the .env file exists in the same directory as the Dockerfile
        if not (save_path.parent / ".env").exists():
            # Also add an empty .env file
            with open(str(save_path.parent / ".env"), "w", encoding="utf-8") as f:
                f.writelines(
                    [
                        "# Uncomment the following line to add your LangSmith API key",
                        "\n",
                        "# LANGSMITH_API_KEY=your-api-key",
                        "\n",
                        "# Or if you have a LangSmith Deployment license key, "
                        "then uncomment the following line: ",
                        "\n",
                        "# LANGGRAPH_CLOUD_LICENSE_KEY=your-license-key",
                        "\n",
                        "# Add any other environment variables go below...",
                    ]
                )

            secho("✅ Created: .env", fg="green")
        else:
            # Do nothing since the .env file already exists. Not a great
            # idea to overwrite in case the user has added custom env vars set
            # in the .env file already.
            secho("➖ Skipped: .env. It already exists!", fg="yellow")

    secho(
        f"🎉 Files generated successfully at path {save_path.parent}!",
        fg="cyan",
        bold=True,
    )


@click.option(
    "--host",
    default="127.0.0.1",
    help="Network interface to bind the development server to. Default 127.0.0.1 is recommended for security. Only use 0.0.0.0 in trusted networks",
)
@click.option(
    "--port",
    default=2024,
    type=int,
    help="Port number to bind the development server to. Example: langgraph dev --port 8000",
)
@click.option(
    "--no-reload",
    is_flag=True,
    help="Disable automatic reloading when code changes are detected",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="langgraph.json",
    help="Path to configuration file declaring dependencies, graphs and environment variables",
)
@click.option(
    "--n-jobs-per-worker",
    default=None,
    type=int,
    help="Maximum number of concurrent jobs each worker process can handle. Default: 10",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Skip automatically opening the browser when the server starts",
)
@click.option(
    "--debug-port",
    default=None,
    type=int,
    help="Enable remote debugging by listening on specified port. Requires debugpy to be installed",
)
@click.option(
    "--wait-for-client",
    is_flag=True,
    help="Wait for a debugger client to connect to the debug port before starting the server",
    default=False,
)
@click.option(
    "--studio-url",
    type=str,
    default=None,
    help="URL of the LangGraph Studio instance to connect to. Defaults to https://smith.langchain.com",
)
@click.option(
    "--allow-blocking",
    is_flag=True,
    help="Don't raise errors for synchronous I/O blocking operations in your code.",
    default=False,
)
@click.option(
    "--tunnel",
    is_flag=True,
    help="Expose the local server via a public tunnel (in this case, Cloudflare) "
    "for remote frontend access. This avoids issues with browsers "
    "or networks blocking localhost connections.",
    default=False,
)
@click.option(
    "--server-log-level",
    type=str,
    default="WARNING",
    help="Set the log level for the API server.",
)
@cli.command(
    "dev",
    help="🏃‍♀️‍➡️ Run LangGraph API server in development mode with hot reloading and debugging support",
)
@log_command
def dev(
    host: str,
    port: int,
    no_reload: bool,
    config: str,
    n_jobs_per_worker: int | None,
    no_browser: bool,
    debug_port: int | None,
    wait_for_client: bool,
    studio_url: str | None,
    allow_blocking: bool,
    tunnel: bool,
    server_log_level: str,
):
    """CLI entrypoint for running the LangGraph API server."""
    try:
        from langgraph_api.cli import run_server  # type: ignore
    except ImportError:
        py_version_msg = ""
        if sys.version_info < (3, 11):
            py_version_msg = (
                "\n\nNote: The in-mem server requires Python 3.11 or higher to be installed."
                f" You are currently using Python {sys.version_info.major}.{sys.version_info.minor}."
                ' Please upgrade your Python version before installing "langgraph-cli[inmem]".'
            )
        try:
            from importlib import util

            if not util.find_spec("langgraph_api"):
                raise click.UsageError(
                    "Required package 'langgraph-api' is not installed.\n"
                    "Please install it with:\n\n"
                    '    pip install -U "langgraph-cli[inmem]"'
                    f"{py_version_msg}"
                ) from None
        except ImportError:
            raise click.UsageError(
                "Could not verify package installation. Please ensure Python is up to date and\n"
                "langgraph-cli is installed with the 'inmem' extra: pip install -U \"langgraph-cli[inmem]\""
                f"{py_version_msg}"
            ) from None
        raise click.UsageError(
            "Could not import run_server. This likely means your installation is incomplete.\n"
            "Please ensure langgraph-cli is installed with the 'inmem' extra: pip install -U \"langgraph-cli[inmem]\""
            f"{py_version_msg}"
        ) from None

    config_json = langgraph_cli.config.validate_config_file(pathlib.Path(config))
    if config_json.get("node_version"):
        raise click.UsageError(
            "In-mem server for JS graphs is not supported in this version of the LangGraph CLI. Please use `npx @langchain/langgraph-cli` instead."
        ) from None

    cwd = os.getcwd()
    sys.path.append(cwd)
    dependencies = config_json.get("dependencies", [])
    for dep in dependencies:
        dep_path = pathlib.Path(cwd) / dep
        if dep_path.is_dir() and dep_path.exists():
            sys.path.append(str(dep_path))

    graphs = config_json.get("graphs", {})

    run_server(
        host,
        port,
        not no_reload,
        graphs,
        n_jobs_per_worker=n_jobs_per_worker,
        open_browser=not no_browser,
        debug_port=debug_port,
        env=config_json.get("env"),
        store=config_json.get("store"),
        wait_for_client=wait_for_client,
        auth=config_json.get("auth"),
        http=config_json.get("http"),
        ui=config_json.get("ui"),
        ui_config=config_json.get("ui_config"),
        studio_url=studio_url,
        allow_blocking=allow_blocking,
        tunnel=tunnel,
        server_level=server_log_level,
    )


@click.argument("path", required=False)
@click.option(
    "--template",
    type=str,
    help=TEMPLATE_HELP_STRING,
)
@cli.command("new", help="🌱 Create a new LangGraph project from a template.")
@log_command
def new(path: str | None, template: str | None) -> None:
    """Create a new LangGraph project from a template."""
    return create_new(path, template)


def prepare_args_and_stdin(
    *,
    capabilities: DockerCapabilities,
    config_path: pathlib.Path,
    config: Config,
    docker_compose: pathlib.Path | None,
    port: int,
    watch: bool,
    debugger_port: int | None = None,
    debugger_base_url: str | None = None,
    postgres_uri: str | None = None,
    api_version: str | None = None,
    # Like "my-tag" (if you already built it locally)
    image: str | None = None,
    # Like "langchain/langgraphjs-api" or "langchain/langgraph-api
    base_image: str | None = None,
) -> tuple[list[str], str]:
    assert config_path.exists(), f"Config file not found: {config_path}"
    # prepare args
    stdin = langgraph_cli.docker.compose(
        capabilities,
        port=port,
        debugger_port=debugger_port,
        debugger_base_url=debugger_base_url,
        postgres_uri=postgres_uri,
        image=image,  # Pass image to compose YAML generator
        base_image=base_image,
        api_version=api_version,
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
        base_image=langgraph_cli.config.default_base_image(config),
        api_version=api_version,
        image=image,
    )
    return args, stdin


def prepare(
    runner,
    *,
    capabilities: DockerCapabilities,
    config_path: pathlib.Path,
    docker_compose: pathlib.Path | None,
    port: int,
    pull: bool,
    watch: bool,
    verbose: bool,
    debugger_port: int | None = None,
    debugger_base_url: str | None = None,
    postgres_uri: str | None = None,
    api_version: str | None = None,
    image: str | None = None,
    base_image: str | None = None,
) -> tuple[list[str], str]:
    """Prepare the arguments and stdin for running the LangGraph API server."""
    config_json = langgraph_cli.config.validate_config_file(config_path)
    warn_non_wolfi_distro(config_json)
    # pull latest images
    if pull:
        runner.run(
            subp_exec(
                "docker",
                "pull",
                langgraph_cli.config.docker_tag(config_json, base_image, api_version),
                verbose=verbose,
            )
        )

    args, stdin = prepare_args_and_stdin(
        capabilities=capabilities,
        config_path=config_path,
        config=config_json,
        docker_compose=docker_compose,
        port=port,
        watch=watch,
        debugger_port=debugger_port,
        debugger_base_url=debugger_base_url or f"http://127.0.0.1:{port}",
        postgres_uri=postgres_uri,
        api_version=api_version,
        image=image,
        base_image=base_image,
    )
    return args, stdin
