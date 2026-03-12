"""CLI entrypoint for LangGraph API server."""

import base64
import copy
import json as json_mod
import os
import pathlib
import platform
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone

import click
import click.exceptions
from click import secho
from dotenv import dotenv_values

import langgraph_cli.config
import langgraph_cli.docker
from langgraph_cli.analytics import log_command
from langgraph_cli.config import Config
from langgraph_cli.constants import DEFAULT_CONFIG, DEFAULT_PORT
from langgraph_cli.docker import DockerCapabilities
from langgraph_cli.exec import Runner, subp_exec
from langgraph_cli.helpers import format_log_entry, level_fg, resolve_deployment_id
from langgraph_cli.host_backend import HostBackendClient, HostBackendError
from langgraph_cli.progress import Progress
from langgraph_cli.templates import TEMPLATE_HELP_STRING, create_new
from langgraph_cli.util import format_deployments_table, warn_non_wolfi_distro
from langgraph_cli.version import __version__

RESERVED_ENV_VARS = frozenset(
    [
        # LANGCHAIN_RESERVED_ENV_VARS from host-backend
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_PROJECT",
        "LANGSMITH_PROJECT",
        "LANGSMITH_LANGGRAPH_GIT_REPO",
        "LANGGRAPH_GIT_REPO_PATH",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_CONTROL_PLANE_API_KEY",
        "POSTGRES_URI",
        "POSTGRES_PASSWORD",
        "DATABASE_URI",
        "LANGSMITH_LANGGRAPH_GIT_REF",
        "LANGSMITH_LANGGRAPH_GIT_REF_SHA",
        "LANGGRAPH_AUTH_TYPE",
        "LANGSMITH_AUTH_ENDPOINT",
        "LANGSMITH_TENANT_ID",
        "LANGSMITH_AUTH_VERIFY_TENANT_ID",
        "LANGSMITH_HOST_PROJECT_ID",
        "LANGSMITH_HOST_PROJECT_NAME",
        "LANGSMITH_HOST_REVISION_ID",
        "LOG_JSON",
        "LOG_DICT_TRACEBACKS",
        "REDIS_URI",
        "LANGCHAIN_CALLBACKS_BACKGROUND",
        "DD_TRACE_PSYCOPG_ENABLED",
        "DD_TRACE_REDIS_ENABLED",
        "LANGSMITH_DEPLOYMENT_NAME",
        "LANGGRAPH_CLOUD_LICENSE_KEY",
        # ALLOWED_SELF_HOSTED_ENV_VARS (rejected for non-self-hosted)
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "POSTGRES_URI_CUSTOM",
        "REDIS_URI_CUSTOM",
        "PATH",
        "PORT",
        "MOUNT_PREFIX",
        "LSD_ENV",
        "LSD_DD_API_KEY",
        "LSD_DD_ENDPOINT",
        "LSD_DEPLOYMENT_TYPE",
    ]
)

_API_KEY_ENV_NAMES = (
    "LANGGRAPH_HOST_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
)

_DEPLOYMENT_NAME_ENV = "LANGSMITH_DEPLOYMENT_NAME"


def _parse_env_from_config(
    config_json: dict, config_path: pathlib.Path
) -> dict[str, str]:
    """Resolve env vars from langgraph.json 'env' field or a .env fallback."""
    env_field = config_json.get("env")
    # validate_config_file will default env to {}
    if isinstance(env_field, dict) and env_field:
        return {str(k): str(v) for k, v in env_field.items()}
    if isinstance(env_field, str):
        env_path = (config_path.parent / env_field).resolve()
        if not env_path.exists():
            click.secho(
                f"Warning: env file '{env_field}' specified in langgraph.json not found.",
                fg="yellow",
            )
            return {}
    else:
        env_path = pathlib.Path.cwd() / ".env"
    return {k: v for k, v in dotenv_values(env_path).items() if v is not None}


def _secrets_from_env(
    env_vars: dict[str, str],
) -> list[dict[str, str]]:
    """Convert env dict to secrets list, filtering reserved vars with warnings."""
    secrets: list[dict[str, str]] = []
    for name, value in env_vars.items():
        if name in RESERVED_ENV_VARS:
            click.secho(f"   Skipping reserved env var: {name}", fg="yellow")
            continue
        if not value:
            continue
        secrets.append({"name": name, "value": value})
    return secrets


_TERMINAL_STATUSES = frozenset(
    [
        "DEPLOYED",
        "CREATE_FAILED",
        "BUILD_FAILED",
        "DEPLOY_FAILED",
        "SKIPPED",
    ]
)


@contextmanager
def _docker_config_for_token(registry_host: str, token: str):
    """Create a temporary Docker config with only the push token.

    Yields the path to a temporary config directory that can be passed
    to ``docker --config <path>`` so that system credential helpers
    (e.g. gcloud) don't interfere with the push token.
    """
    auth_b64 = base64.b64encode(f"oauth2accesstoken:{token}".encode()).decode()
    config_data = {"auths": {registry_host: {"auth": auth_b64}}}
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json_mod.dump(config_data, f)
        yield tmpdir


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

OPT_HOST_API_KEY = click.option(
    "--api-key",
    envvar="LANGGRAPH_HOST_API_KEY",
    help=(
        "API key. Can also be set via LANGGRAPH_HOST_API_KEY, "
        "LANGSMITH_API_KEY, or LANGCHAIN_API_KEY environment variable or .env file."
    ),
)


OPT_HOST_DEPLOYMENT_NAME = click.option(
    "--name",
    envvar=_DEPLOYMENT_NAME_ENV,
    help=(
        "Deployment name. Can also be set via LANGSMITH_DEPLOYMENT_NAME "
        "environment variable or .env file. Defaults to current directory name "
        "if --deployment-id is not provided."
    ),
)

OPT_HOST_URL = click.option(
    "--host-url",
    envvar="LANGGRAPH_HOST_URL",
    default="https://api.host.langchain.com",
    hidden=True,
)

OPT_ENGINE_RUNTIME_MODE = click.option(
    "--engine-runtime-mode",
    type=click.Choice(["combined_queue_worker", "distributed"]),
    default="combined_queue_worker",
    help="Runtime mode. 'distributed' uses separate executor and orchestrator containers.",
)


class NestedHelpGroup(click.Group):
    """Click group that shows one level of nested subcommands in top-level help."""

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        command_entries: list[tuple[str, click.Command]] = []
        # Collect the top-level commands first, then append one level of nested
        # subcommands using names like "deploy list" so they show up in the
        # top-level help output.
        for command_name in self.list_commands(ctx):
            command = self.get_command(ctx, command_name)
            if command is None or command.hidden:
                continue
            command_entries.append((command_name, command))
            if isinstance(command, click.Group):
                # Build a child context so Click resolves the subcommands the same
                # way it would for the nested group itself.
                sub_ctx = click.Context(command, info_name=command_name, parent=ctx)
                for subcommand_name in command.list_commands(sub_ctx):
                    subcommand = command.get_command(sub_ctx, subcommand_name)
                    if subcommand is None or subcommand.hidden:
                        continue
                    command_entries.append(
                        (f"{command_name} {subcommand_name}", subcommand)
                    )

        # Compute the available width for help text up front so we can truncate
        # descriptions before handing them to Click. That keeps each command on
        # a single line instead of allowing wrapped descriptions.
        command_width = max((len(name) for name, _ in command_entries), default=0)
        help_width = max(formatter.width - command_width - 6, 10)
        rows = [
            (name, command.get_short_help_str(help_width))
            for name, command in command_entries
        ]

        if rows:
            # Render the flattened command list using Click's standard
            # definition-list formatter so alignment stays consistent with the
            # rest of the CLI help output.
            with formatter.section("Commands"):
                formatter.write_dl(rows)


class DeployGroup(NestedHelpGroup):
    """Group that treats leading '-' args as passthrough docker flags."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        result = super().parse_args(ctx, args)
        if ctx._protected_args and ctx._protected_args[0].startswith("-"):
            # Click stores the would-be subcommand in _protected_args; if it looks
            # like an option (e.g. --build-arg) treat it as passthrough docker
            # args instead of insisting on a nested command.
            ctx.args = [*ctx._protected_args, *ctx.args]
            ctx._protected_args = []
            return ctx.args
        return result


@click.group(cls=NestedHelpGroup)
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
@OPT_ENGINE_RUNTIME_MODE
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
    engine_runtime_mode: str,
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
            engine_runtime_mode=engine_runtime_mode,
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
    docker_command: Sequence[str] | None = None,
    extra_flags: Sequence[str] = (),
    verbose: bool = True,
):
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

    # Deep copy to avoid mutating the caller's config (config_to_docker
    # rewrites graph paths to container-internal paths in place).
    config_json = copy.deepcopy(config_json)
    stdin, additional_contexts = langgraph_cli.config.config_to_docker(
        config_path=config,
        config=config_json,
        base_image=base_image,
        api_version=api_version,
        install_command=install_command,
        build_command=build_command,
        build_context=build_context,
    )
    # add additional_contexts
    if additional_contexts:
        for k, v in additional_contexts.items():
            args.extend(["--build-context", f"{k}={v}"])
    cmd = tuple(docker_command) if docker_command else ("docker", "build")
    runner.run(
        subp_exec(
            *cmd,
            *args,
            *extra_flags,
            *passthrough,
            build_context,
            input=stdin,
            verbose=verbose,
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
@OPT_ENGINE_RUNTIME_MODE
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
    engine_runtime_mode: str,
    pull: bool,
    tag: str,
    install_command: str | None,
    build_command: str | None,
):
    if install_command and langgraph_cli.config.has_disallowed_build_command_content(
        install_command
    ):
        raise click.UsageError(
            "install_command contains disallowed characters or patterns."
        )
    if build_command and langgraph_cli.config.has_disallowed_build_command_content(
        build_command
    ):
        raise click.UsageError(
            "build_command contains disallowed characters or patterns."
        )
    with Runner() as runner, Progress(message="Pulling...") as set:
        if shutil.which("docker") is None:
            raise click.UsageError("Docker not installed") from None
        config_json = langgraph_cli.config.validate_config_file(config)
        warn_non_wolfi_distro(config_json)
        effective_base_image = base_image
        if engine_runtime_mode == "distributed" and not base_image:
            effective_base_image = langgraph_cli.config.default_base_image(
                config_json, engine_runtime_mode=engine_runtime_mode
            )
        _build(
            runner,
            set,
            config,
            config_json,
            effective_base_image,
            api_version,
            pull,
            tag,
            docker_build_args,
            install_command,
            build_command,
        )


def _deploy_base_options(
    func: Callable | None = None,
    *,
    include_docker_args: bool = True,
    validate_config_path: bool = True,
):
    """Apply shared deploy flags.

    The group shares most options but should not consume subcommands, so the
    docker build args are only attached when requested.
    """

    def _apply(target: Callable) -> Callable:
        decorators = [
            OPT_HOST_API_KEY,
            OPT_HOST_DEPLOYMENT_NAME,
            click.option(
                "--deployment-id",
                help=(
                    "ID of an existing deployment to update. If omitted, "
                    "--name is used to find or create the deployment."
                ),
            ),
            click.option(
                "--deployment-type",
                type=click.Choice(["dev", "prod"]),
                default="dev",
                show_default=True,
                help="Deployment type (used when creating a new deployment).",
            ),
            click.option(
                "--no-wait",
                is_flag=True,
                default=False,
                help="Skip waiting for deployment status.",
            ),
            OPT_VERBOSE,
            OPT_HOST_URL,
            click.option("--image-name", hidden=True),
            click.option(
                "--tag",
                "-t",
                default="latest",
                show_default=True,
                help="Tag to use for the pushed deployment image.",
            ),
            click.option(
                "--config",
                "-c",
                default=DEFAULT_CONFIG,
                hidden=True,
                type=click.Path(
                    exists=validate_config_path,
                    file_okay=True,
                    dir_okay=False,
                    resolve_path=True,
                    path_type=pathlib.Path,
                ),
            ),
            click.option("--pull/--no-pull", default=True, hidden=True),
            click.option("--base-image", hidden=True),
            click.option("--install-command", hidden=True),
            click.option("--build-command", hidden=True),
            click.option("--api-version", type=str, hidden=True),
        ]
        if include_docker_args:
            # Only attach build args to the default command; on the group they
            # would capture subcommand names like `list` before Click resolves
            # them, making those subcommands unreachable.
            decorators.append(
                click.argument("docker_build_args", nargs=-1, type=click.UNPROCESSED)
            )
        for decorator in reversed(decorators):
            target = decorator(target)
        return target

    return _apply(func) if func is not None else _apply


@cli.group(
    cls=DeployGroup,
    help=(
        "[Beta] Build and deploy a LangGraph image to LangSmith Deployments.\n\n"
        "This command is in beta and under active development. "
        "Expect frequent updates and improvements.\n\n"
        "Run from the root of your LangGraph project (where langgraph.json "
        "is located). This command also accepts build flags (--base-image, "
        "--config, --pull, etc.). See 'langgraph build --help' for details."
    ),
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
    invoke_without_command=True,  # allow `deploy` click group to execute without command
)
@_deploy_base_options(include_docker_args=False, validate_config_path=False)
@click.pass_context
@log_command
def deploy(ctx: click.Context, **_: object):
    # We register deploy as both a group and a command here.
    # if we detect no subcommand, we run _deploy (basically run langgraph deploy as a top level command)
    # otherwise, we return None here and click will proceed to actually run the subcommand (list or delete)
    if ctx.invoked_subcommand is not None:
        return
    docker_build_args = tuple(ctx.args)
    ctx.args = []  # Prevent Click from re-processing passthrough args later.
    return ctx.forward(_deploy, docker_build_args=docker_build_args)


@_deploy_base_options()
@click.command(context_settings=dict(ignore_unknown_options=True))
def _deploy(
    config: pathlib.Path,
    pull: bool,
    verbose: bool,
    api_version: str | None,
    host_url: str | None,
    api_key: str | None,
    deployment_id: str | None,
    deployment_type: str,
    name: str | None,
    image_name: str | None,
    tag: str,
    base_image: str | None,
    install_command: str | None,
    build_command: str | None,
    no_wait: bool,
    docker_build_args: Sequence[str],
):
    click.secho(
        "Note: 'langgraph deploy' is in beta. Expect frequent updates and improvements.",
        fg="yellow",
    )
    click.echo()
    config_json = langgraph_cli.config.validate_config_file(config)
    warn_non_wolfi_distro(config_json)

    env_vars = _parse_env_from_config(config_json, config)

    if not deployment_id and not name:
        name = env_vars.get(_DEPLOYMENT_NAME_ENV)
    if not deployment_id and not name:
        default_name = _normalize_image_name(pathlib.Path.cwd().name)
        name = click.prompt("Deployment name", default=default_name)

    secrets = _secrets_from_env(env_vars)

    # Use buildx to cross-compile for amd64 when running on a non-x86_64 host
    # (e.g. Apple Silicon). On amd64 hosts, plain docker build is sufficient.
    needs_buildx = platform.machine() != "x86_64"
    local_tag = f"langgraph-deploy-tmp:{int(time.time())}"

    with Runner() as runner:
        if shutil.which("docker") is None:
            raise click.UsageError(
                "Docker is required but not installed.\n"
                "Install Docker Desktop: https://docs.docker.com/get-docker/\n\n"
                "Remote builds (no Docker required) are coming in a future update."
            )
        if needs_buildx:
            try:
                runner.run(subp_exec("docker", "buildx", "version", collect=True))
            except click.exceptions.Exit:
                raise click.UsageError(
                    "Docker Buildx is required but not installed.\n"
                    "Your machine architecture ("
                    + platform.machine()
                    + ") requires Buildx to cross-compile images for linux/amd64.\n"
                    "Install Buildx: https://docs.docker.com/build/install-buildx/\n\n"
                    "Remote builds (no Docker required) are coming in a future update."
                ) from None

        def log_step(message: str) -> None:
            click.secho(message, fg="cyan")

        client = _create_host_backend_client(host_url, api_key, env_vars=env_vars)
        step = 1
        needs_creation = False

        if deployment_id:
            log_step(f"{step}. Using deployment {deployment_id}")
            _call_host_backend_with_optional_tenant(
                client, lambda c: c.get_deployment(deployment_id)
            )
            step += 1
        else:
            log_step(f"{step}. Looking up deployment '{name}'")
            existing = _call_host_backend_with_optional_tenant(
                client, lambda c: c.list_deployments(name_contains=name)
            )
            found_id = None
            if isinstance(existing, dict):
                for dep in existing.get("resources", []):
                    if isinstance(dep, dict) and dep.get("name") == name:
                        found_id = dep.get("id")
                        break
            if found_id:
                deployment_id = str(found_id)
                click.secho(
                    f"   Found existing deployment (ID: {deployment_id})",
                    fg="green",
                )
            else:
                needs_creation = True
                click.secho(
                    "   No deployment found. Will create after build.", fg="yellow"
                )
            step += 1

        # -- Step: Build image --
        log_step(f"{step}. Building image")
        if needs_buildx:
            build_flags: list[str] = [
                "--platform",
                "linux/amd64",
                "--load",
            ]
            if not verbose:
                build_flags.append("--progress=quiet")
            with Progress(message="Building...", elapsed=not verbose):
                _build(
                    runner,
                    lambda _msg: None,
                    config,
                    config_json,
                    base_image,
                    api_version,
                    pull,
                    local_tag,
                    docker_build_args,
                    install_command,
                    build_command,
                    docker_command=("docker", "buildx", "build"),
                    extra_flags=build_flags,
                    verbose=verbose,
                )
        else:
            with Progress(message="Building...", elapsed=not verbose):
                _build(
                    runner,
                    lambda _msg: None,
                    config,
                    config_json,
                    base_image,
                    api_version,
                    pull,
                    local_tag,
                    docker_build_args,
                    install_command,
                    build_command,
                    verbose=verbose,
                )
        step += 1

        if needs_creation:
            log_step(f"{step}. Creating deployment '{name}'")
            payload = {
                "name": name,
                "source": "internal_docker",
                "source_config": {"deployment_type": deployment_type},
                "source_revision_config": {},
                "secrets": secrets,
            }
            created = client.create_deployment(payload)
            created_id = created.get("id") if isinstance(created, dict) else None
            if not isinstance(created_id, str) or not created_id:
                raise HostBackendError(
                    "POST /v2/deployments succeeded but response missing a valid 'id'"
                )
            deployment_id = created_id
            click.secho(f"   Deployment ID: {deployment_id}", fg="green")
            step += 1

        # -- Step: Get push token and authenticate --
        log_step(f"{step}. Requesting push token")
        try:
            push_data = client.request_push_token(deployment_id)
        except HostBackendError as err:
            if (
                err.status_code == 400
                and "only available for 'internal_docker' source deployments"
                in err.message
            ):
                raise click.ClickException(
                    f"Deployment '{deployment_id}' was not created by 'langgraph deploy' "
                    "and cannot be updated with this command.\n"
                    "Please create a new deployment by running 'langgraph deploy' "
                    "without --deployment-id, or use a different --name."
                ) from None
            raise
        deployment_token = push_data.get("token")
        registry_url = push_data.get("registry_url")
        if not deployment_token or not registry_url:
            raise click.ClickException(
                "Push token response missing token or registry_url"
            )
        step += 1

        normalized_registry = registry_url.rstrip("/")
        if "://" in normalized_registry:
            normalized_registry = normalized_registry.split("//", 1)[1]
        repo_seed = image_name or name or config.parent.name
        repo_name = _normalize_image_name(repo_seed)
        tag_value = _normalize_image_tag(tag)
        remote_image = f"{normalized_registry}/{repo_name}:{tag_value}"

        registry_host = normalized_registry.split("/")[0]

        # Use a clean Docker config with only the push token so that
        # system credential helpers (e.g. gcloud) don't interfere.
        with _docker_config_for_token(registry_host, deployment_token) as cfg:
            log_step(f"{step}. Logging into {registry_host}")
            token_input = (
                deployment_token
                if deployment_token.endswith("\n")
                else f"{deployment_token}\n"
            )
            runner.run(
                subp_exec(
                    "docker",
                    "--config",
                    cfg,
                    "login",
                    "-u",
                    "oauth2accesstoken",
                    "--password-stdin",
                    registry_host,
                    input=token_input,
                    verbose=verbose,
                )
            )
            step += 1

            # -- Step: Tag and push --
            log_step(f"{step}. Pushing image {remote_image}")
            runner.run(
                subp_exec(
                    "docker",
                    "tag",
                    local_tag,
                    remote_image,
                    verbose=verbose,
                )
            )
            max_push_retries = 3
            for attempt in range(max_push_retries):
                try:
                    with Progress(message="Pushing...", elapsed=not verbose):
                        runner.run(
                            subp_exec(
                                "docker",
                                "--config",
                                cfg,
                                "push",
                                remote_image,
                                verbose=verbose,
                            )
                        )
                    break
                except click.exceptions.Exit:
                    if attempt < max_push_retries - 1:
                        click.secho(
                            f"   Push failed, retrying (attempt {attempt + 2} of {max_push_retries})...",
                            fg="yellow",
                        )
                    else:
                        raise
        step += 1

        # -- Step: Update deployment --
        log_step(f"{step}. Updating deployment {deployment_id}")
        updated = client.update_deployment(deployment_id, remote_image, secrets=secrets)
        tenant_id = updated.get("tenant_id") if isinstance(updated, dict) else None
        if tenant_id:
            status_url = (
                f"https://smith.langchain.com/o/{tenant_id}"
                f"/host/deployments/{deployment_id}"
            )
            click.secho(f"   View status: {status_url}", fg="cyan")

        if no_wait:
            click.secho("   Deployment updated", fg="green")
            return

        # -- Poll revision status --
        revisions_resp = client.list_revisions(deployment_id, limit=1)
        resources = (
            revisions_resp.get("resources", [])
            if isinstance(revisions_resp, dict)
            else []
        )
        if not resources:
            click.secho("   Deployment updated", fg="green")
            return

        revision_id = str(resources[0]["id"])
        last_status = ""

        deadline = time.time() + 300
        with Progress(message="Deploying...", elapsed=True) as set_progress:
            while time.time() < deadline:
                rev = client.get_revision(deployment_id, revision_id)
                status = (
                    rev.get("status", "UNKNOWN") if isinstance(rev, dict) else "UNKNOWN"
                )
                if status != last_status:
                    last_status = status
                    # pause spinner so we can avoid conflict when writing status
                    set_progress("")
                    click.secho(f"   Status: {status}", fg="cyan")
                    if status in _TERMINAL_STATUSES:
                        break
                    set_progress(f"{status}...")
                time.sleep(1)
            else:
                set_progress("")

        dep_info = client.get_deployment(deployment_id)
        custom_url = None
        if isinstance(dep_info, dict):
            sc = dep_info.get("source_config")
            if isinstance(sc, dict):
                custom_url = sc.get("custom_url")

        if last_status == "DEPLOYED":
            click.secho("   Deployment successful!", fg="green")
            if custom_url:
                click.secho(f"   URL: {custom_url}", fg="green")
        elif last_status in ("BUILD_FAILED", "DEPLOY_FAILED", "CREATE_FAILED"):
            click.secho(f"   Deployment failed: {last_status}", fg="red")
            raise click.exceptions.Exit(1)
        else:
            click.secho(
                f"   Timed out waiting for deployment (last status: {last_status}).",
                fg="yellow",
            )
            if custom_url:
                click.secho(
                    f"   Check status at: {custom_url}",
                    fg="yellow",
                )
            else:
                click.secho(
                    "   Check status in the LangSmith Deployments dashboard.",
                    fg="yellow",
                )


def _create_host_backend_client(
    host_url: str | None,
    api_key: str | None,
    env_vars: dict[str, str] | None = None,
) -> HostBackendClient:
    if env_vars is None:
        env_vars = _parse_env_from_config({}, pathlib.Path.cwd() / DEFAULT_CONFIG)
    resolved_api_key = api_key
    if not resolved_api_key:
        for key_name in _API_KEY_ENV_NAMES:
            val = env_vars.get(key_name)
            if val:
                resolved_api_key = val
                break
            val = os.environ.get(key_name)
            if val:
                resolved_api_key = val
                break
    if not resolved_api_key:
        click.secho(
            "No LangSmith API key found. Create one at Settings > API Keys in LangSmith.",
            fg="yellow",
        )
        resolved_api_key = click.prompt("Enter LangSmith API key", hide_input=True)
    return HostBackendClient(host_url, resolved_api_key)


def _call_host_backend_with_optional_tenant(
    client: HostBackendClient,
    operation: Callable[[HostBackendClient], object],
) -> object:
    """Run *operation*, prompting for a workspace ID on org-scoped 403s.

    On success the original *client* is returned as-is.  If the user is
    prompted for a workspace ID, the tenant header is set on *client*
    in-place so all subsequent calls through the same instance are
    tenant-aware.
    """
    try:
        return operation(client)
    except HostBackendError as err:
        if err.status_code == 403 and "requires workspace specification" in err.message:
            click.secho(
                "Your API key is org-scoped and requires a workspace ID.",
                fg="yellow",
            )
            click.secho(
                "Find your workspace ID in LangSmith under Settings > Workspaces.",
                fg="yellow",
            )
            tenant_id = click.prompt("Workspace ID")
            client._client.headers["X-Tenant-ID"] = tenant_id
            return operation(client)
        raise


@OPT_HOST_API_KEY
@OPT_HOST_URL
@click.option(
    "--name-contains",
    default="",
    help="Only show deployments whose names contain this value.",
)
@deploy.command("list", help="[Beta] List LangSmith Deployments.")
def deploy_list(api_key: str | None, host_url: str | None, name_contains: str) -> None:
    client = _create_host_backend_client(host_url, api_key)
    response = _call_host_backend_with_optional_tenant(
        client,
        lambda c: c.list_deployments(name_contains=name_contains),
    )
    resources = response.get("resources", []) if isinstance(response, dict) else []
    deployments = [item for item in resources if isinstance(item, dict)]
    if not deployments:
        click.echo("No deployments found.")
        return
    click.echo(format_deployments_table(deployments))


@OPT_HOST_API_KEY
@OPT_HOST_URL
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Delete without prompting for confirmation.",
)
@click.argument("deployment_id")
@deploy.command(
    "delete",
    help=(
        "[Beta] Delete a LangSmith Deployment.\n\n"
        "Use the `deploy list` command to list deployment IDs."
    ),
)
def deploy_delete(
    api_key: str | None, host_url: str | None, force: bool, deployment_id: str
) -> None:
    if not force:
        response = click.prompt(
            click.style(
                f"Are you sure you want to delete deployment ID {deployment_id}? (Y/n)",
                fg="yellow",
            ),
            default="Y",
            show_default=False,
        )
        if response.strip().lower() not in {"y", "yes"}:
            raise click.Abort()
    client = _create_host_backend_client(host_url, api_key)
    _call_host_backend_with_optional_tenant(
        client,
        lambda c: c.delete_deployment(deployment_id),
    )
    click.secho(f"Deleted deployment {deployment_id}.", fg="green")


def _normalize_image_name(value: str | None) -> str:
    """Sanitize a deployment/directory name into a valid Docker repository name.

    Docker repository names must be lowercase and may only contain
    [a-z0-9._-].  Invalid characters are replaced with hyphens.
    """
    if not value:
        return "app"
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-.")
    return slug or "app"


def _normalize_image_tag(value: str) -> str:
    """Validate and return a Docker image tag.

    Tags may only contain [A-Za-z0-9_.-].  Defaults to "latest" when empty.
    """
    if not value:
        value = "latest"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise click.UsageError(
            "Image tag may only contain characters A-Z, a-z, 0-9, '_', '-', '.'"
        )
    return value


@OPT_HOST_API_KEY
@OPT_HOST_DEPLOYMENT_NAME
@click.option(
    "--deployment-id",
    help="Deployment ID. If omitted, --name is used to find the deployment.",
)
@click.option(
    "--type",
    "log_type",
    type=click.Choice(["deploy", "build"]),
    default="deploy",
    show_default=True,
    help=(
        "Log stream to fetch: 'deploy' shows agent server runtime logs; "
        "'build' shows build logs (for deployments built remotely)."
    ),
)
@click.option(
    "--revision-id",
    help="Specific revision ID. For build logs, defaults to latest revision.",
)
@click.option(
    "--level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Filter by log level.",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    show_default=True,
    help="Max log entries to fetch.",
)
@click.option(
    "--query",
    "-q",
    help="Search string filter.",
)
@click.option(
    "--start-time",
    help="ISO8601 start time (e.g. 2026-03-08T00:00:00Z).",
)
@click.option(
    "--end-time",
    help="ISO8601 end time. (e.g. 2026-03-08T00:00:00Z)",
)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    default=False,
    help="Continuously poll for new logs.",
)
@OPT_HOST_URL
@deploy.command(
    "logs",
    help=(
        "[Beta] Fetch LangSmith Deployment logs. Use 'deploy' for agent runtime "
        "logs, or 'build' for remote build logs."
    ),
)
@log_command
def deploy_logs(
    api_key: str | None,
    name: str | None,
    deployment_id: str | None,
    log_type: str,
    revision_id: str | None,
    level: str | None,
    limit: int,
    query: str | None,
    start_time: str | None,
    end_time: str | None,
    follow: bool,
    host_url: str,
):
    env_vars = _parse_env_from_config({}, pathlib.Path.cwd() / DEFAULT_CONFIG)
    client = _create_host_backend_client(host_url, api_key, env_vars=env_vars)
    if not deployment_id and not name:
        name = env_vars.get(_DEPLOYMENT_NAME_ENV)
    dep_id = _call_host_backend_with_optional_tenant(
        client, lambda c: resolve_deployment_id(c, deployment_id, name)
    )

    if log_type == "build" and not revision_id:
        revisions_resp = client.list_revisions(dep_id, limit=1)
        resources = (
            revisions_resp.get("resources", [])
            if isinstance(revisions_resp, dict)
            else []
        )
        if not resources:
            raise click.ClickException(
                "No revisions found for this deployment. Cannot fetch build logs."
            )
        revision_id = str(resources[0]["id"])
        click.secho(f"Using latest revision: {revision_id}", fg="cyan")

    payload: dict = {"limit": limit, "order": "desc"}
    if level:
        payload["level"] = level.upper()
    if query:
        payload["query"] = query
    if start_time:
        payload["start_time"] = start_time
    if end_time:
        payload["end_time"] = end_time

    def _fetch(request_payload: dict) -> list[dict]:
        if log_type == "build":
            resp = client.get_build_logs(dep_id, revision_id, request_payload)
        else:
            resp = client.get_deploy_logs(dep_id, request_payload, revision_id)

        if isinstance(resp, dict):
            return resp.get("logs", [])
        return []

    def _print_entries(entries: list[dict], *, reverse: bool = False) -> None:
        iterable = reversed(entries) if reverse else entries
        for entry in iterable:
            line = format_log_entry(entry)
            fg = level_fg(entry.get("level", ""))
            click.secho(line, fg=fg)

    def _fetch_and_print(request_payload: dict, *, reverse: bool = False) -> list[dict]:
        entries = _fetch(request_payload)
        _print_entries(entries, reverse=reverse)
        return entries

    def _fetch_and_print_new(request_payload: dict, seen_ids: set[str]) -> list[dict]:
        entries = _fetch(request_payload)
        new = [e for e in entries if e.get("id", "") not in seen_ids]
        if new:
            _print_entries(new)
            seen_ids.update(e.get("id", "") for e in new)
        return new

    # initial log fetch will be newest -> oldest, so we need to reverse
    entries = _fetch_and_print(payload, reverse=True)

    if not follow:
        if not entries:
            click.secho("No log entries found.", fg="yellow")
        return

    payload["order"] = "asc"
    seen_ids: set[str] = {e.get("id", "") for e in entries if e.get("id")}

    def _update_start_time(ts) -> None:
        if ts is None:
            return
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            payload["start_time"] = dt.isoformat()
        else:
            payload["start_time"] = str(ts)

    if entries:
        # entries are in descending order here, so index 0 is the newest log
        _update_start_time(entries[0].get("timestamp"))

    try:
        while True:
            time.sleep(2)
            new_entries = _fetch_and_print_new(payload, seen_ids)
            if new_entries:
                _update_start_time(new_entries[-1].get("timestamp"))
    except KeyboardInterrupt:
        click.echo("\nStopped.")


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
@OPT_ENGINE_RUNTIME_MODE
@log_command
def dockerfile(
    save_path: str,
    config: pathlib.Path,
    add_docker_compose: bool,
    base_image: str | None = None,
    api_version: str | None = None,
    engine_runtime_mode: str = "combined_queue_worker",
) -> None:
    save_path = pathlib.Path(save_path).absolute()
    secho(f"🔍 Validating configuration at path: {config}", fg="yellow")
    config_json = langgraph_cli.config.validate_config_file(config)
    warn_non_wolfi_distro(config_json)
    secho("✅ Configuration validated!", fg="green")

    effective_base_image = base_image
    if engine_runtime_mode == "distributed" and not base_image:
        effective_base_image = langgraph_cli.config.default_base_image(
            config_json, engine_runtime_mode=engine_runtime_mode
        )

    secho(f"📝 Generating Dockerfile at {save_path}", fg="yellow")
    dockerfile, additional_contexts = langgraph_cli.config.config_to_docker(
        config_path=config,
        config=config_json,
        base_image=effective_base_image,
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
        webhooks=config_json.get("webhooks"),
        studio_url=studio_url,
        allow_blocking=allow_blocking,
        tunnel=tunnel,
        server_level=server_log_level,
        checkpointer=config_json.get("checkpointer"),
        disable_persistence=config_json.get("disable_persistence", False),
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
    engine_runtime_mode: str = "combined_queue_worker",
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
        image=image,
        base_image=base_image,
        api_version=api_version,
        engine_runtime_mode=engine_runtime_mode,
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
        engine_runtime_mode=engine_runtime_mode,
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
    engine_runtime_mode: str = "combined_queue_worker",
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
        if engine_runtime_mode == "distributed":
            executor_base = langgraph_cli.config.default_base_image(
                config_json, engine_runtime_mode="distributed"
            )
            runner.run(
                subp_exec(
                    "docker",
                    "pull",
                    langgraph_cli.config.docker_tag(
                        config_json, executor_base, api_version
                    ),
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
        engine_runtime_mode=engine_runtime_mode,
        image=image,
        base_image=base_image,
    )
    return args, stdin
