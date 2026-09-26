"""Deploy command and subcommands for the LangGraph CLI."""

import base64
import json as json_mod
import os
import pathlib
import platform
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Protocol, TypeVar

import click
import click.exceptions
from dotenv import dotenv_values, set_key

import langgraph_cli.config
from langgraph_cli.analytics import log_command
from langgraph_cli.config import Config
from langgraph_cli.constants import DEFAULT_CONFIG
from langgraph_cli.dependency_tracking import find_tracked_packages
from langgraph_cli.docker import build_docker_image, can_build_locally
from langgraph_cli.exec import CommandRunner, Runner, subp_exec
from langgraph_cli.host_backend import (
    MAX_PAGE_SIZE,
    ControlPlaneEndpoints,
    HostBackendClient,
    HostBackendError,
    SourceName,
)
from langgraph_cli.image_reference import ImageReference
from langgraph_cli.progress import Progress
from langgraph_cli.util import warn_non_wolfi_distro

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

_T = TypeVar("_T")

_DEPLOYMENT_NAME_ENV = "LANGSMITH_DEPLOYMENT_NAME"
_DEFAULT_IMAGE_TAG = "latest"
_DEPLOYMENT_PLATFORM = "linux/amd64"
_NATIVE_AMD64_MACHINE = "x86_64"
_PUSH_ATTEMPTS = 3
_LOCAL_BUILD_TAG_PREFIX = "langgraph-deploy-tmp"
_OPERATOR_DEFAULT_RESOURCE_SPEC: Mapping[str, object] = {}
_CUSTOMER_REGISTRY_SOURCE: SourceName = "external_docker"
_LISTENER_REQUIRED_MARKER = "listener_id' is required"
_LISTENERS_SHOWN = 10
_LISTENER_NOT_FOUND_STATUSES = frozenset({404, 422})
_LISTENERS_DOCS_URL = "https://docs.langchain.com/langsmith/control-plane#listeners"
_NO_LISTENERS = (
    "This workspace has no listeners, so --listener-id and --k8s-namespace "
    "do not apply."
)


_TERMINAL_STATUSES = frozenset(
    [
        "DEPLOYED",
        "CREATE_FAILED",
        "BUILD_FAILED",
        "DEPLOY_FAILED",
        "SKIPPED",
    ]
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    """Captures the outcome of a build stage so the shared wait tail can be
    parameterized identically for both local and remote builds."""

    updated: dict = field(default_factory=dict)
    progress_message: str = ""
    timeout_seconds: int = 300
    poll_interval_seconds: int = 1
    no_result_message: str = "Deployment updated"
    on_poll: Callable[[str, str, Callable[[str], None]], None] | None = None
    on_interrupt: Callable[[str], None] | None = None
    show_build_logs_on_failure: bool = False


@dataclass(frozen=True, slots=True)
class ById:
    deployment_id: str


@dataclass(frozen=True, slots=True)
class ByName:
    name: str


@dataclass(frozen=True, slots=True)
class ByAgent:
    agent_id: str
    environment: str


DeploymentSelector = ById | ByName | ByAgent


@dataclass(frozen=True, slots=True)
class Listener:
    id: str
    compute_id: str
    namespaces: tuple[str, ...]

    @classmethod
    def from_resource(cls, resource: Mapping[str, object]) -> "Listener":
        identifier = str(resource.get("id") or "")
        if not identifier:
            raise HostBackendError(
                "The control plane returned a listener without an id."
            )
        compute_config = resource.get("compute_config")
        namespaces = (
            compute_config.get("k8s_namespaces")
            if isinstance(compute_config, Mapping)
            else None
        )
        return cls(
            identifier,
            str(resource.get("compute_id", "")),
            tuple(str(namespace) for namespace in namespaces)
            if isinstance(namespaces, list)
            else (),
        )


@dataclass(frozen=True, slots=True)
class Unplaced:
    @property
    def summary(self) -> str:
        return ""

    def source_config(self) -> dict[str, object]:
        return {}


@dataclass(frozen=True, slots=True)
class OnListener:
    listener_id: str
    k8s_namespace: str

    @property
    def summary(self) -> str:
        return (
            f"Deploying through listener {self.listener_id} "
            f"in namespace {self.k8s_namespace}"
        )

    def source_config(self) -> dict[str, object]:
        return {
            "listener_id": self.listener_id,
            "listener_config": {"k8s_namespace": self.k8s_namespace},
        }


Placement = Unplaced | OnListener


@dataclass(frozen=True, slots=True)
class RequestedPlacement:
    listener_id: str | None = None
    k8s_namespace: str | None = None

    @property
    def requested(self) -> bool:
        return self.listener_id is not None or self.k8s_namespace is not None

    def ensure_not_requested(self, deployment_id: str) -> None:
        if self.requested:
            raise click.UsageError(
                "Listener and namespace are fixed when a deployment is created. "
                f"Deployment {deployment_id} already exists, so drop --listener-id "
                "and --k8s-namespace, or create a new deployment with a different "
                "--name."
            )

    def on(self, listener: Listener) -> Placement:
        return OnListener(listener.id, self._namespace(listener))

    def among(self, listeners: Sequence[Listener]) -> Placement:
        if not listeners:
            if self.requested:
                raise click.UsageError(_NO_LISTENERS)
            return Unplaced()
        if len(listeners) > 1:
            raise click.UsageError(
                "This workspace has several listeners. Choose one with "
                f"--listener-id:\n{_describe_listeners(listeners)}"
            )
        return self.on(listeners[0])

    def _namespace(self, listener: Listener) -> str:
        if not listener.namespaces:
            raise click.UsageError(
                f"Listener {listener.id} serves no namespaces. Check its configuration."
            )
        if self.k8s_namespace is None:
            if len(listener.namespaces) == 1:
                return listener.namespaces[0]
            raise click.UsageError(
                f"Listener {listener.id} serves several namespaces. Choose one with "
                f"--k8s-namespace: {', '.join(listener.namespaces)}"
            )
        if self.k8s_namespace not in listener.namespaces:
            raise click.UsageError(
                f"Listener {listener.id} does not serve namespace "
                f"'{self.k8s_namespace}'. Choose one of: "
                f"{', '.join(listener.namespaces)}"
            )
        return self.k8s_namespace


def _describe_listeners(listeners: Sequence[Listener]) -> str:
    shown = listeners[:_LISTENERS_SHOWN]
    lines = [
        f"  {listener.id}  cluster {listener.compute_id}  "
        f"namespaces: {', '.join(listener.namespaces)}"
        for listener in shown
    ]
    if len(listeners) > len(shown):
        lines.append(f"  ... and {len(listeners) - len(shown)} more")
    if len(listeners) == MAX_PAGE_SIZE:
        lines.append(f"  (only the first {MAX_PAGE_SIZE} listeners were read)")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ExistingDeployment:
    id: str
    source: str | None


# ---------------------------------------------------------------------------
# Structured output emitter
# ---------------------------------------------------------------------------

_emitter: "_Emitter | None" = None
_no_input: bool = False


class _Emitter:
    """Dual-mode output: JSON-lines (``--json``) or human-readable click text."""

    def __init__(self, json_mode: bool) -> None:
        self._json = json_mode

    @property
    def json_mode(self) -> bool:
        return self._json

    # -- Structured event helpers ------------------------------------------

    def step(self, step: int, message: str, **extra: object) -> None:
        if self._json:
            self._write({"event": "step", "step": step, "message": message, **extra})
        else:
            click.secho(f"{step}. {message}", fg="cyan")

    def info(self, message: str, **extra: object) -> None:
        if self._json:
            self._write({"event": "info", "message": message, **extra})
        else:
            click.secho(f"   {message}", fg="green")

    def warn(self, message: str, **extra: object) -> None:
        """Warning nested under a step. Text mode indents; JSON mode strips leading whitespace."""
        if self._json:
            self._write({"event": "warn", "message": message.lstrip(), **extra})
        else:
            click.secho(f"   {message}", fg="yellow")

    def note(self, message: str, **extra: object) -> None:
        """Top-level banner (pre-step). Text mode does not indent."""
        if self._json:
            self._write({"event": "note", "message": message, **extra})
        else:
            click.secho(message, fg="yellow")

    def error(self, message: str, **extra: object) -> None:
        if self._json:
            self._write({"event": "error", "message": message, **extra})
        else:
            click.secho(f"   {message}", fg="red")

    def status_change(
        self,
        status: str,
        elapsed_seconds: float,
        finished: bool = False,
    ) -> None:
        mins, secs = divmod(int(elapsed_seconds), 60)
        elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        if self._json:
            self._write(
                {
                    "event": "status_change",
                    "status": status,
                    "elapsed_seconds": round(elapsed_seconds, 1),
                    "message": f"{status}... ({elapsed_str})",
                }
            )
        else:
            click.echo(f"   {status}... ({elapsed_str})")

    def log(self, message: str) -> None:
        if self._json:
            self._write({"event": "log", "message": message})
        else:
            click.echo(f"   | {message}")

    def status_url(self, url: str) -> None:
        if self._json:
            self._write({"event": "status_url", "url": url})
        else:
            click.secho(f"   View status: {url}", fg="cyan")

    def result(
        self,
        status: str,
        *,
        deployment_id: str,
        url: str | None = None,
        status_url: str | None = None,
        fallback_status_message: str | None = None,
    ) -> None:
        if self._json:
            if status == "succeeded":
                message = "Deployment successful!"
            elif status == "failed":
                message = "Deployment failed"
            else:
                message = "Timed out waiting for deployment."
            payload: dict = {
                "event": "result",
                "status": status,
                "deployment_id": deployment_id,
                "message": message,
            }
            if url:
                payload["url"] = url
            if status_url:
                payload["status_url"] = status_url
            self._write(payload)
        else:
            if status == "succeeded":
                click.secho("   Deployment successful!", fg="green")
                if url:
                    click.secho(f"   URL: {url}", fg="green")
                if status_url:
                    click.secho(f"   View status: {status_url}", fg="green")
            elif status == "failed":
                click.secho("   Deployment failed", fg="red")
                if status_url:
                    click.secho(f"   View status: {status_url}", fg="red")
            elif status == "timed_out":
                click.secho("   Timed out waiting for deployment.", fg="yellow")
                if status_url:
                    click.secho(f"   Check status at: {status_url}", fg="yellow")
                elif fallback_status_message:
                    click.secho(f"   {fallback_status_message}", fg="yellow")

    def heartbeat(self, status: str, elapsed_seconds: float) -> None:
        if self._json:
            mins, secs = divmod(int(elapsed_seconds), 60)
            elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            self._write(
                {
                    "event": "heartbeat",
                    "status": status,
                    "elapsed_seconds": round(elapsed_seconds, 1),
                    "message": f"{status}... ({elapsed_str})",
                }
            )

    def upload_progress(self, size_mb: float, pct: int) -> None:
        if self._json:
            self._write(
                {
                    "event": "upload_progress",
                    "size_mb": round(size_mb, 1),
                    "pct": pct,
                }
            )
        else:
            click.echo(f"\r   Uploading ({size_mb:.1f} MB)... {pct}%", nl=False)

    def _write(self, obj: dict) -> None:
        import sys as _sys

        _sys.stdout.write(json_mod.dumps(obj, default=str) + "\n")
        _sys.stdout.flush()


def _get_emitter() -> _Emitter:
    """Return the module-level emitter (falls back to text mode)."""
    return _emitter or _Emitter(json_mode=False)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def deployment_selector(deployment_id: str | None, name: str | None) -> ById | ByName:
    if deployment_id:
        return ById(deployment_id)
    if name:
        return ByName(name)
    raise click.UsageError("Either --deployment-id or --name is required.")


def validate_deploy_commands(
    install_command: str | None, build_command: str | None
) -> None:
    """Validate optional deploy commands for disallowed content."""
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


# ---------------------------------------------------------------------------
# Deployment lookup
# ---------------------------------------------------------------------------


def _source_of(resource: object) -> str | None:
    if not isinstance(resource, dict):
        return None
    source = resource.get("source")
    return source if isinstance(source, str) else None


def find_deployment_by_name(
    client: HostBackendClient, name: str
) -> ExistingDeployment | None:
    listed = client.list_deployments(name=name, name_contains=name, limit=MAX_PAGE_SIZE)
    for resource in listed:
        if resource.get("name") == name and resource.get("id"):
            return ExistingDeployment(str(resource["id"]), _source_of(resource))
    if len(listed) >= MAX_PAGE_SIZE:
        raise click.ClickException(
            "This workspace has more deployments than the CLI can search, so it "
            f"cannot tell whether '{name}' already exists. Pass --deployment-id to "
            "update an existing deployment."
        )
    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def normalize_name(value: str | None) -> str:
    """Sanitize a deployment/directory name into a valid deployment name.

    LangSmith Deployment names only allow lowercase
    alphanumeric characters and hyphens ([a-z0-9-]).
    Invalid characters are replaced with hyphens.
    """
    if not value:
        return "app"
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return slug or "app"


def normalize_image_tag(value: str) -> str:
    """Validate and return a Docker image tag.

    Tags may only contain [A-Za-z0-9_.-].  Defaults to "latest" when empty.
    """
    if not value:
        value = _DEFAULT_IMAGE_TAG
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise click.UsageError(
            "Image tag may only contain characters A-Z, a-z, 0-9, '_', '-', '.'"
        )
    return value


def _validate_prebuilt_image(
    runner: CommandRunner, image: str, *, verbose: bool
) -> None:
    """Ensure a prebuilt image exists locally for linux/amd64."""
    try:
        stdout, _ = runner.run(
            subp_exec(
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Os}}/{{.Architecture}}",
                image,
                verbose=verbose,
                collect=True,
            )
        )
    except FileNotFoundError:
        raise click.ClickException(
            "Docker is required but not installed.\n"
            "Install Docker Desktop: https://docs.docker.com/get-docker/"
        ) from None
    except click.exceptions.Exit:
        raise click.ClickException(
            f"Docker image '{image}' was not found locally. Build or pull the image "
            "before deploying with --image."
        ) from None

    image_platform = (stdout or "").strip()
    if image_platform != _DEPLOYMENT_PLATFORM:
        detected = image_platform or "unknown"
        raise click.ClickException(
            f"Docker image '{image}' targets {detected}, but LangSmith Deployment "
            f"requires {_DEPLOYMENT_PLATFORM}. Rebuild or pull the image for "
            f"{_DEPLOYMENT_PLATFORM} before deploying with --image."
        )
    _get_emitter().info(f"Image is available for {_DEPLOYMENT_PLATFORM}")


def _extract_deployment_url(deployment: dict[str, object]) -> str:
    source_config = deployment.get("source_config")
    if isinstance(source_config, dict):
        custom_url = source_config.get("custom_url")
        if isinstance(custom_url, str) and custom_url:
            return custom_url
    return "-"


def format_deployments_table(deployments: Sequence[dict[str, object]]) -> str:
    headers = ("Deployment ID", "Deployment Name", "Deployment URL")
    rows = [
        (
            str(deployment.get("id", "-") or "-"),
            str(deployment.get("name", "-") or "-"),
            _extract_deployment_url(deployment),
        )
        for deployment in deployments
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    lines = [format_row(headers), format_row(tuple("-" * width for width in widths))]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def format_revisions_table(revisions: Sequence[dict[str, object]]) -> str:
    headers = ("Revision ID", "Status", "Created At")
    latest_deployed_seen = False
    rows = []
    for revision in revisions:
        status = str(revision.get("status", "-") or "-")
        if status == "DEPLOYED":
            if latest_deployed_seen:
                status = "REPLACED"
            else:
                latest_deployed_seen = True
        rows.append(
            (
                str(revision.get("id", "-") or "-"),
                status,
                str(revision.get("created_at", "-") or "-"),
            )
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    lines = [format_row(headers), format_row(tuple("-" * width for width in widths))]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def format_timestamp(ts) -> str:
    """Convert a timestamp (epoch ms or string) to a readable string."""
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts) if ts else ""


def format_log_entry(entry: dict) -> str:
    """Format a single log entry for display."""
    ts = format_timestamp(entry.get("timestamp", ""))
    level = entry.get("level", "")
    message = entry.get("message", "")
    if ts and level:
        return f"[{ts}] [{level}] {message}"
    elif ts:
        return f"[{ts}] {message}"
    return message


def level_fg(level: str) -> str | None:
    """Return click color for a log level."""
    level_upper = level.upper() if level else ""
    if level_upper in {"ERROR", "CRITICAL"}:
        return "red"
    if level_upper == "WARNING":
        return "yellow"
    return None


# ---------------------------------------------------------------------------
# Env / secrets helpers
# ---------------------------------------------------------------------------


def _resolve_env_path(
    config_json: dict, config_path: pathlib.Path
) -> pathlib.Path | None:
    """Return the .env file path implied by the config, or None for inline dicts."""
    env_field = config_json.get("env")
    if isinstance(env_field, dict) and env_field:
        return None
    if isinstance(env_field, str):
        env_path = (config_path.parent / env_field).resolve()
        if not env_path.exists():
            _get_emitter().note(
                f"Warning: env file '{env_field}' specified in langgraph.json not found."
            )
            return None
        return env_path
    return pathlib.Path.cwd() / ".env"


def _parse_env_from_config(
    config_json: dict, config_path: pathlib.Path
) -> dict[str, str]:
    """Resolve env vars from langgraph.json 'env' field or a .env fallback."""
    env_field = config_json.get("env")
    if isinstance(env_field, dict) and env_field:
        return {str(k): str(v) for k, v in env_field.items()}
    env_path = _resolve_env_path(config_json, config_path)
    if env_path is None:
        return {}
    return {k: v for k, v in dotenv_values(env_path).items() if v is not None}


def _env_without_deployment_name(env_vars: dict[str, str]) -> dict[str, str]:
    """Return env vars copy with deployment-name key removed."""
    filtered = dict(env_vars)
    filtered.pop(_DEPLOYMENT_NAME_ENV, None)
    return filtered


def _secrets_from_env(
    env_vars: dict[str, str],
) -> list[dict[str, str]]:
    """Convert env dict to secrets list, filtering reserved vars with warnings."""
    secrets: list[dict[str, str]] = []
    for name, value in env_vars.items():
        if name in RESERVED_ENV_VARS:
            _get_emitter().note(f"Skipping reserved env var: {name}")
            continue
        if not value:
            continue
        secrets.append({"name": name, "value": value})
    return secrets


# ---------------------------------------------------------------------------
# Build mode resolution
# ---------------------------------------------------------------------------


def _resolve_build_mode(
    remote_build_flag: bool | None,
    *,
    force_local: bool = False,
) -> tuple[bool, str | None]:
    """Determine whether to use a remote build.

    Returns (use_remote_build, local_build_error).  Raises UsageError when
    --no-remote is set but the machine cannot build locally. When
    `force_local` is set, the function short-circuits and always selects a
    local build.
    """
    if force_local:
        return False, None
    local_build_supported, local_build_error = can_build_locally()

    if remote_build_flag is True:
        return True, local_build_error

    if remote_build_flag is False:
        if not local_build_supported:
            details = "\n\nOr re-run with --remote to use remote builds."
            raise click.UsageError(
                f"{local_build_error or 'Unable to build locally.'}{details}"
            )
        return False, None

    # auto-detect
    return not local_build_supported, local_build_error


# ---------------------------------------------------------------------------
# Deployment orchestration helpers
# ---------------------------------------------------------------------------


def _log_deploy_step(step: int, message: str, **extra: object) -> None:
    _get_emitter().step(step, message, **extra)


def _fetch_deployment(
    client: HostBackendClient, step: int, selector: ById
) -> tuple[ExistingDeployment, int]:
    _log_deploy_step(step, f"Using deployment {selector.deployment_id}")
    resource = _call_host_backend_with_optional_tenant(
        client, lambda c: c.get_deployment(selector.deployment_id)
    )
    return ExistingDeployment(selector.deployment_id, _source_of(resource)), step + 1


def _find_deployment(
    client: HostBackendClient,
    step: int,
    selector: ByName | ByAgent,
    *,
    not_found_message: str,
) -> tuple[ExistingDeployment | None, int]:
    if isinstance(selector, ByAgent):
        _log_deploy_step(
            step, f"Looking up agent '{selector.agent_id}' in {selector.environment}"
        )
        existing = _call_host_backend_with_optional_tenant(
            client,
            lambda c: c.list_deployments(
                agent_id=selector.agent_id,
                agent_environment=selector.environment,
                limit=MAX_PAGE_SIZE,
            ),
        )
        if len(existing) > 1:
            raise click.ClickException(
                "This control plane does not filter deployments by agent, so the "
                f"CLI cannot tell which one belongs to '{selector.agent_id}' in "
                f"{selector.environment}. Deploy by --name instead."
            )
        found = next(
            (
                ExistingDeployment(str(dep["id"]), _source_of(dep))
                for dep in existing
                if dep.get("id") and not dep.get("is_preview")
            ),
            None,
        )
    else:
        _log_deploy_step(step, f"Looking up deployment '{selector.name}'")
        found = _call_host_backend_with_optional_tenant(
            client, lambda c: find_deployment_by_name(c, selector.name)
        )
    em = _get_emitter()
    if found is None:
        em.warn(not_found_message)
    else:
        em.info(f"Found existing deployment (ID: {found.id})")
    return found, step + 1


@dataclass(frozen=True, slots=True)
class CreatedDeployment:
    id: str
    resource: dict[str, object]


def _create_deployment(
    client: HostBackendClient,
    step: int,
    *,
    name: str | None,
    source: str,
    source_config: dict[str, object],
    source_revision_config: dict[str, object],
    secrets: list[dict[str, str]],
    agent: dict[str, str] | None = None,
) -> tuple[CreatedDeployment, int]:
    _log_deploy_step(
        step,
        f"Creating deployment for agent '{agent['agent_id']}' in {agent['environment']}"
        if agent is not None
        else f"Creating deployment '{name}'",
    )
    try:
        created = client.create_deployment(
            name=name,
            source=source,
            source_config=source_config,
            source_revision_config=source_revision_config,
            secrets=secrets,
            agent=agent,
        )
    except HostBackendError as err:
        if agent is not None and err.status_code == 409:
            raise HostBackendError(
                "This agent already has a deployment in this environment.",
                status_code=409,
            ) from None
        raise
    created_id = created.get("id") if isinstance(created, dict) else None
    if not isinstance(created_id, str) or not created_id:
        raise HostBackendError(
            "POST /v2/deployments succeeded but response missing a valid 'id'"
        )
    if agent is not None:
        _get_emitter().info(f"Deployment name: {created.get('name')}")
    _get_emitter().info(f"Deployment ID: {created_id}", deployment_id=created_id)
    return CreatedDeployment(created_id, created), step + 1


def _get_deployment_status_url(
    updated: object, deployment_id: str, endpoints: ControlPlaneEndpoints
) -> str | None:
    tenant_id = updated.get("tenant_id") if isinstance(updated, dict) else None
    if not tenant_id:
        return None
    return f"{endpoints.dashboard_url}/o/{tenant_id}/host/deployments/{deployment_id}"


def _emit_deployment_status_url(
    updated: object, deployment_id: str, endpoints: ControlPlaneEndpoints
) -> str | None:
    url = _get_deployment_status_url(updated, deployment_id, endpoints)
    if url:
        _get_emitter().status_url(url)
    return url


def _poll_revision_status(
    client: HostBackendClient,
    deployment_id: str,
    *,
    progress_message: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
    on_poll: Callable[[str, str, Callable[[str], None]], None] | None = None,
    on_interrupt: Callable[[str], None] | None = None,
) -> tuple[str, str | None]:
    """Poll latest revision status until terminal status or timeout."""
    em = _get_emitter()
    revisions = client.list_revisions(deployment_id, limit=1)
    if not revisions:
        return "", None

    revision_id = str(revisions[0]["id"])
    last_status = ""
    deadline = time.time() + timeout_seconds
    start_time = time.monotonic()
    last_heartbeat = start_time
    json_mode = em.json_mode
    with Progress(
        message=progress_message, elapsed=True, json_mode=json_mode
    ) as set_progress:
        while time.time() < deadline:
            try:
                rev = client.get_revision(deployment_id, revision_id)
            except KeyboardInterrupt:
                set_progress("")
                if on_interrupt is not None:
                    on_interrupt(revision_id)
                    raise click.exceptions.Exit(1) from None
                raise

            status = (
                rev.get("status", "UNKNOWN") if isinstance(rev, dict) else "UNKNOWN"
            )
            if status != last_status:
                set_progress("")
                if last_status:
                    em.status_change(last_status, time.monotonic() - start_time)
                last_status = status
                if status in _TERMINAL_STATUSES:
                    break
                set_progress(f"{status}...")
                last_heartbeat = time.monotonic()
            elif json_mode and time.monotonic() - last_heartbeat > 10:
                em.heartbeat(last_status, time.monotonic() - start_time)
                last_heartbeat = time.monotonic()

            if on_poll is not None:
                on_poll(status, revision_id, set_progress)
            time.sleep(poll_interval_seconds)
        else:
            set_progress("")

    return last_status, revision_id


def _print_deployment_result(
    client: HostBackendClient,
    deployment_id: str,
    last_status: str,
    *,
    dashboard_label: str,
    status_url: str | None = None,
) -> None:
    """Print final deployment status and raise on failure."""
    em = _get_emitter()
    dep_info = client.get_deployment(deployment_id)
    custom_url = None
    if isinstance(dep_info, dict):
        sc = dep_info.get("source_config")
        if isinstance(sc, dict):
            custom_url = sc.get("custom_url")

    if last_status == "DEPLOYED":
        em.result(
            "succeeded",
            deployment_id=deployment_id,
            url=custom_url,
            status_url=status_url,
        )
    elif last_status in ("BUILD_FAILED", "DEPLOY_FAILED", "CREATE_FAILED"):
        em.result(
            "failed",
            deployment_id=deployment_id,
            status_url=status_url,
        )
        raise click.exceptions.Exit(1)
    else:
        em.result(
            "timed_out",
            deployment_id=deployment_id,
            status_url=status_url,
            fallback_status_message=(
                f"Check status in the LangSmith {dashboard_label}."
            ),
        )


# ---------------------------------------------------------------------------
# Docker push auth
# ---------------------------------------------------------------------------


@contextmanager
def _docker_config_for_token(registry_host: str, token: str):
    """Create a temporary Docker config with only the push token.

    Yields the path to a temporary config directory that can be passed
    to `docker --config <path>` so that system credential helpers
    (e.g. gcloud) don't interfere with the push token.
    """
    auth_b64 = base64.b64encode(f"oauth2accesstoken:{token}".encode()).decode()
    config_data = {"auths": {registry_host: {"auth": auth_b64}}}
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json_mod.dump(config_data, f)
        yield tmpdir


# ---------------------------------------------------------------------------
# GCS upload
# ---------------------------------------------------------------------------

_UPLOAD_TIMEOUT_SECONDS = 300
_BYTES_PER_MIB = 1_048_576


class _ProgressReader:
    """File-like wrapper that reports upload progress via the emitter."""

    def __init__(self, fobj, file_size: int, emitter: "_Emitter"):
        self._fobj = fobj
        self._file_size = file_size
        self._emitter = emitter
        self._uploaded = 0

    def read(self, size=-1):
        data = self._fobj.read(size)
        if data:
            self._uploaded += len(data)
            pct = (
                int(self._uploaded * 100 / self._file_size) if self._file_size else 100
            )
            self._emitter.upload_progress(self._file_size / _BYTES_PER_MIB, pct)
        return data

    def __len__(self):
        return self._file_size


def _upload_to_gcs(signed_url: str, file_path: str, file_size: int) -> None:
    """Upload tarball to GCS via signed PUT URL with progress display."""
    import urllib.error
    import urllib.request

    em = _get_emitter()

    with open(file_path, "rb") as f:
        reader = _ProgressReader(f, file_size, em)
        req = urllib.request.Request(
            signed_url,
            data=reader,
            method="PUT",
            headers={
                "Content-Type": "application/gzip",
                "Content-Length": str(file_size),
                "X-Goog-Content-Length-Range": "0,209715200",
            },
        )
        try:
            urllib.request.urlopen(req, timeout=_UPLOAD_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="ignore")
            raise click.ClickException(
                f"Upload failed with status {err.code}: {detail}"
            ) from None
    if not em.json_mode:
        click.echo()


# ---------------------------------------------------------------------------
# Build runners
# ---------------------------------------------------------------------------


def _resolve_pushed_image_digest(
    runner: CommandRunner,
    *,
    remote_image: str,
    docker_config_dir: str | None,
    verbose: bool,
) -> str:
    """Return ``{registry}/{repo}@sha256:<hex>`` for a freshly-pushed image.

    Reads ``RepoDigests`` via ``docker image inspect`` — the local daemon
    records the registry's manifest digest there after a successful push.
    Falls back to ``remote_image`` with a warning if no matching digest is
    found, rather than failing the deploy.
    """
    reference = ImageReference.parse(remote_image)
    stdout, _ = runner.run(
        subp_exec(
            *_docker_argv(docker_config_dir),
            "image",
            "inspect",
            "--format",
            "{{json .RepoDigests}}",
            remote_image,
            collect=True,
            verbose=verbose,
        )
    )
    try:
        digests = json_mod.loads(stdout or "[]") or []
    except json_mod.JSONDecodeError:
        digests = []
    for d in digests:
        if isinstance(d, str) and reference.matches_digest(d):
            return d
    _get_emitter().warn(
        f"Could not resolve image digest for {remote_image}; "
        "falling back to the tag-based reference. Re-run with --verbose for details."
    )
    return remote_image


@dataclass(frozen=True, slots=True)
class BuildSpec:
    config: pathlib.Path
    config_json: Config
    base_image: str | None
    api_version: str | None
    pull: bool
    docker_build_args: Sequence[str]
    install_command: str | None
    build_command: str | None


@dataclass(frozen=True, slots=True)
class DockerBuildCommand:
    command: tuple[str, ...]
    flags: tuple[str, ...]

    @classmethod
    def for_host(cls, machine: str, *, verbose: bool) -> "DockerBuildCommand":
        if machine == _NATIVE_AMD64_MACHINE:
            return cls(("docker", "build"), ())
        flags: tuple[str, ...] = ("--platform", _DEPLOYMENT_PLATFORM, "--load")
        if not verbose:
            flags += ("--progress=quiet",)
        return cls(("docker", "buildx", "build"), flags)


def _docker_argv(docker_config_dir: str | None) -> tuple[str, ...]:
    if docker_config_dir is None:
        return ("docker",)
    return ("docker", "--config", docker_config_dir)


def _build_image(
    runner: CommandRunner, spec: BuildSpec, tag: str, *, verbose: bool
) -> None:
    build = DockerBuildCommand.for_host(platform.machine(), verbose=verbose)
    with Progress(message="Building...", elapsed=not verbose):
        build_docker_image(
            runner,
            lambda _msg: None,
            spec.config,
            spec.config_json,
            spec.base_image,
            spec.api_version,
            spec.pull,
            tag,
            spec.docker_build_args,
            spec.install_command,
            spec.build_command,
            docker_command=build.command,
            extra_flags=build.flags,
            verbose=verbose,
        )


def _push_image(
    runner: CommandRunner,
    image: str,
    *,
    docker_config_dir: str | None,
    verbose: bool,
) -> None:
    for attempt in range(1, _PUSH_ATTEMPTS + 1):
        try:
            with Progress(message="Pushing...", elapsed=not verbose):
                runner.run(
                    subp_exec(
                        *_docker_argv(docker_config_dir), "push", image, verbose=verbose
                    )
                )
            return
        except click.exceptions.Exit:
            if attempt == _PUSH_ATTEMPTS:
                raise
            _get_emitter().warn(
                f"   Push failed, retrying (attempt {attempt + 1} of {_PUSH_ATTEMPTS})..."
            )


def _image_revision_result(resource: object, no_result_message: str) -> BuildResult:
    return BuildResult(
        updated=resource if isinstance(resource, dict) else {},
        progress_message="Deploying...",
        timeout_seconds=300,
        poll_interval_seconds=1,
        no_result_message=no_result_message,
    )


def _run_local_build(
    *,
    client: HostBackendClient,
    deployment_id: str,
    step: int,
    spec: BuildSpec,
    verbose: bool,
    image_name: str | None,
    prebuilt_image: str | None,
    name: str | None,
    tag: str,
    secrets: list[dict[str, str]],
    tracked_packages: list[str] | None,
) -> BuildResult:
    """Build locally with Docker, push to registry, update deployment."""
    local_tag = f"{_LOCAL_BUILD_TAG_PREFIX}:{int(time.time())}"
    image_to_push = prebuilt_image or local_tag

    with Runner() as runner:
        if prebuilt_image:
            _log_deploy_step(step, f"Validating image {prebuilt_image}")
            _validate_prebuilt_image(runner, prebuilt_image, verbose=verbose)
        else:
            _log_deploy_step(step, "Building image")
            _build_image(runner, spec, local_tag, verbose=verbose)
        step += 1

        _log_deploy_step(step, "Requesting push token")
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
        repo_seed = image_name or name or spec.config.parent.name
        remote_image = str(
            ImageReference(
                f"{normalized_registry}/{normalize_name(repo_seed)}",
                tag,
            )
        )
        registry_host = normalized_registry.split("/")[0]

        with _docker_config_for_token(registry_host, deployment_token) as cfg:
            _log_deploy_step(step, f"Logging into {registry_host}")
            token_input = (
                deployment_token
                if deployment_token.endswith("\n")
                else f"{deployment_token}\n"
            )
            runner.run(
                subp_exec(
                    *_docker_argv(cfg),
                    "login",
                    "-u",
                    "oauth2accesstoken",
                    "--password-stdin",
                    registry_host,
                    input=token_input,
                    verbose=False,
                )
            )
            step += 1

            _log_deploy_step(step, f"Pushing image {remote_image}")
            runner.run(
                subp_exec("docker", "tag", image_to_push, remote_image, verbose=verbose)
            )
            _push_image(runner, remote_image, docker_config_dir=cfg, verbose=verbose)
        step += 1

        resolved_image = _resolve_pushed_image_digest(
            runner,
            remote_image=remote_image,
            docker_config_dir=None,
            verbose=verbose,
        )

        _log_deploy_step(step, f"Updating deployment {deployment_id}")
        updated = client.update_deployment(
            deployment_id,
            resolved_image,
            revision_source="internal_docker",
            secrets=secrets,
            tracked_packages=tracked_packages,
        )

    return _image_revision_result(updated, "Deployment updated")


def _run_remote_build(
    *,
    client: HostBackendClient,
    deployment_id: str,
    step: int,
    spec: BuildSpec,
    verbose: bool,
    secrets: list[dict[str, str]],
    tracked_packages: list[str] | None,
) -> BuildResult:
    """Upload source tarball and trigger a remote build."""
    from langgraph_cli.archive import create_archive

    em = _get_emitter()
    _log_deploy_step(step, "Creating source archive")
    with create_archive(spec.config, spec.config_json) as (
        archive_path,
        file_size,
        config_rel,
    ):
        em.info(f"Archive created ({file_size / _BYTES_PER_MIB:.1f} MB)")
        step += 1

        _log_deploy_step(step, "Requesting upload URL")
        upload_data = client.request_upload_url(deployment_id)
        signed_url = upload_data.get("upload_url")
        object_path = upload_data.get("object_path")
        if not signed_url or not object_path:
            raise click.ClickException("Upload URL response missing required fields")
        step += 1

        _log_deploy_step(step, "Uploading source")
        _upload_to_gcs(signed_url, archive_path, file_size)
    step += 1

    _log_deploy_step(step, "Triggering remote build")
    updated = client.update_deployment_internal_source(
        deployment_id,
        source_tarball_path=object_path,
        config_path=config_rel,
        secrets=secrets,
        install_command=spec.install_command,
        build_command=spec.build_command,
        tracked_packages=tracked_packages,
    )

    log_offset: str | None = None
    logs_header_printed = False

    def _stream_build_logs(
        status: str, revision_id: str, set_progress: Callable[[str], None]
    ) -> None:
        nonlocal log_offset, logs_header_printed
        if not (verbose and status in ("AWAITING_BUILD", "BUILDING")):
            return
        try:
            logs_resp = client.get_build_logs(
                deployment_id,
                revision_id,
                {"order": "asc", "limit": 50, "offset": log_offset}
                if log_offset
                else {"order": "asc", "limit": 50},
            )
            if isinstance(logs_resp, dict):
                entries = logs_resp.get("logs", [])
                has_output = any(entry.get("message") for entry in entries)
                if has_output:
                    set_progress("")
                    if not logs_header_printed:
                        em.info(f"{status} (build logs):")
                        logs_header_printed = True
                for entry in entries:
                    msg = entry.get("message", "")
                    if msg:
                        em.log(msg)
                log_offset = logs_resp.get("next_offset") or log_offset
                if has_output:
                    set_progress(f"{status}...")
        except Exception:
            pass

    def _handle_interrupt(revision_id: str) -> None:
        em.warn(
            f"\nInterrupted. Deployment ID: {deployment_id}, Revision ID: {revision_id}"
        )
        em.warn("The build will continue remotely.")

    return BuildResult(
        updated=updated if isinstance(updated, dict) else {},
        progress_message="",
        timeout_seconds=900,
        poll_interval_seconds=3,
        no_result_message="Build triggered",
        on_poll=_stream_build_logs,
        on_interrupt=_handle_interrupt,
        show_build_logs_on_failure=True,
    )


# ---------------------------------------------------------------------------
# Deployment sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeployContext:
    client: HostBackendClient
    endpoints: ControlPlaneEndpoints
    spec: BuildSpec
    verbose: bool
    selector: DeploymentSelector
    deployment_type: str
    secrets: list[dict[str, str]]
    tracked_packages: list[str] | None


@dataclass(frozen=True, slots=True)
class DeployOutcome:
    deployment_id: str
    build_result: BuildResult


class DeploymentSource(Protocol):
    def run(self, ctx: DeployContext) -> DeployOutcome: ...


def _resolve_or_create(
    ctx: DeployContext, *, source: SourceName, not_found_message: str
) -> tuple[str, int]:
    if isinstance(ctx.selector, ById):
        existing, step = _fetch_deployment(ctx.client, 1, ctx.selector)
        return existing.id, step
    found, step = _find_deployment(
        ctx.client, 1, ctx.selector, not_found_message=not_found_message
    )
    if found is not None:
        return found.id, step
    try:
        created, step = _create_deployment(
            ctx.client,
            step,
            name=ctx.selector.name if isinstance(ctx.selector, ByName) else None,
            agent=asdict(ctx.selector) if isinstance(ctx.selector, ByAgent) else None,
            source=source,
            source_config={"deployment_type": ctx.deployment_type},
            source_revision_config={},
            secrets=ctx.secrets,
        )
    except HostBackendError as err:
        if _needs_a_listener(err):
            raise ListenerRequiredError(
                "The image has to come from a registry you manage, so re-run with "
                "--push-to <registry>/<repository>."
            ) from None
        raise
    return created.id, step


class ListenerRequiredError(click.UsageError):
    def __init__(self, remedy: str) -> None:
        super().__init__(
            "This workspace deploys through a listener in your own cluster. "
            f"{remedy}\nLearn about listeners: {_LISTENERS_DOCS_URL}"
        )


def _needs_a_listener(err: HostBackendError) -> bool:
    return err.status_code == 400 and _LISTENER_REQUIRED_MARKER in (
        err.detail or err.message
    )


def _requested_listener(client: HostBackendClient, listener_id: str) -> Listener:
    try:
        resource = _call_host_backend_with_optional_tenant(
            client, lambda c: c.get_listener(listener_id)
        )
    except HostBackendError as err:
        if err.status_code not in _LISTENER_NOT_FOUND_STATUSES:
            raise
        available = _available_listeners(client)
        if not available:
            raise click.UsageError(_NO_LISTENERS) from None
        raise click.UsageError(
            f"Listener {listener_id} was not found in this workspace. "
            f"Available listeners:\n{_describe_listeners(available)}"
        ) from None
    return Listener.from_resource(resource)


def _available_listeners(client: HostBackendClient) -> tuple[Listener, ...]:
    resources = _call_host_backend_with_optional_tenant(
        client, lambda c: c.list_listeners()
    )
    return tuple(Listener.from_resource(resource) for resource in resources)


def _ensure_customer_registry_source(existing: ExistingDeployment) -> None:
    if existing.source != _CUSTOMER_REGISTRY_SOURCE:
        raise click.UsageError(
            f"Deployment {existing.id} was not created from an external image "
            "and cannot be updated with --push-to. Run without --push-to to keep "
            "its current build mode, or use a different --name to create a new "
            "deployment."
        )


@dataclass(frozen=True, slots=True)
class ManagedRegistrySource:
    prebuilt_image: str | None
    image_name: str | None
    tag: str

    def run(self, ctx: DeployContext) -> DeployOutcome:
        deployment_id, step = _resolve_or_create(
            ctx,
            source="internal_docker",
            not_found_message="No deployment found. Will create after build.",
        )
        build_result = _run_local_build(
            client=ctx.client,
            deployment_id=deployment_id,
            step=step,
            spec=ctx.spec,
            verbose=ctx.verbose,
            image_name=self.image_name,
            prebuilt_image=self.prebuilt_image,
            name=ctx.selector.name if isinstance(ctx.selector, ByName) else None,
            tag=self.tag,
            secrets=ctx.secrets,
            tracked_packages=ctx.tracked_packages,
        )
        return DeployOutcome(deployment_id, build_result)


@dataclass(frozen=True, slots=True)
class RemoteBuildSource:
    def run(self, ctx: DeployContext) -> DeployOutcome:
        deployment_id, step = _resolve_or_create(
            ctx,
            source="internal_source",
            not_found_message="No deployment found. Will create.",
        )
        build_result = _run_remote_build(
            client=ctx.client,
            deployment_id=deployment_id,
            step=step,
            spec=ctx.spec,
            verbose=ctx.verbose,
            secrets=ctx.secrets,
            tracked_packages=ctx.tracked_packages,
        )
        return DeployOutcome(deployment_id, build_result)


@dataclass(frozen=True, slots=True)
class CustomerRegistrySource:
    reference: ImageReference
    prebuilt_image: str | None
    requested_placement: RequestedPlacement

    def run(self, ctx: DeployContext) -> DeployOutcome:
        if isinstance(ctx.selector, ById):
            existing, step = _fetch_deployment(ctx.client, 1, ctx.selector)
            return self._update(ctx, existing, step)
        found, step = _find_deployment(
            ctx.client,
            1,
            ctx.selector,
            not_found_message="No deployment found. Will create after push.",
        )
        if found is not None:
            return self._update(ctx, found, step)
        return self._create(
            ctx, ctx.selector.name if isinstance(ctx.selector, ByName) else None, step
        )

    def _update(
        self, ctx: DeployContext, existing: ExistingDeployment, step: int
    ) -> DeployOutcome:
        _ensure_customer_registry_source(existing)
        self.requested_placement.ensure_not_requested(existing.id)
        image_uri, step = self._publish(ctx, step)
        _log_deploy_step(step, f"Updating deployment {existing.id}")
        updated = ctx.client.update_deployment(
            existing.id,
            image_uri,
            revision_source=None,
            secrets=ctx.secrets,
            tracked_packages=ctx.tracked_packages,
        )
        return DeployOutcome(
            existing.id, _image_revision_result(updated, "Deployment updated")
        )

    def _resolve_placement(self, ctx: DeployContext) -> Placement:
        requested = self.requested_placement
        if requested.listener_id is not None:
            return requested.on(_requested_listener(ctx.client, requested.listener_id))
        if not (ctx.endpoints.is_cloud or requested.requested):
            return Unplaced()
        return requested.among(_available_listeners(ctx.client))

    def _announce(self, placement: Placement) -> None:
        if isinstance(placement, OnListener):
            _get_emitter().info(
                placement.summary,
                listener_id=placement.listener_id,
                k8s_namespace=placement.k8s_namespace,
            )

    def _create(self, ctx: DeployContext, name: str | None, step: int) -> DeployOutcome:
        placement = self._resolve_placement(ctx)
        self._announce(placement)
        image_uri, step = self._publish(ctx, step)
        try:
            created, _ = _create_deployment(
                ctx.client,
                step,
                name=name,
                agent=asdict(ctx.selector)
                if isinstance(ctx.selector, ByAgent)
                else None,
                source=_CUSTOMER_REGISTRY_SOURCE,
                source_config={
                    "resource_spec": _OPERATOR_DEFAULT_RESOURCE_SPEC,
                    **placement.source_config(),
                },
                source_revision_config={"image_uri": image_uri},
                secrets=ctx.secrets,
            )
        except HostBackendError as err:
            if _needs_a_listener(err):
                raise ListenerRequiredError(
                    "Re-run with --listener-id and --k8s-namespace.\n"
                    f"{err.detail or err.message}"
                ) from None
            raise
        return DeployOutcome(
            created.id, _image_revision_result(created.resource, "Deployment created")
        )

    def _publish(self, ctx: DeployContext, step: int) -> tuple[str, int]:
        image = str(self.reference)
        with Runner() as runner:
            if self.prebuilt_image:
                _log_deploy_step(step, f"Validating image {self.prebuilt_image}")
                _validate_prebuilt_image(
                    runner, self.prebuilt_image, verbose=ctx.verbose
                )
                runner.run(
                    subp_exec(
                        "docker", "tag", self.prebuilt_image, image, verbose=ctx.verbose
                    )
                )
            else:
                _log_deploy_step(step, f"Building image {image}")
                _build_image(runner, ctx.spec, image, verbose=ctx.verbose)
            step += 1
            _log_deploy_step(step, f"Pushing image {image}")
            _push_image(runner, image, docker_config_dir=None, verbose=ctx.verbose)
            step += 1
            digest = _resolve_pushed_image_digest(
                runner, remote_image=image, docker_config_dir=None, verbose=ctx.verbose
            )
        return digest, step


def _require_local_docker() -> None:
    supported, error = can_build_locally()
    if not supported:
        raise click.UsageError(error or "Unable to build locally.")


def _push_reference(push_to: str, tag: str | None) -> ImageReference:
    try:
        reference = ImageReference.parse(push_to)
    except ValueError:
        raise click.UsageError(
            "--push-to takes a repository with an optional tag, not a digest."
        ) from None
    if reference.tag is not None and tag is not None:
        raise click.UsageError(
            "--push-to already includes a tag; do not combine it with --tag."
        )
    if reference.tag is not None:
        return reference
    return reference.with_tag(normalize_image_tag(tag or _DEFAULT_IMAGE_TAG))


def _select_source(
    *,
    push_to: str | None,
    image: str | None,
    image_name: str | None,
    tag: str | None,
    remote_build_flag: bool | None,
    placement: RequestedPlacement,
    selector: DeploymentSelector,
) -> DeploymentSource:
    if push_to is None and placement.requested:
        raise click.UsageError(
            "--listener-id and --k8s-namespace only apply when creating a "
            "deployment with --push-to."
        )
    if placement.requested and isinstance(selector, ById):
        raise click.UsageError(
            "Listener and namespace are fixed when a deployment is created, so "
            "they cannot be set for an existing --deployment-id. Drop them, or "
            "create a new deployment with --name."
        )
    if push_to is not None:
        if remote_build_flag is True:
            raise click.UsageError("--push-to cannot be combined with --remote.")
        reference = _push_reference(push_to, tag)
        if image is None:
            _require_local_docker()
        return CustomerRegistrySource(
            reference=reference,
            prebuilt_image=image,
            requested_placement=placement,
        )
    if image and remote_build_flag is True:
        raise click.UsageError("--image cannot be combined with --remote builds.")
    use_remote_build, local_build_error = _resolve_build_mode(
        remote_build_flag, force_local=image is not None
    )
    if not use_remote_build:
        return ManagedRegistrySource(
            prebuilt_image=image,
            image_name=image_name,
            tag=normalize_image_tag(tag or _DEFAULT_IMAGE_TAG),
        )
    if remote_build_flag is None and local_build_error:
        em = _get_emitter()
        em.note(f"{local_build_error}\nUsing remote build instead.")
        if not em.json_mode:
            click.echo()
    return RemoteBuildSource()


# ---------------------------------------------------------------------------
# Host backend client factory
# ---------------------------------------------------------------------------


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
        if _no_input:
            raise click.ClickException(
                "No LangSmith API key found. Set LANGSMITH_API_KEY in the "
                "environment or .env file."
            )
        click.secho(
            "No LangSmith API key found. Create one at Settings > API Keys in LangSmith.",
            fg="yellow",
        )
        resolved_api_key = click.prompt("Enter LangSmith API key", hide_input=True)
    tenant_id = env_vars.get("LANGSMITH_TENANT_ID") or os.environ.get(
        "LANGSMITH_TENANT_ID"
    )
    langsmith_endpoint = env_vars.get("LANGSMITH_ENDPOINT") or os.environ.get(
        "LANGSMITH_ENDPOINT"
    )
    endpoints = ControlPlaneEndpoints.resolve(host_url, langsmith_endpoint)
    return HostBackendClient(
        endpoints.control_plane_url, resolved_api_key, tenant_id=tenant_id
    )


def _call_host_backend_with_optional_tenant(
    client: HostBackendClient,
    operation: Callable[[HostBackendClient], _T],
) -> _T:
    """Run *operation*, prompting for a workspace ID on org-scoped 403s.

    On success the original *client* is returned as-is.  If the user is
    prompted for a workspace ID, the tenant header is set on *client*
    in-place so all subsequent calls through the same instance are
    tenant-aware.
    """
    prompted_for_tenant = False

    while True:
        try:
            return operation(client)
        except HostBackendError as err:
            if (
                not prompted_for_tenant
                and err.status_code == 403
                and "requires workspace specification" in err.message
            ):
                if _no_input:
                    raise click.ClickException(
                        "API key is org-scoped and requires a workspace ID. "
                        "Set LANGSMITH_TENANT_ID in your .env file or "
                        "use a workspace-scoped API key."
                    ) from None
                click.secho(
                    "Your API key is org-scoped and requires a workspace ID.",
                    fg="yellow",
                )
                click.secho(
                    "Find your workspace ID in LangSmith under Settings > Workspaces.",
                    fg="yellow",
                )
                client.set_tenant(click.prompt("Workspace ID"))
                prompted_for_tenant = True
                continue
            if err.status_code == 403 and "not enabled" in err.message.lower():
                smith_base = client.endpoints.dashboard_url
                raise HostBackendError(
                    "LangSmith Deployment is not enabled for this organization. "
                    f"Enable it at {smith_base}/host/deployments"
                    " (ensure this matches the organization for your API key).",
                    status_code=403,
                ) from None
            raise


# ---------------------------------------------------------------------------
# Click options shared by deploy commands
# ---------------------------------------------------------------------------

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
    default=None,
    hidden=True,
)

OPT_AGENT_ID = click.option(
    "--agent-id",
    envvar="LANGSMITH_AGENT_ID",
    show_envvar=True,
    help="Logical agent ID (requires agent mode enabled for the tenant).",
)

OPT_AGENT_ENVIRONMENT = click.option(
    "--agent-environment",
    "environment",
    envvar="LANGSMITH_AGENT_ENVIRONMENT",
    show_envvar=True,
    type=click.Choice(["development", "staging", "production"]),
    help="Agent environment (requires agent mode enabled for the tenant).",
)

OPT_VERBOSE = click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show more output from the server logs",
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
        """Treat leading option-like subcommand tokens as passthrough args.

        Click stores the unresolved nested command token on the context after
        `Group.parse_args()` runs, but the backing attribute changed across
        supported Click versions. Click 8.1.x stores the value directly on
        `protected_args`, while Click 8.2+ stores it on `_protected_args`
        and exposes `protected_args` as a deprecated compatibility property.
        Since this package allows `click>=8.1.7`, we need to check both
        names to support the full version range without relying on one
        version-specific internal detail.
        """
        result = super().parse_args(ctx, args)
        protected_args = ctx.__dict__.get("protected_args")
        if protected_args is None:
            protected_args = ctx.__dict__.get("_protected_args", [])
        if protected_args and protected_args[0].startswith("-"):
            ctx.args = [*protected_args, *ctx.args]
            if "protected_args" in ctx.__dict__:
                ctx.protected_args = []
            elif "_protected_args" in ctx.__dict__:
                ctx._protected_args = []
            return ctx.args
        return result


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
            OPT_AGENT_ID,
            OPT_AGENT_ENVIRONMENT,
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
                help=(
                    "Deployment type (used when creating a new deployment). "
                    "Ignored with --push-to."
                ),
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
                default=None,
                help="Tag to use for the pushed deployment image. [default: latest]",
            ),
            click.option(
                "--image",
                help=(
                    "Use an existing local image reference (e.g. repo:tag) and "
                    "skip building. The image must target linux/amd64."
                ),
            ),
            click.option(
                "--push-to",
                help=(
                    "Push the image to this repository in a registry you manage, "
                    "then deploy it from there. For self-hosted and hybrid "
                    "LangSmith. Uses your existing Docker credentials. Builds the "
                    "project, or retags the local image given with --image. "
                    "Give the tag here or with --tag (default: latest)."
                ),
            ),
            click.option(
                "--listener-id",
                help=(
                    "Listener that will run the deployment, for workspaces that "
                    "deploy through a listener in your own cluster. Only used when "
                    "creating a deployment with --push-to."
                ),
            ),
            click.option(
                "--k8s-namespace",
                help=(
                    "Kubernetes namespace the listener deploys into. Only used when "
                    "creating a deployment with --push-to."
                ),
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
            click.option(
                "--remote/--no-remote",
                "remote_build_flag",
                default=None,
                help=(
                    "Force remote or local build. By default, builds remotely "
                    "if Docker is not available locally."
                ),
            ),
            click.option(
                "--json",
                "json_output",
                is_flag=True,
                default=False,
                help="Emit structured JSON-lines to stdout instead of human-readable text.",
            ),
            click.option(
                "--no-input",
                is_flag=True,
                default=False,
                help="Never prompt for input; fail with an error if a required value is missing.",
            ),
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


# ---------------------------------------------------------------------------
# Deploy CLI group and commands
# ---------------------------------------------------------------------------


@click.group(
    cls=DeployGroup,
    help=(
        "[Beta] Build and deploy a LangGraph image to LangSmith Deployment.\n\n"
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
    # if we detect no subcommand, we run _deploy_cmd (basically run langgraph deploy as a top level command)
    # otherwise, we return None here and click will proceed to actually run the subcommand (list or delete)
    if ctx.invoked_subcommand is not None:
        return
    if (
        ctx.params.get("agent_id") is not None
        or ctx.params.get("environment") is not None
    ) and ctx.get_parameter_source("name") == click.core.ParameterSource.ENVIRONMENT:
        # Ignore the inherited name default so it does not conflict with agent mode.
        ctx.params["name"] = None
    docker_build_args = tuple(ctx.args)
    ctx.args = []  # Prevent Click from re-processing passthrough args later.
    return ctx.forward(_deploy_cmd, docker_build_args=docker_build_args)


@_deploy_base_options()
@click.command(context_settings=dict(ignore_unknown_options=True))
def _deploy_cmd(
    config: pathlib.Path,
    pull: bool,
    verbose: bool,
    api_version: str | None,
    host_url: str | None,
    api_key: str | None,
    deployment_id: str | None,
    deployment_type: str,
    name: str | None,
    agent_id: str | None,
    environment: str | None,
    image_name: str | None,
    image: str | None,
    push_to: str | None,
    listener_id: str | None,
    k8s_namespace: str | None,
    tag: str | None,
    base_image: str | None,
    install_command: str | None,
    build_command: str | None,
    no_wait: bool,
    remote_build_flag: bool | None,
    docker_build_args: Sequence[str],
    json_output: bool,
    no_input: bool,
):
    global _emitter, _no_input
    _emitter = _Emitter(json_mode=json_output)
    _no_input = no_input
    em = _emitter

    em.note(
        "Note: 'langgraph deploy' is in beta. Expect frequent updates and improvements."
    )
    if not json_output:
        click.echo()

    validate_deploy_commands(install_command, build_command)
    agent = None
    if agent_id is not None or environment is not None:
        em.note("Note: --agent-id and --agent-environment flags are in private beta")
        if not agent_id or not agent_id.strip() or not environment:
            raise click.UsageError(
                "--agent-id and --agent-environment are required together."
            )
        if name is not None or deployment_id is not None:
            raise click.UsageError(
                "--agent-id and --agent-environment cannot be combined with --name or --deployment-id."
            )
        agent = {"agent_id": agent_id, "environment": environment}
    if not config.exists():
        message = (
            "We couldn't find a langgraph.json file. Run `langgraph deploy` from "
            "the root of a LangSmith Deployment project. To get started, visit "
            "https://docs.langchain.com/langsmith/deployment-quickstart."
        )
        if json_output:
            em.error(message)
            raise click.exceptions.Exit(1)
        raise click.ClickException(message)
    config_json = langgraph_cli.config.validate_config_file(config)
    warn_non_wolfi_distro(config_json, emit=em.note)

    env_vars = _parse_env_from_config(config_json, config)

    if not agent and not deployment_id and not name:
        name = env_vars.get(_DEPLOYMENT_NAME_ENV)
    if not agent and not deployment_id and not name:
        default_name = normalize_name(pathlib.Path.cwd().name)
        if no_input:
            name = default_name
        else:
            name = click.prompt("Deployment name", default=default_name)
    if name and not deployment_id:
        name = normalize_name(name)
        if not no_input:
            env_path = _resolve_env_path(config_json, config)
            if env_path is not None:
                set_key(str(env_path), _DEPLOYMENT_NAME_ENV, name)
                em.info(f"Saved deployment name to {env_path}")

    secrets = _secrets_from_env(_env_without_deployment_name(env_vars))

    selector = ByAgent(**agent) if agent else deployment_selector(deployment_id, name)
    source = _select_source(
        push_to=push_to,
        image=image,
        image_name=image_name,
        tag=tag,
        remote_build_flag=remote_build_flag,
        placement=RequestedPlacement(listener_id, k8s_namespace),
        selector=selector,
    )

    client = _create_host_backend_client(host_url, api_key, env_vars=env_vars)
    try:
        tracked_packages = find_tracked_packages(config, config_json) or None
    except Exception as exc:
        em.warn(f"Skipped tracked-package scan: {exc}")
        tracked_packages = None

    outcome = source.run(
        DeployContext(
            client=client,
            endpoints=client.endpoints,
            spec=BuildSpec(
                config=config,
                config_json=config_json,
                base_image=base_image,
                api_version=api_version,
                pull=pull,
                docker_build_args=docker_build_args,
                install_command=install_command,
                build_command=build_command,
            ),
            verbose=verbose,
            selector=selector,
            deployment_type=deployment_type,
            secrets=secrets,
            tracked_packages=tracked_packages,
        )
    )
    dep_status_url = _emit_deployment_status_url(
        outcome.build_result.updated,
        outcome.deployment_id,
        client.endpoints,
    )

    if no_wait:
        em.info(outcome.build_result.no_result_message)
        return

    last_status, revision_id = _poll_revision_status(
        client,
        outcome.deployment_id,
        progress_message=outcome.build_result.progress_message,
        timeout_seconds=outcome.build_result.timeout_seconds,
        poll_interval_seconds=outcome.build_result.poll_interval_seconds,
        on_poll=outcome.build_result.on_poll,
        on_interrupt=outcome.build_result.on_interrupt,
    )
    if not last_status:
        em.info(outcome.build_result.no_result_message)
        return

    if (
        outcome.build_result.show_build_logs_on_failure
        and last_status == "BUILD_FAILED"
        and not verbose
        and revision_id is not None
    ):
        em.error("Last build log lines:")
        try:
            logs_resp = client.get_build_logs(
                outcome.deployment_id,
                revision_id,
                {"order": "desc", "limit": 30},
            )
            if isinstance(logs_resp, dict):
                entries = list(reversed(logs_resp.get("logs", [])))
                for entry in entries:
                    msg = entry.get("message", "")
                    if msg:
                        em.log(msg)
        except Exception:
            em.error("(failed to fetch build logs)")
        em.warn("Re-run with --verbose to see full build output.")

    _print_deployment_result(
        client,
        outcome.deployment_id,
        last_status,
        dashboard_label="Deployment dashboard",
        status_url=dep_status_url,
    )


# ---------------------------------------------------------------------------
# deploy list
# ---------------------------------------------------------------------------


@OPT_HOST_API_KEY
@OPT_HOST_URL
@OPT_AGENT_ID
@OPT_AGENT_ENVIRONMENT
@click.option(
    "--name-contains",
    default="",
    help="Only show deployments whose names contain this value.",
)
@deploy.command("list", help="[Beta] List LangSmith Deployments.")
def deploy_list(
    api_key: str | None,
    host_url: str | None,
    name_contains: str,
    agent_id: str | None,
    environment: str | None,
) -> None:
    if agent_id is not None or environment is not None:
        click.secho(
            "Note: --agent-id and --agent-environment flags are in private beta",
            fg="yellow",
        )
    if agent_id is not None and not agent_id.strip():
        raise click.UsageError("--agent-id must not be empty.")
    filters = {}
    if agent_id is not None:
        filters["agent_id"] = agent_id
    if environment is not None:
        filters["agent_environment"] = environment
    client = _create_host_backend_client(host_url, api_key)
    deployments = _call_host_backend_with_optional_tenant(
        client,
        lambda c: c.list_deployments(name_contains=name_contains, **filters),
    )
    if not deployments:
        click.echo("No deployments found.")
        return
    click.echo(format_deployments_table(deployments))


# ---------------------------------------------------------------------------
# deploy revisions
# ---------------------------------------------------------------------------


@deploy.group(
    "revisions", cls=NestedHelpGroup, help="[Beta] Manage deployment revisions."
)
def deploy_revisions() -> None:
    pass


@OPT_HOST_API_KEY
@OPT_HOST_URL
@click.option(
    "--limit",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of revisions to return.",
)
@click.argument("deployment_id")
@deploy_revisions.command(
    "list",
    help=(
        "[Beta] List revisions for a LangSmith Deployment.\n\n"
        "Use the `deploy list` command to list deployment IDs."
    ),
)
def deploy_revisions_list(
    api_key: str | None, host_url: str | None, limit: int, deployment_id: str
) -> None:
    client = _create_host_backend_client(host_url, api_key)
    revisions = _call_host_backend_with_optional_tenant(
        client,
        lambda c: c.list_revisions(deployment_id, limit=limit),
    )
    if not revisions:
        click.echo(f"No revisions found for deployment {deployment_id}.")
        return
    click.echo(format_revisions_table(revisions))


# ---------------------------------------------------------------------------
# deploy delete
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# deploy logs
# ---------------------------------------------------------------------------


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
    host_url: str | None,
):
    env_vars = _parse_env_from_config({}, pathlib.Path.cwd() / DEFAULT_CONFIG)
    client = _create_host_backend_client(host_url, api_key, env_vars=env_vars)
    if not deployment_id and not name:
        name = env_vars.get(_DEPLOYMENT_NAME_ENV)
    selector = deployment_selector(deployment_id, name)
    if isinstance(selector, ById):
        dep_id = selector.deployment_id
    else:
        name_to_find = selector.name
        found = _call_host_backend_with_optional_tenant(
            client, lambda c: find_deployment_by_name(c, name_to_find)
        )
        if found is None:
            raise click.ClickException(f"Deployment '{name_to_find}' not found.")
        dep_id = found.id

    if log_type == "build" and not revision_id:
        revisions = client.list_revisions(dep_id, limit=1)
        if not revisions:
            raise click.ClickException(
                "No revisions found for this deployment. Cannot fetch build logs."
            )
        revision_id = str(revisions[0]["id"])
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
