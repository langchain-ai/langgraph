"""Deploy command and subcommands for the LangGraph CLI."""

import base64
import json as json_mod
import os
import pathlib
import platform
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

import click
import click.exceptions
from dotenv import dotenv_values, set_key

import langgraph_cli.config
from langgraph_cli.analytics import log_command
from langgraph_cli.constants import DEFAULT_CONFIG
from langgraph_cli.docker import build_docker_image, can_build_locally
from langgraph_cli.exec import Runner, subp_exec
from langgraph_cli.host_backend import HostBackendClient, HostBackendError
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

_DEPLOYMENT_NAME_ENV = "LANGSMITH_DEPLOYMENT_NAME"

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


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_deployment_selector(deployment_id: str | None, name: str | None) -> None:
    """Ensure either deployment_id or name is provided."""
    if deployment_id:
        return
    if not name:
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


def find_deployment_id_by_name(
    client: HostBackendClient, name: str | None
) -> str | None:
    """Return deployment ID for an exact name match, or None if not found."""
    if not name:
        return None
    existing = client.list_deployments(name_contains=name)
    if isinstance(existing, dict):
        for dep in existing.get("resources", []):
            if isinstance(dep, dict) and dep.get("name") == name:
                found_id = dep.get("id")
                if found_id:
                    return str(found_id)
    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def normalize_image_name(value: str | None) -> str:
    """Sanitize a deployment/directory name into a valid Docker repository name.

    Docker repository names must be lowercase and may only contain
    [a-z0-9._-].  Invalid characters are replaced with hyphens.
    """
    if not value:
        return "app"
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-.")
    return slug or "app"


def normalize_image_tag(value: str) -> str:
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
            click.secho(
                f"Warning: env file '{env_field}' specified in langgraph.json not found.",
                fg="yellow",
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
            click.secho(f"   Skipping reserved env var: {name}", fg="yellow")
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
) -> tuple[bool, str | None]:
    """Determine whether to use a remote build.

    Returns (use_remote_build, local_build_error).  Raises UsageError when
    --no-remote is set but the machine cannot build locally.
    """
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


def _log_deploy_step(step: int, message: str) -> None:
    click.secho(f"{step}. {message}", fg="cyan")


def _resolve_deployment(
    client: HostBackendClient,
    step: int,
    deployment_id: str | None,
    name: str | None,
    *,
    not_found_message: str,
) -> tuple[str | None, bool, int]:
    """Resolve an existing deployment by ID or exact name match."""
    needs_creation = False
    if deployment_id:
        _log_deploy_step(step, f"Using deployment {deployment_id}")
        _call_host_backend_with_optional_tenant(
            client, lambda c: c.get_deployment(deployment_id)
        )
        return deployment_id, needs_creation, step + 1

    _log_deploy_step(step, f"Looking up deployment '{name}'")
    found_id = _call_host_backend_with_optional_tenant(
        client, lambda c: find_deployment_id_by_name(c, name)
    )
    if found_id:
        deployment_id = str(found_id)
        click.secho(f"   Found existing deployment (ID: {deployment_id})", fg="green")
    else:
        needs_creation = True
        click.secho(not_found_message, fg="yellow")
    return deployment_id, needs_creation, step + 1


def _create_deployment(
    client: HostBackendClient,
    step: int,
    *,
    name: str,
    deployment_type: str,
    source: str,
    config_rel: str | None = None,
    secrets: list[dict[str, str]] | None = None,
) -> tuple[str, int]:
    """Create a deployment and return its ID and next step number."""
    _log_deploy_step(step, f"Creating deployment '{name}'")
    created = client.create_deployment(
        name=name,
        deployment_type=deployment_type,
        source=source,
        config_path=config_rel,
        secrets=secrets,
    )
    created_id = created.get("id") if isinstance(created, dict) else None
    if not isinstance(created_id, str) or not created_id:
        raise HostBackendError(
            "POST /v2/deployments succeeded but response missing a valid 'id'"
        )
    click.secho(f"   Deployment ID: {created_id}", fg="green")
    return created_id, step + 1


def _smith_dashboard_base_url(host_url: str | None) -> str:
    """Derive the LangSmith dashboard base URL from the API host URL."""
    from urllib.parse import urlparse

    if not host_url:
        return "https://smith.langchain.com"
    parsed = urlparse(host_url)
    hostname = parsed.hostname or ""
    if hostname in ("localhost", "127.0.0.1"):
        return host_url.rstrip("/")
    if hostname.startswith("eu."):
        return "https://eu.smith.langchain.com"
    return "https://smith.langchain.com"


def _print_deployment_status_url(
    updated: object, deployment_id: str, host_url: str | None = None
) -> None:
    """Print the deployment status URL when tenant metadata is available."""
    tenant_id = updated.get("tenant_id") if isinstance(updated, dict) else None
    if not tenant_id:
        return
    base = _smith_dashboard_base_url(host_url)
    status_url = f"{base}/o/{tenant_id}/host/deployments/{deployment_id}"
    click.secho(f"   View status: {status_url}", fg="cyan")


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
    revisions_resp = client.list_revisions(deployment_id, limit=1)
    resources = (
        revisions_resp.get("resources", []) if isinstance(revisions_resp, dict) else []
    )
    if not resources:
        return "", None

    revision_id = str(resources[0]["id"])
    last_status = ""
    deadline = time.time() + timeout_seconds
    start_time = time.monotonic()
    with Progress(message=progress_message, elapsed=True) as set_progress:
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
                    elapsed = time.monotonic() - start_time
                    mins, secs = divmod(int(elapsed), 60)
                    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
                    click.echo(f"   {last_status}... ({elapsed_str})")
                last_status = status
                if status in _TERMINAL_STATUSES:
                    break
                set_progress(f"{status}...")

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
) -> None:
    """Print final deployment status and raise on failure."""
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
            click.secho(f"   Check status at: {custom_url}", fg="yellow")
        else:
            click.secho(
                f"   Check status in the LangSmith {dashboard_label}.",
                fg="yellow",
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


class _ProgressReader:
    """File-like wrapper that displays upload progress via click."""

    def __init__(self, fobj, file_size: int):
        self._fobj = fobj
        self._file_size = file_size
        self._uploaded = 0

    def read(self, size=-1):
        data = self._fobj.read(size)
        if data:
            self._uploaded += len(data)
            pct = (
                int(self._uploaded * 100 / self._file_size) if self._file_size else 100
            )
            click.echo(
                f"\r   Uploading ({self._file_size / 1_048_576:.1f} MB)... {pct}%",
                nl=False,
            )
        return data

    def __len__(self):
        return self._file_size


def _upload_to_gcs(signed_url: str, file_path: str, file_size: int) -> None:
    """Upload tarball to GCS via signed PUT URL with progress display."""
    import urllib.error
    import urllib.request

    with open(file_path, "rb") as f:
        req = urllib.request.Request(
            signed_url,
            data=_ProgressReader(f, file_size),
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
    click.echo()


# ---------------------------------------------------------------------------
# Build runners
# ---------------------------------------------------------------------------


def _run_local_build(
    *,
    client: HostBackendClient,
    deployment_id: str,
    step: int,
    config: pathlib.Path,
    config_json: dict,
    verbose: bool,
    pull: bool,
    api_version: str | None,
    base_image: str | None,
    image_name: str | None,
    name: str | None,
    tag: str,
    install_command: str | None,
    build_command: str | None,
    docker_build_args: Sequence[str],
    secrets: list[dict[str, str]],
) -> BuildResult:
    """Build locally with Docker, push to registry, update deployment."""
    # Use buildx to cross-compile for amd64 when running on a non-x86_64 host
    # (e.g. Apple Silicon). On amd64 hosts, plain docker build is sufficient.
    needs_buildx = platform.machine() != "x86_64"
    local_tag = f"langgraph-deploy-tmp:{int(time.time())}"

    with Runner() as runner:
        # -- Step: Build image --
        _log_deploy_step(step, "Building image")
        if needs_buildx:
            build_flags: list[str] = [
                "--platform",
                "linux/amd64",
                "--load",
            ]
            if not verbose:
                build_flags.append("--progress=quiet")
            with Progress(message="Building...", elapsed=not verbose):
                build_docker_image(
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
                build_docker_image(
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

        # -- Step: Get push token and authenticate --
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
        repo_seed = image_name or name or config.parent.name
        repo_name = normalize_image_name(repo_seed)
        tag_value = normalize_image_tag(tag)
        remote_image = f"{normalized_registry}/{repo_name}:{tag_value}"

        registry_host = normalized_registry.split("/")[0]

        # Use a clean Docker config with only the push token so that
        # system credential helpers (e.g. gcloud) don't interfere.
        with _docker_config_for_token(registry_host, deployment_token) as cfg:
            _log_deploy_step(step, f"Logging into {registry_host}")
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
            _log_deploy_step(step, f"Pushing image {remote_image}")
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
        _log_deploy_step(step, f"Updating deployment {deployment_id}")
        updated = client.update_deployment(deployment_id, remote_image, secrets=secrets)

    return BuildResult(
        updated=updated if isinstance(updated, dict) else {},
        progress_message="Deploying...",
        timeout_seconds=300,
        poll_interval_seconds=1,
        no_result_message="Deployment updated",
    )


def _run_remote_build(
    *,
    client: HostBackendClient,
    deployment_id: str,
    step: int,
    config: pathlib.Path,
    config_json: dict,
    verbose: bool,
    install_command: str | None,
    build_command: str | None,
    secrets: list[dict[str, str]],
) -> BuildResult:
    """Upload source tarball and trigger a remote build."""
    from langgraph_cli.archive import create_archive

    _log_deploy_step(step, "Creating source archive")
    with create_archive(config, config_json) as (archive_path, file_size, config_rel):
        click.secho(f"   Archive created ({file_size / 1_048_576:.1f} MB)", fg="green")
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
        install_command=install_command,
        build_command=build_command,
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
                        click.echo(f"   {status} (build logs):")
                        logs_header_printed = True
                for entry in entries:
                    msg = entry.get("message", "")
                    if msg:
                        click.echo(f"   | {msg}")
                log_offset = logs_resp.get("next_offset") or log_offset
                if has_output:
                    set_progress(f"{status}...")
        except Exception:
            pass

    def _handle_interrupt(revision_id: str) -> None:
        click.secho(
            f"\n   Interrupted. Deployment ID: {deployment_id}, Revision ID: {revision_id}",
            fg="yellow",
        )
        click.secho("   The build will continue remotely.", fg="yellow")

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
                click.secho(
                    "Your API key is org-scoped and requires a workspace ID.",
                    fg="yellow",
                )
                click.secho(
                    "Find your workspace ID in LangSmith under Settings > Workspaces.",
                    fg="yellow",
                )
                client._client.headers["X-Tenant-ID"] = click.prompt("Workspace ID")
                prompted_for_tenant = True
                continue
            if err.status_code == 403 and "not enabled" in err.message.lower():
                smith_base = _smith_dashboard_base_url(client._base_url)
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
    default="https://api.host.langchain.com",
    hidden=True,
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
            click.option(
                "--remote/--no-remote",
                "remote_build_flag",
                default=None,
                help=(
                    "Force remote or local build. By default, builds remotely "
                    "if Docker is not available locally."
                ),
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
    image_name: str | None,
    tag: str,
    base_image: str | None,
    install_command: str | None,
    build_command: str | None,
    no_wait: bool,
    remote_build_flag: bool | None,
    docker_build_args: Sequence[str],
):
    click.secho(
        "Note: 'langgraph deploy' is in beta. Expect frequent updates and improvements.",
        fg="yellow",
    )
    click.echo()

    # -- 1. Preflight --
    validate_deploy_commands(install_command, build_command)
    config_json = langgraph_cli.config.validate_config_file(config)
    warn_non_wolfi_distro(config_json)

    env_vars = _parse_env_from_config(config_json, config)

    if not deployment_id and not name:
        name = env_vars.get(_DEPLOYMENT_NAME_ENV)
    if not deployment_id and not name:
        default_name = normalize_image_name(pathlib.Path.cwd().name)
        name = click.prompt("Deployment name", default=default_name)
        env_path = _resolve_env_path(config_json, config)
        if env_path is not None:
            set_key(str(env_path), _DEPLOYMENT_NAME_ENV, name)
            click.echo(f"Saved deployment name to {env_path}")

    secrets = _secrets_from_env(_env_without_deployment_name(env_vars))

    use_remote_build, local_build_error = _resolve_build_mode(remote_build_flag)
    if use_remote_build and remote_build_flag is None and local_build_error:
        click.secho(f"{local_build_error}\nUsing remote build instead.", fg="yellow")
        click.echo()

    # -- 2. Resolve / create deployment --
    client = _create_host_backend_client(host_url, api_key, env_vars=env_vars)
    step = 1

    deployment_id, needs_creation, step = _resolve_deployment(
        client,
        step,
        deployment_id,
        name,
        not_found_message=(
            "   No deployment found. Will create."
            if use_remote_build
            else "   No deployment found. Will create after build."
        ),
    )

    if needs_creation:
        deployment_id, step = _create_deployment(
            client,
            step,
            name=name,
            deployment_type=deployment_type,
            source="internal_source" if use_remote_build else "internal_docker",
            secrets=secrets,
        )

    if not deployment_id:
        raise click.ClickException("Failed to determine deployment ID")

    # -- 3. Build (divergent path) --
    if use_remote_build:
        build_result = _run_remote_build(
            client=client,
            deployment_id=deployment_id,
            step=step,
            config=config,
            config_json=config_json,
            verbose=verbose,
            install_command=install_command,
            build_command=build_command,
            secrets=secrets,
        )
    else:
        build_result = _run_local_build(
            client=client,
            deployment_id=deployment_id,
            step=step,
            config=config,
            config_json=config_json,
            verbose=verbose,
            pull=pull,
            api_version=api_version,
            base_image=base_image,
            image_name=image_name,
            name=name,
            tag=tag,
            install_command=install_command,
            build_command=build_command,
            docker_build_args=docker_build_args,
            secrets=secrets,
        )

    # -- 4. Shared wait + result --
    _print_deployment_status_url(build_result.updated, deployment_id, host_url)

    if no_wait:
        click.secho(f"   {build_result.no_result_message}", fg="green")
        return

    last_status, revision_id = _poll_revision_status(
        client,
        deployment_id,
        progress_message=build_result.progress_message,
        timeout_seconds=build_result.timeout_seconds,
        poll_interval_seconds=build_result.poll_interval_seconds,
        on_poll=build_result.on_poll,
        on_interrupt=build_result.on_interrupt,
    )
    if not last_status:
        click.secho(f"   {build_result.no_result_message}", fg="green")
        return

    if (
        build_result.show_build_logs_on_failure
        and last_status == "BUILD_FAILED"
        and not verbose
        and revision_id is not None
    ):
        click.secho("   Last build log lines:", fg="red")
        try:
            logs_resp = client.get_build_logs(
                deployment_id,
                revision_id,
                {"order": "desc", "limit": 30},
            )
            if isinstance(logs_resp, dict):
                entries = list(reversed(logs_resp.get("logs", [])))
                for entry in entries:
                    msg = entry.get("message", "")
                    if msg:
                        click.echo(f"   | {msg}")
        except Exception:
            click.secho("   (failed to fetch build logs)", fg="red")
        click.secho(
            "   Re-run with --verbose to see full build output.",
            fg="yellow",
        )

    _print_deployment_result(
        client,
        deployment_id,
        last_status,
        dashboard_label="Deployment dashboard",
    )


# ---------------------------------------------------------------------------
# deploy list
# ---------------------------------------------------------------------------


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
    response = _call_host_backend_with_optional_tenant(
        client,
        lambda c: c.list_revisions(deployment_id, limit=limit),
    )
    resources = response.get("resources", []) if isinstance(response, dict) else []
    revisions = [item for item in resources if isinstance(item, dict)]
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
    host_url: str,
):
    env_vars = _parse_env_from_config({}, pathlib.Path.cwd() / DEFAULT_CONFIG)
    client = _create_host_backend_client(host_url, api_key, env_vars=env_vars)
    if not deployment_id and not name:
        name = env_vars.get(_DEPLOYMENT_NAME_ENV)
    validate_deployment_selector(deployment_id, name)
    if deployment_id:
        dep_id = deployment_id
    else:
        found = _call_host_backend_with_optional_tenant(
            client, lambda c: find_deployment_id_by_name(c, name)
        )
        if not found:
            raise click.ClickException(f"Deployment '{name}' not found.")
        dep_id = str(found)

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
