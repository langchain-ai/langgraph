"""HTTP client for LangGraph host backend deployments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import UUID

import click
import httpx

CLOUD_CONTROL_PLANE_URL = "https://api.host.langchain.com"
CLOUD_DASHBOARD_URL = "https://smith.langchain.com"
CLOUD_DOMAIN = "langchain.com"
CLOUD_API_HOST = "api.smith.langchain.com"
CLOUD_CONTROL_PLANE_HOST = "api.host.langchain.com"
CLOUD_DASHBOARD_HOST = "smith.langchain.com"
CLOUD_LANGSMITH_API_URL = f"https://{CLOUD_API_HOST}"
CONTROL_PLANE_PATH = "/api-host"
INFO_PATH = "/api/v1/info"
LANGSMITH_API_PATHS = ("/api/v1", "/api")
LOCAL_HOSTNAMES = ("localhost", "127.0.0.1")
MAX_PAGE_SIZE = 100
WORKSPACE_LOOKUP_TIMEOUT_SECONDS = 5
SourceName = Literal["internal_docker", "internal_source", "external_docker"]


@dataclass(frozen=True, slots=True)
class ControlPlaneEndpoints:
    control_plane_url: str
    dashboard_url: str
    info_url: str | None = None

    @classmethod
    def resolve(
        cls, host_url: str | None, langsmith_endpoint: str | None
    ) -> ControlPlaneEndpoints:
        if host_url:
            return cls.from_control_plane_url(host_url)
        if langsmith_endpoint:
            return cls.from_langsmith_endpoint(langsmith_endpoint)
        return cls(CLOUD_CONTROL_PLANE_URL, CLOUD_DASHBOARD_URL)

    @property
    def is_cloud(self) -> bool:
        hostname = urlparse(self.control_plane_url).hostname or ""
        return hostname == CLOUD_CONTROL_PLANE_HOST or hostname.endswith(
            f".{CLOUD_CONTROL_PLANE_HOST}"
        )

    @classmethod
    def from_control_plane_url(cls, url: str) -> ControlPlaneEndpoints:
        control_plane_url = url.rstrip("/")
        hostname = urlparse(control_plane_url).hostname or ""
        if control_plane_url.endswith(CONTROL_PLANE_PATH):
            return cls(
                control_plane_url, control_plane_url.removesuffix(CONTROL_PLANE_PATH)
            )
        if hostname in LOCAL_HOSTNAMES:
            return cls(control_plane_url, control_plane_url)
        return cls(control_plane_url, _cloud_dashboard_for(hostname))

    @classmethod
    def from_langsmith_endpoint(cls, endpoint: str) -> ControlPlaneEndpoints:
        hostname = urlparse(endpoint).hostname or ""
        if _is_cloud_host(hostname):
            return cls.from_control_plane_url(
                f"https://{_cloud_control_plane_host_for(hostname)}"
            )
        root = langsmith_api_url(endpoint)
        return cls(f"{root}{CONTROL_PLANE_PATH}", root)

    @classmethod
    def on_data_plane(
        cls, data_plane_url: str, langsmith_endpoint: str | None
    ) -> ControlPlaneEndpoints:
        _require_https(data_plane_url)
        endpoint_is_the_data_plane = (
            langsmith_endpoint is not None
            and urlparse(langsmith_endpoint).hostname
            == urlparse(data_plane_url).hostname
        )
        dashboard_url = cls.resolve(
            None, None if endpoint_is_the_data_plane else langsmith_endpoint
        ).dashboard_url
        root = data_plane_url.rstrip("/")
        return cls(f"{root}{CONTROL_PLANE_PATH}", dashboard_url, f"{root}{INFO_PATH}")

    @classmethod
    def for_workspace(
        cls, data_plane_url: str | None, langsmith_endpoint: str | None
    ) -> ControlPlaneEndpoints:
        if data_plane_url:
            return cls.on_data_plane(data_plane_url, langsmith_endpoint)
        return cls.resolve(None, langsmith_endpoint)


def _require_https(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme == "https" or (
        parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTNAMES
    ):
        return
    raise click.ClickException(
        f"Refusing to send your API key to the data plane at {url}: "
        "data plane URLs must use https."
    )


def langsmith_api_url(langsmith_endpoint: str | None) -> str:
    if not langsmith_endpoint:
        return CLOUD_LANGSMITH_API_URL
    parsed = urlparse(langsmith_endpoint.rstrip("/"))
    return f"{parsed.scheme}://{parsed.netloc}{_without_api_path(parsed.path)}"


def _is_cloud_host(hostname: str) -> bool:
    return hostname == CLOUD_DOMAIN or hostname.endswith(f".{CLOUD_DOMAIN}")


def _cloud_control_plane_host_for(langsmith_api_host: str) -> str:
    if langsmith_api_host.endswith(f".{CLOUD_API_HOST}"):
        region = langsmith_api_host[: -len(CLOUD_API_HOST)]
        return f"{region}{CLOUD_CONTROL_PLANE_HOST}"
    return CLOUD_CONTROL_PLANE_HOST


def _cloud_dashboard_for(control_plane_host: str) -> str:
    if control_plane_host.endswith(f".{CLOUD_CONTROL_PLANE_HOST}"):
        region = control_plane_host[: -len(CLOUD_CONTROL_PLANE_HOST) - 1]
        return f"https://{region}.{CLOUD_DASHBOARD_HOST}"
    return CLOUD_DASHBOARD_URL


def _without_api_path(path: str) -> str:
    for api_path in LANGSMITH_API_PATHS:
        if path.endswith(api_path):
            return path[: -len(api_path)]
    return path


def _resources(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    resources = payload.get("resources")
    if not isinstance(resources, list):
        return []
    return [item for item in resources if isinstance(item, dict)]


class HostBackendError(click.ClickException):
    """Raised when the host backend returns an error response."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        detail: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def _error_detail(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, str) else None


@dataclass(frozen=True, slots=True)
class Workspace:
    id: str
    data_plane_url: str | None

    @classmethod
    def from_api(cls, workspace: dict[str, Any]) -> Workspace:
        data_plane_url = (workspace.get("data_plane_url") or "").rstrip("/")
        return cls(workspace["id"], data_plane_url or None)

    def has_id(self, workspace_id: str) -> bool:
        return _as_uuid(self.id) == _as_uuid(workspace_id)


def _as_uuid(workspace_id: str) -> UUID | str:
    try:
        return UUID(workspace_id)
    except ValueError:
        return workspace_id


@dataclass(frozen=True, slots=True)
class WorkspaceRouter:
    workspaces: Sequence[Workspace]
    langsmith_endpoint: str | None

    def endpoints_for(self, workspace_id: str) -> ControlPlaneEndpoints | None:
        workspace = next((w for w in self.workspaces if w.has_id(workspace_id)), None)
        if workspace is None:
            return None
        return ControlPlaneEndpoints.for_workspace(
            workspace.data_plane_url, self.langsmith_endpoint
        )


@dataclass(frozen=True, slots=True)
class InstanceFlags:
    scale_to_zero: bool
    dedicated: bool
    cli_builds: bool

    @classmethod
    def from_info(cls, info: object) -> InstanceFlags:
        flags = info.get("instance_flags") if isinstance(info, dict) else None
        if not isinstance(flags, dict):
            flags = {}
        return cls(
            scale_to_zero=flags.get("deployments_scale_to_zero_supported") is True,
            dedicated=flags.get("deployments_dedicated_supported") is True,
            cli_builds=flags.get("deployments_cli_builds_supported") is True,
        )


def _api_key_headers(api_key: str) -> dict[str, str]:
    return {"X-Api-Key": api_key, "Accept": "application/json"}


class _JsonClient:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        transport: httpx.BaseTransport | None,
        *,
        timeout: float = 30,
        retries: int = 3,
    ):
        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            transport=transport or httpx.HTTPTransport(retries=retries),
            timeout=timeout,
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            resp = self._client.request(method, path, json=payload, params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as err:
            detail = _error_detail(err.response)
            reason = detail or err.response.text or str(err.response.status_code)
            raise HostBackendError(
                f"{method} {path} failed with status {err.response.status_code}: {reason}",
                status_code=err.response.status_code,
                detail=detail,
            ) from None
        except httpx.TransportError as err:
            raise HostBackendError(str(err)) from None

        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError as err:
            raise HostBackendError(
                f"Failed to decode response from {path}: {err}"
            ) from None


class LangSmithApiClient(_JsonClient):
    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        super().__init__(
            api_url.rstrip("/"),
            _api_key_headers(api_key),
            transport,
            timeout=WORKSPACE_LOOKUP_TIMEOUT_SECONDS,
            retries=0,
        )

    def list_workspaces(self) -> list[Workspace]:
        workspaces = self._request("GET", "/api/v1/workspaces")
        if not isinstance(workspaces, list):
            return []
        return [
            Workspace.from_api(workspace)
            for workspace in workspaces
            if isinstance(workspace, dict)
            and isinstance(workspace.get("id"), str)
            and not workspace.get("is_deleted")
        ]


class HostBackendClient(_JsonClient):
    """Minimal JSON HTTP client for the host backend deployment service."""

    def __init__(
        self,
        control_plane: str | ControlPlaneEndpoints,
        api_key: str,
        tenant_id: str | None = None,
        *,
        router: WorkspaceRouter | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        if not control_plane:
            raise click.UsageError("Host backend URL is required")
        headers = _api_key_headers(api_key)
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id
        self._endpoints = (
            control_plane
            if isinstance(control_plane, ControlPlaneEndpoints)
            else ControlPlaneEndpoints.from_control_plane_url(control_plane)
        )
        self._router = router or WorkspaceRouter((), None)
        super().__init__(self._endpoints.control_plane_url, headers, transport)

    @property
    def base_url(self) -> str:
        return self._endpoints.control_plane_url

    @property
    def endpoints(self) -> ControlPlaneEndpoints:
        return self._endpoints

    def set_tenant(self, tenant_id: str) -> None:
        endpoints = self._router.endpoints_for(tenant_id)
        if endpoints is not None:
            self._endpoints = endpoints
            self._client.base_url = endpoints.control_plane_url
        self._client.headers["X-Tenant-ID"] = tenant_id

    def instance_flags(self) -> InstanceFlags | None:
        """Data plane capabilities, or None when the host backend is not on one."""
        if self._endpoints.info_url is None:
            return None
        return InstanceFlags.from_info(self._request("GET", self._endpoints.info_url))

    def create_deployment(
        self,
        *,
        name: str | None,
        source: SourceName,
        source_config: dict[str, object],
        source_revision_config: dict[str, object],
        secrets: list[dict[str, str]] | None = None,
        agent: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": source,
            "source_config": source_config,
            "source_revision_config": source_revision_config,
        }
        if agent is not None:
            payload["agent"] = agent
        else:
            payload["name"] = name
        if secrets is not None:
            payload["secrets"] = secrets
        return self._request("POST", "/v2/deployments", payload)

    def list_deployments(
        self,
        *,
        name: str | None = None,
        name_contains: str | None = None,
        agent_id: str | None = None,
        agent_environment: str | None = None,
    ) -> list[dict[str, object]]:
        given = (
            ("name", name),
            ("name_contains", name_contains),
            ("agent_id", agent_id),
            ("agent_environment", agent_environment),
        )
        filters = {key: value for key, value in given if value is not None}
        return self._every_page("/v2/deployments", filters)

    def _every_page(
        self, path: str, filters: Mapping[str, object]
    ) -> list[dict[str, object]]:
        resources: list[dict[str, object]] = []
        seen_ids: set[object] = set()
        while True:
            page = _resources(
                self._request(
                    "GET",
                    path,
                    params={
                        **filters,
                        "limit": MAX_PAGE_SIZE,
                        "offset": len(resources),
                    },
                )
            )
            page_ids = {resource.get("id") for resource in page}
            if page_ids and page_ids <= seen_ids:
                raise HostBackendError(
                    f"GET {path} returned the same resources again: this control "
                    "plane ignores 'offset', so the CLI cannot read past the first page."
                )
            seen_ids |= page_ids
            resources.extend(page)
            if len(page) < MAX_PAGE_SIZE:
                return resources

    def get_listener(self, listener_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v2/listeners/{listener_id}")

    def list_listeners(self) -> list[dict[str, object]]:
        return self._every_page("/v2/listeners", {})

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v2/deployments/{deployment_id}")

    def delete_deployment(self, deployment_id: str) -> None:
        return self._request("DELETE", f"/v2/deployments/{deployment_id}")

    def request_push_token(self, deployment_id: str) -> dict[str, Any]:
        # Tells the control plane this client logs in with the returned
        # `username` and pushes to `repository_url`. LangSmith AWS requires it.
        return self._request(
            "POST",
            f"/v2/deployments/{deployment_id}/push-token",
            {"supports_registry_username": True},
        )

    def request_upload_url(self, deployment_id: str) -> dict[str, Any]:
        """Get a signed URL for uploading the source tarball."""
        # Tells the control plane this client sends the returned
        # `upload_headers` with its upload, which some storage requires.
        return self._request(
            "POST",
            f"/v2/deployments/{deployment_id}/upload-url",
            {"supports_upload_headers": True},
        )

    def update_deployment(
        self,
        deployment_id: str,
        image_uri: str,
        *,
        revision_source: SourceName | None,
        secrets: list[dict[str, str]] | None = None,
        tracked_packages: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_revision_config": {"image_uri": image_uri},
        }
        if revision_source is not None:
            payload["revision_source"] = revision_source
        if tracked_packages:
            payload["tracked_packages"] = tracked_packages
        if secrets is not None:
            payload["secrets"] = secrets
        return self._request("PATCH", f"/v2/deployments/{deployment_id}", payload)

    def update_deployment_internal_source(
        self,
        deployment_id: str,
        source_tarball_path: str,
        config_path: str,
        secrets: list[dict[str, str]] | None = None,
        install_command: str | None = None,
        build_command: str | None = None,
        tracked_packages: list[str] | None = None,
    ) -> dict[str, Any]:
        """Trigger a remote build revision with the uploaded tarball."""
        payload: dict[str, Any] = {
            "revision_source": "internal_source",
            "source_revision_config": {
                "source_tarball_path": source_tarball_path,
                "langgraph_config_path": config_path,
            },
        }
        if tracked_packages:
            payload["tracked_packages"] = tracked_packages

        source_config: dict[str, Any] = {}
        if install_command is not None:
            source_config["install_command"] = install_command
        if build_command is not None:
            source_config["build_command"] = build_command
        if source_config:
            payload["source_config"] = source_config

        if secrets is not None:
            payload["secrets"] = secrets
        return self._request("PATCH", f"/v2/deployments/{deployment_id}", payload)

    def list_revisions(
        self, deployment_id: str, limit: int = 1
    ) -> list[dict[str, Any]]:
        return _resources(
            self._request(
                "GET",
                f"/v2/deployments/{deployment_id}/revisions",
                params={"limit": limit},
            )
        )

    def get_revision(self, deployment_id: str, revision_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v2/deployments/{deployment_id}/revisions/{revision_id}",
        )

    def get_deployment_metrics(
        self, deployment_id: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request(
            "GET", f"/v2/deployments/{deployment_id}/metrics", params=params
        )

    def get_build_logs(
        self, project_id: str, revision_id: str, payload: dict[str, Any]
    ) -> Any:
        return self._request(
            "POST",
            f"/v1/projects/{project_id}/revisions/{revision_id}/build_logs",
            payload,
        )

    def get_deploy_logs(
        self,
        project_id: str,
        payload: dict[str, Any],
        revision_id: str | None = None,
    ) -> Any:
        if revision_id:
            path = f"/v1/projects/{project_id}/revisions/{revision_id}/deploy_logs"
        else:
            path = f"/v1/projects/{project_id}/deploy_logs"
        return self._request("POST", path, payload)
