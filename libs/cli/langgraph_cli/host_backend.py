"""HTTP client for LangGraph host backend deployments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

import click
import httpx

CLOUD_CONTROL_PLANE_URL = "https://api.host.langchain.com"
CLOUD_DASHBOARD_URL = "https://smith.langchain.com"
CLOUD_DOMAIN = "langchain.com"
CLOUD_API_HOST = "api.smith.langchain.com"
CLOUD_CONTROL_PLANE_HOST = "api.host.langchain.com"
CLOUD_DASHBOARD_HOST = "smith.langchain.com"
CONTROL_PLANE_PATH = "/api-host"
LANGSMITH_API_PATHS = ("/api/v1", "/api")
LOCAL_HOSTNAMES = ("localhost", "127.0.0.1")
MAX_PAGE_SIZE = 100
SourceName = Literal["internal_docker", "internal_source", "external_docker"]


@dataclass(frozen=True, slots=True)
class ControlPlaneEndpoints:
    control_plane_url: str
    dashboard_url: str

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
            return cls(control_plane_url, control_plane_url[: -len(CONTROL_PLANE_PATH)])
        if hostname in LOCAL_HOSTNAMES:
            return cls(control_plane_url, control_plane_url)
        return cls(control_plane_url, _cloud_dashboard_for(hostname))

    @classmethod
    def from_langsmith_endpoint(cls, endpoint: str) -> ControlPlaneEndpoints:
        parsed = urlparse(endpoint.rstrip("/"))
        hostname = parsed.hostname or ""
        if _is_cloud_host(hostname):
            return cls.from_control_plane_url(
                f"https://{_cloud_control_plane_host_for(hostname)}"
            )
        root = f"{parsed.scheme}://{parsed.netloc}{_without_api_path(parsed.path)}"
        return cls(f"{root}{CONTROL_PLANE_PATH}", root)


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


class HostBackendClient:
    """Minimal JSON HTTP client for the host backend deployment service."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        tenant_id: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        if not base_url:
            raise click.UsageError("Host backend URL is required")
        headers: dict[str, str] = {
            "X-Api-Key": api_key,
            "Accept": "application/json",
        }
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id
        self._endpoints = ControlPlaneEndpoints.from_control_plane_url(base_url)
        self._base_url = self._endpoints.control_plane_url
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            transport=transport or httpx.HTTPTransport(retries=3),
            timeout=30,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def endpoints(self) -> ControlPlaneEndpoints:
        return self._endpoints

    def set_tenant(self, tenant_id: str) -> None:
        self._client.headers["X-Tenant-ID"] = tenant_id

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
        limit: int | None = None,
        agent_id: str | None = None,
        agent_environment: str | None = None,
    ) -> list[dict[str, Any]]:
        given = (
            ("name", name),
            ("name_contains", name_contains),
            ("limit", limit),
            ("agent_id", agent_id),
            ("agent_environment", agent_environment),
        )
        params = {key: value for key, value in given if value is not None}
        return _resources(self._request("GET", "/v2/deployments", params=params))

    def get_listener(self, listener_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v2/listeners/{listener_id}")

    def list_listeners(self) -> list[dict[str, Any]]:
        return _resources(
            self._request("GET", "/v2/listeners", params={"limit": MAX_PAGE_SIZE})
        )

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v2/deployments/{deployment_id}")

    def delete_deployment(self, deployment_id: str) -> None:
        return self._request("DELETE", f"/v2/deployments/{deployment_id}")

    def request_push_token(self, deployment_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v2/deployments/{deployment_id}/push-token",
        )

    def request_upload_url(self, deployment_id: str) -> dict[str, Any]:
        """Get a signed GCS URL for uploading the source tarball."""
        return self._request(
            "POST",
            f"/v2/deployments/{deployment_id}/upload-url",
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
