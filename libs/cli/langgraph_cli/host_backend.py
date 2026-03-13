"""HTTP client for LangGraph host backend deployments."""

from __future__ import annotations

from typing import Any

import click
import httpx


class HostBackendError(click.ClickException):
    """Raised when the host backend returns an error response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class HostBackendClient:
    """Minimal JSON HTTP client for the host backend deployment service."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        tenant_id: str | None = None,
    ):
        if not base_url:
            raise click.UsageError("Host backend URL is required")
        transport = httpx.HTTPTransport(retries=3)
        headers: dict[str, str] = {
            "X-Api-Key": api_key,
            "Accept": "application/json",
        }
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            transport=transport,
            timeout=30,
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
            detail = err.response.text or str(err.response.status_code)
            raise HostBackendError(
                f"{method} {path} failed with status {err.response.status_code}: {detail}",
                status_code=err.response.status_code,
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

    def create_deployment(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v2/deployments", payload)

    def list_deployments(self, name_contains: str = "") -> dict[str, Any]:
        return self._request(
            "GET",
            "/v2/deployments",
            params={"name_contains": name_contains},
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

    def update_deployment(
        self,
        deployment_id: str,
        image_uri: str,
        secrets: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_revision_config": {"image_uri": image_uri},
        }
        if secrets is not None:
            payload["secrets"] = secrets
        return self._request(
            "PATCH",
            f"/v2/deployments/{deployment_id}",
            payload,
        )

    def list_revisions(self, deployment_id: str, limit: int = 1) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v2/deployments/{deployment_id}/revisions?limit={limit}",
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
