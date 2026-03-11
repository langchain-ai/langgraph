"""HTTP client for LangGraph host backend deployments."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import click


class HostBackendError(click.ClickException):
    """Raised when the host backend returns an error response."""


class HostBackendClient:
    """Minimal JSON HTTP client for the host backend deployment service."""

    def __init__(self, base_url: str, api_key: str):
        if not base_url:
            raise click.UsageError("Host backend URL is required")
        base_url = base_url.rstrip("/")
        self._base_url = base_url
        self._api_key = api_key

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        url = f"{self._base_url}{path}"
        data: bytes | None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        else:
            data = None
        headers: dict[str, str] = {
            "X-Api-Key": self._api_key,
            "Accept": "application/json",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="ignore")
            message = detail or err.reason
            raise HostBackendError(
                f"{method} {path} failed with status {err.code}: {message}"
            ) from None
        except urllib.error.URLError as err:
            raise HostBackendError(str(err.reason)) from None

        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as err:
            raise HostBackendError(
                f"Failed to decode response from {path}: {err.msg}"
            ) from None

    def create_deployment(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v2/deployments", payload)

    def list_deployments(self, name_contains: str) -> dict[str, Any]:
        encoded = urllib.parse.quote(name_contains, safe="")
        return self._request("GET", f"/v2/deployments?name_contains={encoded}")

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v2/deployments/{deployment_id}")

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

    def request_upload_url(self, deployment_id: str) -> dict[str, Any]:
        """Get a signed GCS URL for uploading the source tarball."""
        return self._request(
            "POST",
            f"/v2/deployments/{deployment_id}/upload-url",
        )

    def update_deployment_internal_source(
        self,
        deployment_id: str,
        source_tarball_path: str,
        secrets: list[dict[str, str]] | None = None,
        config_path: str | None = None,
        install_command: str | None = None,
        build_command: str | None = None,
    ) -> dict[str, Any]:
        """Trigger a remote build revision with the uploaded tarball."""
        src_config: dict[str, Any] = {
            "source_tarball_path": source_tarball_path,
        }
        if config_path is not None:
            src_config["langgraph_config_path"] = config_path

        payload: dict[str, Any] = {"source_revision_config": src_config}

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

    def list_build_logs(
        self,
        deployment_id: str,
        revision_id: str,
        order: str = "asc",
        limit: int = 50,
        offset: str | None = None,
    ) -> dict[str, Any]:
        """Fetch build logs for a revision."""
        payload: dict[str, Any] = {"order": order, "limit": limit}
        if offset:
            payload["offset"] = offset
        return self._request(
            "POST",
            f"/v1/projects/{deployment_id}/revisions/{revision_id}/build_logs",
            payload,
        )
