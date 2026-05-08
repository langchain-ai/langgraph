"""HTTP client for LangGraph host backend deployments."""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

import click
import httpx

# Allowlist of permitted host patterns for outbound HTTP requests.
# Entries may be exact hostnames or fnmatch-style patterns.
_ALLOWED_HOST_PATTERNS: list[str] = [
    "api.langchain.com",
    "api.smith.langchain.com",
    "*.langchain.com",
    "*.langgraph.com",
    "localhost",
    "127.0.0.1",
]

# Private/reserved IP ranges that must never be contacted.
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local / metadata
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("::1/128"),
]


def _fnmatch_host(pattern: str, host: str) -> bool:
    """Return True if *host* matches *pattern* (supports leading '*.' wildcard)."""
    if pattern.startswith("*."):
        suffix = pattern[1:]  # e.g. ".langchain.com"
        return host == pattern[2:] or host.endswith(suffix)
    return host == pattern


def _validate_base_url(base_url: str) -> None:
    """Raise HostBackendError if *base_url* is not on the allowlist."""
    parsed = urlparse(base_url)

    # Enforce https (or http for localhost in dev).
    if parsed.scheme not in ("https", "http"):
        raise HostBackendError(
            f"Disallowed URL scheme '{parsed.scheme}' in base_url '{base_url}'. "
            "Only 'https' is permitted."
        )

    host = parsed.hostname or ""
    if not host:
        raise HostBackendError(f"Could not determine hostname from base_url '{base_url}'.")

    # Block private / metadata IP ranges.
    try:
        addr = ipaddress.ip_address(host)
        for net in _PRIVATE_NETWORKS:
            if addr in net:
                raise HostBackendError(
                    f"base_url '{base_url}' resolves to a private/reserved address "
                    f"'{host}' which is not permitted."
                )
    except ValueError:
        pass  # host is a domain name, not an IP — continue with pattern check.

    # Check against the allowlist.
    for pattern in _ALLOWED_HOST_PATTERNS:
        if _fnmatch_host(pattern, host):
            return

    raise HostBackendError(
        f"base_url host '{host}' is not on the permitted allowlist. "
        f"Allowed patterns: {_ALLOWED_HOST_PATTERNS}"
    )


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

        # Validate the base_url against the allowlist before use.
        _validate_base_url(base_url)

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

    def create_deployment(
        self,
        name: str,
        deployment_type: str,
        source: str,
        config_path: str | None = None,
        secrets: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Create a deployment."""
        payload: dict[str, Any] = {
            "name": name,
            "source": source,
            "source_config": {"deployment_type": deployment_type},
            "source_revision_config": {},
        }
        if source == "internal_source" and config_path:
            payload["source_revision_config"]["langgraph_config_path"] = config_path
        if secrets is not None:
            payload["secrets"] = secrets
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
        """Delete a deployment after obtaining explicit human confirmation."""
        confirmed = click.confirm(
            f"Are you sure you want to delete deployment '{deployment_id}'? "
            "This action is irreversible.",
            default=False,
        )
        if not confirmed:
            raise click.Abort()
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
        secrets: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "revision_source": "internal_docker",
            "source_revision_config": {"image_uri": image_uri},
        }
        if secrets is not None:
            payload["secrets"] = secrets
        return self._request(
            "PATCH",
            f"/v2/deployments/{deployment_id}",
            payload,
        )

    def update_deployment_internal_source(
        self,
        deployment_id: str,
        source_tarball_path: str,
        config_path: str,
        secrets: list[dict[str, str]] | None = None,
        install_command: str | None = None,
        build_command: str | None = None,
    ) -> dict[str, Any]:
        """Trigger a remote build revision with the uploaded tarball."""
        payload: dict[str, Any] = {
            "revision_source": "internal_source",
            "source_revision_config": {
                "source_tarball_path": source_tarball_path,
                "langgraph_config_path": config_path,
            },
        }

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