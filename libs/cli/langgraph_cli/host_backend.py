"""HTTP client for LangGraph host backend deployments."""

from __future__ import annotations

import json
import urllib.error
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
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self._api_key,
            "Accept": "application/json",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310
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

    def request_push_token(self, deployment_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v2/deployments/{deployment_id}/push-token",
        )

    def update_deployment_image(
        self, deployment_id: str, image_uri: str
    ) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/v2/deployments/{deployment_id}",
            {"source_revision_config": {"image_uri": image_uri}},
        )
