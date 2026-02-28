"""Shared utility functions for async and sync clients."""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Mapping
from typing import Any, cast

import httpx

import langgraph_sdk
from langgraph_sdk.schema import RunCreateMetadata

RESERVED_HEADERS = ("x-api-key",)

NOT_PROVIDED = cast(None, object())


def _get_api_key(api_key: str | None = NOT_PROVIDED) -> str | None:
    """Get the API key from the environment.
    Precedence:
        1. explicit string argument
        2. LANGGRAPH_API_KEY (if api_key not provided)
        3. LANGSMITH_API_KEY (if api_key not provided)
        4. LANGCHAIN_API_KEY (if api_key not provided)

    Args:
        api_key: The API key to use. Can be:
            - A string: use this exact API key
            - None: explicitly skip loading from environment
            - NOT_PROVIDED (default): auto-load from environment variables
    """
    if isinstance(api_key, str):
        return api_key
    if api_key is NOT_PROVIDED:
        # api_key is not explicitly provided, try to load from environment
        for prefix in ["LANGGRAPH", "LANGSMITH", "LANGCHAIN"]:
            if env := os.getenv(f"{prefix}_API_KEY"):
                return env.strip().strip('"').strip("'")
    # api_key is explicitly None, don't load from environment
    return None


def _get_headers(
    api_key: str | None,
    custom_headers: Mapping[str, str] | None,
) -> dict[str, str]:
    """Combine api_key and custom user-provided headers."""
    custom_headers = custom_headers or {}
    for header in RESERVED_HEADERS:
        if header in custom_headers:
            raise ValueError(f"Cannot set reserved header '{header}'")

    headers = {
        "User-Agent": f"langgraph-sdk-py/{langgraph_sdk.__version__}",
        **custom_headers,
    }
    resolved_api_key = _get_api_key(api_key)
    if resolved_api_key:
        headers["x-api-key"] = resolved_api_key

    return headers


def _orjson_default(obj: Any) -> Any:
    is_class = isinstance(obj, type)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        if is_class:
            raise TypeError(
                f"Cannot JSON-serialize type object: {obj!r}. Did you mean to pass an instance of the object instead?"
                f"\nReceived type: {obj!r}"
            )
        return obj.model_dump()
    elif hasattr(obj, "dict") and callable(obj.dict):
        if is_class:
            raise TypeError(
                f"Cannot JSON-serialize type object: {obj!r}. Did you mean to pass an instance of the object instead?"
                f"\nReceived type: {obj!r}"
            )
        return obj.dict()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Compiled regex pattern for extracting run metadata from Content-Location header
_RUN_METADATA_PATTERN = re.compile(
    r"(\/threads\/(?P<thread_id>.+))?\/runs\/(?P<run_id>.+)"
)


def _get_run_metadata_from_response(
    response: httpx.Response,
) -> RunCreateMetadata | None:
    """Extract run metadata from the response headers."""
    if (content_location := response.headers.get("Content-Location")) and (
        match := _RUN_METADATA_PATTERN.search(content_location)
    ):
        return RunCreateMetadata(
            run_id=match.group("run_id"),
            thread_id=match.group("thread_id") or None,
        )

    return None


def _provided_vals(d: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


_registered_transports: list[httpx.ASGITransport] = []


# Do not move; this is used in the server.
def configure_loopback_transports(app: Any) -> None:
    for transport in _registered_transports:
        transport.app = app


@functools.lru_cache(maxsize=1)
def get_asgi_transport() -> type[httpx.ASGITransport]:
    try:
        from langgraph_api import asgi_transport  # type: ignore[unresolved-import]

        return asgi_transport.ASGITransport
    except ImportError:
        # Older versions of the server
        return httpx.ASGITransport
