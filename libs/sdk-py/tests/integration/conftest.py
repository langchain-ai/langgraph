"""Shared fixtures for the integration suite.

These tests require a running langgraph-api server at `LANGGRAPH_INTEGRATION_URL`
(defaults to `http://localhost:2024`). Stand it up via the docker stack in
`libs/sdk-py/integration/`:

    cd libs/sdk-py/integration && docker compose up -d

The `integration` marker is registered in `pyproject.toml` and excluded by
default in pytest's `addopts`; opt in with `pytest -m integration`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest

BASE_URL = os.environ.get("LANGGRAPH_INTEGRATION_URL", "http://localhost:2024")
ASSISTANT_ID = "agent"
TOOLS_ASSISTANT_ID = "tools_agent"
DEEP_AGENT_ASSISTANT_ID = "deep_agent"

EXPECTED_TERMINAL_ITEMS = ["streamed", "tool", "asked", "sub"]


@pytest.fixture(scope="session", autouse=True)
def _require_running_api() -> None:
    """Skip the whole integration suite if the API isn't reachable.

    Autouse + session-scoped so a missing stack short-circuits before any
    test runs (no per-test connection timeouts piling up).
    """
    try:
        resp = httpx.get(f"{BASE_URL}/ok", timeout=2.0)
        resp.raise_for_status()
    except Exception as err:
        pytest.skip(
            f"langgraph-api not reachable at {BASE_URL}: {err!r}. "
            f"Bring up the stack with `cd libs/sdk-py/integration && docker compose up -d`."
        )


@pytest.fixture
async def async_threads() -> AsyncIterator[tuple[object, httpx.AsyncClient]]:
    """Build an async ThreadsClient. Yields `(threads, raw_httpx)` so tests can close raw."""
    from langgraph_sdk._async.http import HttpClient
    from langgraph_sdk._async.threads import ThreadsClient

    raw = httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)
    try:
        yield ThreadsClient(HttpClient(raw)), raw
    finally:
        await raw.aclose()


@pytest.fixture
def sync_threads() -> Iterator[tuple[object, httpx.Client]]:
    """Build a sync ThreadsClient. Yields `(threads, raw_httpx)` so tests can close raw."""
    from langgraph_sdk._sync.http import SyncHttpClient
    from langgraph_sdk._sync.threads import SyncThreadsClient

    raw = httpx.Client(base_url=BASE_URL, timeout=30.0)
    try:
        yield SyncThreadsClient(SyncHttpClient(raw)), raw
    finally:
        raw.close()
