"""Shared helpers for the v3 streaming integration scripts.

All scripts share the same expectations:

- A `langgraph-api` server is reachable at `BASE_URL` (default
  http://localhost:2024 — set by `docker-compose.yml`, which builds
  on `langchain/langgraph-api:latest-py3.12`).
- The example graph (`integration/graph/streaming_graph.py:graph`) is
  registered under the assistant id `agent` (see `integration/langgraph.json`).

Each script imports `make_async_client()` / `make_sync_client()` from here
to construct the v3 SDK client. Override `BASE_URL` via the
`LANGGRAPH_INTEGRATION_URL` env var if you're running the API elsewhere.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from langgraph_sdk._async.threads import ThreadsClient as AsyncThreadsClient
    from langgraph_sdk._sync.threads import SyncThreadsClient

BASE_URL = os.environ.get("LANGGRAPH_INTEGRATION_URL", "http://localhost:2024")
ASSISTANT_ID = "agent"


def make_async_client() -> tuple[AsyncThreadsClient, httpx.AsyncClient]:
    """Build an async ThreadsClient pointing at the integration API.

    Returns the client and the underlying httpx client so callers can close
    it. Typical usage:

    ```python
    threads, raw = make_async_client()
    try:
        async with threads.stream(...) as thread:
            ...
    finally:
        await raw.aclose()
    ```
    """
    from langgraph_sdk._async.http import HttpClient
    from langgraph_sdk._async.threads import ThreadsClient

    raw = httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)
    return ThreadsClient(HttpClient(raw)), raw


def make_sync_client() -> tuple[SyncThreadsClient, httpx.Client]:
    """Build a sync ThreadsClient pointing at the integration API."""
    from langgraph_sdk._sync.http import SyncHttpClient
    from langgraph_sdk._sync.threads import SyncThreadsClient

    raw = httpx.Client(base_url=BASE_URL, timeout=30.0)
    return SyncThreadsClient(SyncHttpClient(raw)), raw


def header(title: str) -> None:
    """Print a section header for script output."""
    bar = "=" * (len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}")


def auto_respond_async(thread: Any, response: Any = "yes") -> asyncio.Task[None]:
    """Spawn a background task that responds to the first interrupt and exits.

    Opens a private `thread.values` subscription and drains it; the SDK's
    `_signal_paused` mechanism pushes the terminal sentinel into the
    iterator when `interrupted` flips True, so the loop exits at the
    interrupt. The auto-responder then calls `run.respond(...)` and the
    foreground iteration sees the rest of the run.

    Returns the task so callers can `await` it before tearing down the
    stream (recommended) or cancel it.
    """

    async def _runner() -> None:
        async for _ in thread.values:
            if thread.interrupted:
                break
        if thread.interrupted:
            with contextlib.suppress(Exception):
                await thread.run.respond(response)

    return asyncio.create_task(_runner())


def auto_respond_sync(thread: Any, response: Any = "yes") -> threading.Thread:
    """Sync analogue of `auto_respond_async`."""

    def _runner() -> None:
        for _ in thread.values:
            if thread.interrupted:
                break
        if thread.interrupted:
            with contextlib.suppress(Exception):
                thread.run.respond(response)

    t = threading.Thread(target=_runner, daemon=True, name="auto-respond")
    t.start()
    return t


def check_api_reachable() -> None:
    """Fail fast with a helpful message if the API isn't reachable.

    Call this at the top of `main()` in each script.
    """
    try:
        resp = httpx.get(f"{BASE_URL}/ok", timeout=2.0)
        resp.raise_for_status()
    except Exception as err:
        raise SystemExit(
            f"\nCannot reach the integration API at {BASE_URL}: {err!r}\n"
            f"Did you run `docker compose up -d` from `libs/sdk-py/integration/`?\n"
            f"Or set LANGGRAPH_INTEGRATION_URL=... to point elsewhere.\n"
        ) from err
