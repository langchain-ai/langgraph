"""Async LangGraph client."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from types import TracebackType

import httpx

from langgraph_sdk._async.assistants import AssistantsClient
from langgraph_sdk._async.cron import CronClient
from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.runs import RunsClient
from langgraph_sdk._async.store import StoreClient
from langgraph_sdk._async.threads import ThreadsClient
from langgraph_sdk._shared.types import TimeoutTypes
from langgraph_sdk._shared.utilities import (
    NOT_PROVIDED,
    _get_headers,
    _registered_transports,
    get_asgi_transport,
)

logger = logging.getLogger(__name__)


def get_client(
    *,
    url: str | None = None,
    api_key: str | None = NOT_PROVIDED,
    headers: Mapping[str, str] | None = None,
    timeout: TimeoutTypes | None = None,
) -> LangGraphClient:
    """Create and configure a LangGraphClient.

    The client provides programmatic access to LangSmith Deployment. It supports
    both remote servers and local in-process connections (when running inside a LangGraph server).

    Args:
        url:
            Base URL of the LangGraph API.
            - If `None`, the client first attempts an in-process connection via ASGI transport.
              If that fails, it defers registration until after app initialization. This
              only works if the client is used from within the Agent server.
        api_key:
            API key for authentication. Can be:
              - A string: use this exact API key
              - `None`: explicitly skip loading from environment variables
              - Not provided (default): auto-load from environment in this order:
                1. `LANGGRAPH_API_KEY`
                2. `LANGSMITH_API_KEY`
                3. `LANGCHAIN_API_KEY`
        headers:
            Additional HTTP headers to include in requests. Merged with authentication headers.
        timeout:
            HTTP timeout configuration. May be:
              - `httpx.Timeout` instance
              - float (total seconds)
              - tuple `(connect, read, write, pool)` in seconds
            Defaults: connect=5, read=300, write=300, pool=5.

    Returns:
        LangGraphClient:
            A top-level client exposing sub-clients for assistants, threads,
            runs, and cron operations.

    ???+ example "Connect to a remote server:"

        ```python
        from langgraph_sdk import get_client

        # get top-level LangGraphClient
        client = get_client(url="http://localhost:8123")

        # example usage: client.<model>.<method_name>()
        assistants = await client.assistants.get(assistant_id="some_uuid")
        ```

    ???+ example "Connect in-process to a running LangGraph server:"

        ```python
        from langgraph_sdk import get_client

        client = get_client(url=None)

        async def my_node(...):
            subagent_result = await client.runs.wait(
                thread_id=None,
                assistant_id="agent",
                input={"messages": [{"role": "user", "content": "Foo"}]},
            )
        ```

    ???+ example "Skip auto-loading API key from environment:"

        ```python
        from langgraph_sdk import get_client

        # Don't load API key from environment variables
        client = get_client(
            url="http://localhost:8123",
            api_key=None
        )
        ```
    """

    transport: httpx.AsyncBaseTransport | None = None
    if url is None:
        url = "http://api"
        if os.environ.get("__LANGGRAPH_DEFER_LOOPBACK_TRANSPORT") == "true":
            transport = get_asgi_transport()(app=None, root_path="/noauth")
            _registered_transports.append(transport)
        else:
            try:
                from langgraph_api.server import app  # type: ignore

                transport = get_asgi_transport()(app, root_path="/noauth")
            except Exception:
                logger.debug(
                    "Failed to connect to in-process LangGraph server. Deferring configuration.",
                    exc_info=True,
                )
                transport = get_asgi_transport()(app=None, root_path="/noauth")
                _registered_transports.append(transport)

    if transport is None:
        transport = httpx.AsyncHTTPTransport(retries=5)
    client = httpx.AsyncClient(
        base_url=url,
        transport=transport,
        timeout=(
            httpx.Timeout(timeout)  # type: ignore[arg-type]
            if timeout is not None
            else httpx.Timeout(connect=5, read=300, write=300, pool=5)
        ),
        headers=_get_headers(api_key, headers),
    )
    return LangGraphClient(client)


class LangGraphClient:
    """Top-level client for LangGraph API.

    Attributes:
        assistants: Manages versioned configuration for your graphs.
        threads: Handles (potentially) multi-turn interactions, such as conversational threads.
        runs: Controls individual invocations of the graph.
        crons: Manages scheduled operations.
        store: Interfaces with persistent, shared data storage.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.http = HttpClient(client)
        self.assistants = AssistantsClient(self.http)
        self.threads = ThreadsClient(self.http)
        self.runs = RunsClient(self.http)
        self.crons = CronClient(self.http)
        self.store = StoreClient(self.http)

    async def __aenter__(self) -> LangGraphClient:
        """Enter the async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the async context manager."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if hasattr(self, "http"):
            await self.http.client.aclose()
