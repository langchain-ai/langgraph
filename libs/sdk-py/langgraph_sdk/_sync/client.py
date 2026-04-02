"""Sync LangGraph client."""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType

import httpx

from langgraph_sdk._shared.types import TimeoutTypes
from langgraph_sdk._shared.utilities import NOT_PROVIDED, _get_headers
from langgraph_sdk._sync.assistants import SyncAssistantsClient
from langgraph_sdk._sync.cron import SyncCronClient
from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.runs import SyncRunsClient
from langgraph_sdk._sync.store import SyncStoreClient
from langgraph_sdk._sync.threads import SyncThreadsClient


def get_sync_client(
    *,
    url: str | None = None,
    api_key: str | None = NOT_PROVIDED,
    headers: Mapping[str, str] | None = None,
    timeout: TimeoutTypes | None = None,
) -> SyncLangGraphClient:
    """Get a synchronous LangGraphClient instance.

    Args:
        url: The URL of the LangGraph API.
        api_key: API key for authentication. Can be:
            - A string: use this exact API key
            - `None`: explicitly skip loading from environment variables
            - Not provided (default): auto-load from environment in this order:
                1. `LANGGRAPH_API_KEY`
                2. `LANGSMITH_API_KEY`
                3. `LANGCHAIN_API_KEY`
        headers: Optional custom headers
        timeout: Optional timeout configuration for the HTTP client.
            Accepts an httpx.Timeout instance, a float (seconds), or a tuple of timeouts.
            Tuple format is (connect, read, write, pool)
            If not provided, defaults to connect=5s, read=300s, write=300s, and pool=5s.
    Returns:
        SyncLangGraphClient: The top-level synchronous client for accessing AssistantsClient,
        ThreadsClient, RunsClient, and CronClient.

    ???+ example "Example"

        ```python
        from langgraph_sdk import get_sync_client

        # get top-level synchronous LangGraphClient
        client = get_sync_client(url="http://localhost:8123")

        # example usage: client.<model>.<method_name>()
        assistant = client.assistants.get(assistant_id="some_uuid")
        ```

    ???+ example "Skip auto-loading API key from environment:"

        ```python
        from langgraph_sdk import get_sync_client

        # Don't load API key from environment variables
        client = get_sync_client(
            url="http://localhost:8123",
            api_key=None
        )
        ```
    """

    if url is None:
        url = "http://localhost:8123"

    transport = httpx.HTTPTransport(retries=5)
    client = httpx.Client(
        base_url=url,
        transport=transport,
        timeout=(
            httpx.Timeout(timeout)  # type: ignore[arg-type]
            if timeout is not None
            else httpx.Timeout(connect=5, read=300, write=300, pool=5)
        ),
        headers=_get_headers(api_key, headers),
    )
    return SyncLangGraphClient(client)


class SyncLangGraphClient:
    """Synchronous client for interacting with the LangGraph API.

    This class provides synchronous access to LangGraph API endpoints for managing
    assistants, threads, runs, cron jobs, and data storage.

    ???+ example "Example"

        ```python
        client = get_sync_client(url="http://localhost:2024")
        assistant = client.assistants.get("asst_123")
        ```
    """

    def __init__(self, client: httpx.Client) -> None:
        self.http = SyncHttpClient(client)
        self.assistants = SyncAssistantsClient(self.http)
        self.threads = SyncThreadsClient(self.http)
        self.runs = SyncRunsClient(self.http)
        self.crons = SyncCronClient(self.http)
        self.store = SyncStoreClient(self.http)

    def __enter__(self) -> SyncLangGraphClient:
        """Enter the sync context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the sync context manager."""
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if hasattr(self, "http"):
            self.http.client.close()
