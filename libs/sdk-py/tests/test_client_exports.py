"""Test that all expected symbols are exported from langgraph_sdk.client.

This test ensures backwards compatibility during refactoring.
"""

import httpx

from langgraph_sdk import get_client as public_get_client
from langgraph_sdk import get_sync_client as public_get_sync_client
from langgraph_sdk.client import (
    AssistantsClient,
    CronClient,
    HttpClient,
    LangGraphClient,
    RunsClient,
    StoreClient,
    SyncAssistantsClient,
    SyncCronClient,
    SyncHttpClient,
    SyncLangGraphClient,
    SyncRunsClient,
    SyncStoreClient,
    SyncThreadsClient,
    ThreadsClient,
    _adecode_json,
    _aencode_json,
    _decode_json,
    _encode_json,
    configure_loopback_transports,
    get_client,
    get_sync_client,
)


def test_client_exports():
    """Verify all expected symbols can be imported from langgraph_sdk.client."""
    # Factory functions (public API)
    assert callable(get_client)
    assert callable(get_sync_client)

    # Top-level client classes
    assert LangGraphClient is not None
    assert SyncLangGraphClient is not None

    # HTTP client classes
    assert HttpClient is not None
    assert SyncHttpClient is not None

    # Resource client classes - Async
    assert AssistantsClient is not None
    assert ThreadsClient is not None
    assert RunsClient is not None
    assert CronClient is not None
    assert StoreClient is not None

    # Resource client classes - Sync
    assert SyncAssistantsClient is not None
    assert SyncThreadsClient is not None
    assert SyncRunsClient is not None
    assert SyncCronClient is not None
    assert SyncStoreClient is not None

    # Internal utilities (used by tests)
    assert callable(_aencode_json)
    assert callable(_adecode_json)

    # Sync JSON utilities (might be used internally)
    assert callable(_encode_json)
    assert callable(_decode_json)

    # Loopback transport configuration (used by langgraph-api)
    assert callable(configure_loopback_transports)


def test_public_api_exports():
    """Verify public API exports from langgraph_sdk package."""
    assert callable(public_get_client)
    assert callable(public_get_sync_client)


def test_client_instantiation():
    """Verify that we can instantiate clients."""
    # Test async client instantiation
    async_http = httpx.AsyncClient(base_url="http://test.example.com")
    async_client = HttpClient(async_http)
    assert async_client is not None

    # Test sync client instantiation
    sync_http = httpx.Client(base_url="http://test.example.com")
    sync_client = SyncHttpClient(sync_http)
    assert sync_client is not None
