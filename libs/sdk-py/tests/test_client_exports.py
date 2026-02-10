"""Test that all expected symbols are exported from langgraph_sdk.client.

This test ensures backwards compatibility during refactoring.
"""


def test_client_exports():
    """Verify all expected symbols can be imported from langgraph_sdk.client."""
    # Factory functions (public API)
    from langgraph_sdk.client import get_client, get_sync_client

    assert callable(get_client)
    assert callable(get_sync_client)

    # Top-level client classes
    from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient

    assert LangGraphClient is not None
    assert SyncLangGraphClient is not None

    # HTTP client classes
    from langgraph_sdk.client import HttpClient, SyncHttpClient

    assert HttpClient is not None
    assert SyncHttpClient is not None

    # Resource client classes - Async
    from langgraph_sdk.client import (
        AssistantsClient,
        CronClient,
        RunsClient,
        StoreClient,
        ThreadsClient,
    )

    assert AssistantsClient is not None
    assert ThreadsClient is not None
    assert RunsClient is not None
    assert CronClient is not None
    assert StoreClient is not None

    # Resource client classes - Sync
    from langgraph_sdk.client import (
        SyncAssistantsClient,
        SyncCronClient,
        SyncRunsClient,
        SyncStoreClient,
        SyncThreadsClient,
    )

    assert SyncAssistantsClient is not None
    assert SyncThreadsClient is not None
    assert SyncRunsClient is not None
    assert SyncCronClient is not None
    assert SyncStoreClient is not None

    # Internal utilities (used by tests)
    from langgraph_sdk.client import _adecode_json, _aencode_json

    assert callable(_aencode_json)
    assert callable(_adecode_json)

    # Sync JSON utilities (might be used internally)
    from langgraph_sdk.client import _decode_json, _encode_json

    assert callable(_encode_json)
    assert callable(_decode_json)


def test_public_api_exports():
    """Verify public API exports from langgraph_sdk package."""
    from langgraph_sdk import get_client, get_sync_client

    assert callable(get_client)
    assert callable(get_sync_client)


def test_client_instantiation():
    """Verify that we can instantiate clients."""
    import httpx

    from langgraph_sdk.client import HttpClient, SyncHttpClient

    # Test async client instantiation
    async_http = httpx.AsyncClient(base_url="http://test.example.com")
    async_client = HttpClient(async_http)
    assert async_client is not None

    # Test sync client instantiation
    sync_http = httpx.Client(base_url="http://test.example.com")
    sync_client = SyncHttpClient(sync_http)
    assert sync_client is not None
