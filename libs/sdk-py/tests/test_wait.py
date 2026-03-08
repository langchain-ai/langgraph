from __future__ import annotations

import httpx
import pytest

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.runs import RunsClient as AsyncRunsClient
from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.runs import SyncRunsClient
from langgraph_sdk.schema import GraphOutput


def test_sync_wait_v1():
    """Verify that version='v1' (default) returns the raw dict unchanged."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "completed", "__interrupt__": [{"value": "test", "id": "123"}]},
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        runs = SyncRunsClient(http=SyncHttpClient(client))
        result = runs.wait(
            thread_id="thread-id",
            assistant_id="agent",
            version="v1",
        )

    assert isinstance(result, dict)
    assert result == {
        "status": "completed",
        "__interrupt__": [{"value": "test", "id": "123"}],
    }


def test_sync_wait_v2_single():
    """Verify that version='v2' extracts __interrupt__ and wraps in GraphOutput."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "completed", "__interrupt__": [{"value": "test", "id": "123"}]},
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        runs = SyncRunsClient(http=SyncHttpClient(client))
        result = runs.wait(
            thread_id="thread-id",
            assistant_id="agent",
            version="v2",
        )

    assert isinstance(result, GraphOutput)
    assert result.value == {"status": "completed"}
    assert result.interrupts == ({"value": "test", "id": "123"},)

    # Test dictionary backward compatibility
    with pytest.warns(DeprecationWarning, match="Accessing GraphOutput via `result\\[key\\]` is deprecated"):
        assert result["status"] == "completed"
    with pytest.warns(DeprecationWarning, match="Accessing GraphOutput via `result\\[key\\]` is deprecated"):
        assert result["__interrupt__"] == ({"value": "test", "id": "123"},)
    with pytest.warns(DeprecationWarning, match="Accessing GraphOutput via `key in result` is deprecated"):
        assert "__interrupt__" in result
    with pytest.warns(DeprecationWarning, match="Accessing GraphOutput via `key in result` is deprecated"):
        assert "status" in result


def test_sync_wait_v2_list():
    """Verify that version='v2' handles a list response from a batch run."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"status": "completed", "__interrupt__": [{"value": "test", "id": "123"}]},
                {"status": "pending"}
            ],
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        runs = SyncRunsClient(http=SyncHttpClient(client))
        result = runs.wait(
            thread_id=None,
            assistant_id="agent",
            version="v2",
        )

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], GraphOutput)
    assert result[0].value == {"status": "completed"}
    assert result[0].interrupts == ({"value": "test", "id": "123"},)
    
    assert isinstance(result[1], GraphOutput)
    assert result[1].value == {"status": "pending"}
    assert result[1].interrupts == ()


@pytest.mark.asyncio
async def test_async_wait_v1():
    """Verify async version='v1' returns original dict."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "completed", "__interrupt__": [{"value": "test", "id": "123"}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        runs = AsyncRunsClient(http=HttpClient(client))
        result = await runs.wait(
            thread_id="thread-id",
            assistant_id="agent",
            version="v1",
        )

    assert isinstance(result, dict)
    assert "__interrupt__" in result


@pytest.mark.asyncio
async def test_async_wait_v2_multiple_interrupts():
    """Verify async version='v2' correctly parses multiple interrupts."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "interrupted", 
                "__interrupt__": [
                    {"value": "question 1", "id": "1"},
                    {"value": "question 2", "id": "2"}
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        runs = AsyncRunsClient(http=HttpClient(client))
        result = await runs.wait(
            thread_id="thread-id",
            assistant_id="agent",
            version="v2",
        )

    assert isinstance(result, GraphOutput)
    assert result.value == {"status": "interrupted"}
    assert len(result.interrupts) == 2
    assert result.interrupts[0] == {"value": "question 1", "id": "1"}
    assert result.interrupts[1] == {"value": "question 2", "id": "2"}
