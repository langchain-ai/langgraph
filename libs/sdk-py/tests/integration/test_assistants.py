"""`AssistantsClient` against the integration API.

Covers the CRUD round-trip (create / get / update / delete), search by
metadata, and the graph introspection helpers (`get_graph`,
`get_schemas`). Both async and sync.
"""

from __future__ import annotations

import pytest

from .conftest import ASSISTANT_ID

pytestmark = pytest.mark.integration


def _async_assistants(raw):
    from langgraph_sdk._async.assistants import AssistantsClient
    from langgraph_sdk._async.http import HttpClient

    return AssistantsClient(HttpClient(raw))


def _sync_assistants(raw):
    from langgraph_sdk._sync.assistants import SyncAssistantsClient
    from langgraph_sdk._sync.http import SyncHttpClient

    return SyncAssistantsClient(SyncHttpClient(raw))


async def test_assistants_crud_async(async_threads) -> None:
    _, raw = async_threads
    client = _async_assistants(raw)
    created = await client.create(
        graph_id=ASSISTANT_ID,
        metadata={"suite": "integration", "label": "crud-async"},
        name="crud-async",
    )
    aid = created["assistant_id"]
    try:
        fetched = await client.get(aid)
        assert fetched["assistant_id"] == aid
        assert fetched["graph_id"] == ASSISTANT_ID

        updated = await client.update(
            aid, metadata={"suite": "integration", "label": "crud-async-updated"}
        )
        assert updated["metadata"]["label"] == "crud-async-updated"

        results = await client.search(metadata={"label": "crud-async-updated"})
        assert any(a["assistant_id"] == aid for a in results)
    finally:
        await client.delete(aid)


def test_assistants_crud_sync(sync_threads) -> None:
    _, raw = sync_threads
    client = _sync_assistants(raw)
    created = client.create(
        graph_id=ASSISTANT_ID,
        metadata={"suite": "integration", "label": "crud-sync"},
        name="crud-sync",
    )
    aid = created["assistant_id"]
    try:
        fetched = client.get(aid)
        assert fetched["assistant_id"] == aid
        assert fetched["graph_id"] == ASSISTANT_ID

        updated = client.update(
            aid, metadata={"suite": "integration", "label": "crud-sync-updated"}
        )
        assert updated["metadata"]["label"] == "crud-sync-updated"

        results = client.search(metadata={"label": "crud-sync-updated"})
        assert any(a["assistant_id"] == aid for a in results)
    finally:
        client.delete(aid)


async def test_assistants_graph_introspection_async(async_threads) -> None:
    _, raw = async_threads
    client = _async_assistants(raw)
    # Introspection endpoints require a UUID. langgraph-api auto-creates
    # one assistant per registered graph on startup; look it up by graph_id.
    matches = await client.search(graph_id=ASSISTANT_ID, limit=1)
    assert matches, f"no auto-created assistant for graph_id={ASSISTANT_ID!r}"
    aid = matches[0]["assistant_id"]

    graph = await client.get_graph(aid)
    node_ids = [n["id"] for n in graph.get("nodes", [])]
    assert "stream_message" in node_ids
    assert "ask_human" in node_ids

    graph_xray = await client.get_graph(aid, xray=True)
    assert "nodes" in graph_xray and "edges" in graph_xray

    schemas = await client.get_schemas(aid)
    # Just verify the shape rather than exact field names (server-side
    # schema generation may evolve).
    assert "state_schema" in schemas


def test_assistants_graph_introspection_sync(sync_threads) -> None:
    _, raw = sync_threads
    client = _sync_assistants(raw)
    matches = client.search(graph_id=ASSISTANT_ID, limit=1)
    assert matches, f"no auto-created assistant for graph_id={ASSISTANT_ID!r}"
    aid = matches[0]["assistant_id"]

    graph = client.get_graph(aid)
    node_ids = [n["id"] for n in graph.get("nodes", [])]
    assert "stream_message" in node_ids
    assert "ask_human" in node_ids

    graph_xray = client.get_graph(aid, xray=True)
    assert "nodes" in graph_xray and "edges" in graph_xray

    schemas = client.get_schemas(aid)
    assert "state_schema" in schemas
