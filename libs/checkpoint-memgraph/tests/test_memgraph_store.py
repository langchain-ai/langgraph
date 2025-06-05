import os
import pytest
"""
from langgraph.store.memgraph import MemgraphStore
from langgraph.store.memgraph.aio import AsyncMemgraphStore

def is_memgraph_available(uri):
    try:
        from neo4j import GraphDatabase
        user = os.environ.get("MEMGRAPH_USER", "testuser123")
        password = os.environ.get("MEMGRAPH_PASSWORD", "BiggerPassword1233")
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                # Simple query to check connection
                session.run("RETURN 1")
        return True
    except Exception:
        return False

@pytest.fixture(scope="module")
def memgraph_uri():
    # Provide a Memgraph instance connection string
    return os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")

def test_memgraph_store_sync(memgraph_uri):
    if not is_memgraph_available(memgraph_uri):
        pytest.skip("Memgraph is not available")
    user = os.environ.get("MEMGRAPH_USER", "testuser123")
    password = os.environ.get("MEMGRAPH_PASSWORD", "BiggerPassword1233")
    with MemgraphStore.from_conn_string(memgraph_uri, user=user, password=password) as store:
        store.setup()
        # put
        store.put(("testuser",), "key1", {"foo": "bar"})
        item = store.get(("testuser",), "key1")
        assert item is not None
        assert item.value["foo"] == "bar"
        # search
        results = store.search(("testuser",), query="bar", limit=5)
        assert len(results) > 0
        # delete
        store.delete(("testuser",), "key1")
        item2 = store.get(("testuser",), "key1")
        assert item2 is None

@pytest.mark.asyncio
async def test_memgraph_store_async(memgraph_uri):
    if not is_memgraph_available(memgraph_uri):
        pytest.skip("Memgraph is not available")
    user = os.environ.get("MEMGRAPH_USER", "testuser123")
    password = os.environ.get("MEMGRAPH_PASSWORD", "BiggerPassword1233")
    async with AsyncMemgraphStore.from_conn_string(memgraph_uri, user=user, password=password) as store:
        await store.setup()
        await store.aput(("asyncuser",), "akey", {"hello": "async"})
        got = await store.aget(("asyncuser",), "akey")
        assert got is not None
        assert got.value["hello"] == "async"
        # search
        results = await store.asearch(("asyncuser",), query="async", limit=3)
        assert len(results) > 0
        # delete
        await store.adelete(("asyncuser",), "akey")
        gone = await store.aget(("asyncuser",), "akey")
        assert gone is None
"""