import os
import pytest
from langchain_core.runnables import RunnableConfig
import asyncio
from langgraph.checkpoint.memgraph import MemgraphSaver
from langgraph.checkpoint.memgraph.aio import AsyncMemgraphSaver

@pytest.fixture(scope="module")
def memgraph_uri():
    # Provide a Memgraph instance connection string
    return os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")

def is_memgraph_available(uri):
    """Check if Memgraph is available at the given URI."""
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

@pytest.mark.parametrize("cls", [MemgraphSaver])
def test_memgraph_checkpoint_sync(memgraph_uri, cls):
    user = os.environ.get("MEMGRAPH_USER", "testuser123")
    password = os.environ.get("MEMGRAPH_PASSWORD", "BiggerPassword1233")
    with cls.from_conn_string(memgraph_uri, user=user, password=password) as saver:
        saver.setup()
        config = {"configurable": {"thread_id": "T1", "checkpoint_ns": "main"}}
        checkpoint = {"id": "cp1", "data": {"hello": "world"}}
        metadata = {"meta": "test"}
        saver.put(config, checkpoint, metadata, {})
        cp_tuple = saver.get_tuple(config)
        assert cp_tuple is not None
        assert cp_tuple.checkpoint["data"]["hello"] == "world"
        # list
        cpts = list(saver.list(config))
        assert len(cpts) == 1
        assert cpts[0].checkpoint["id"] == "cp1"

@pytest.mark.asyncio
async def test_memgraph_checkpoint_async(memgraph_uri):
    user = os.environ.get("MEMGRAPH_USER", "testuser123")
    password = os.environ.get("MEMGRAPH_PASSWORD", "BiggerPassword1233")
    async with AsyncMemgraphSaver.from_conn_string(memgraph_uri, user=user, password=password) as saver:
        await saver.setup()
        config = {"configurable": {"thread_id": "T2", "checkpoint_ns": "main"}}
        checkpoint = {"id": "cp2", "payload": 123}
        metadata = {"source": "test"}
        await saver.aput(config, checkpoint, metadata, {})
        cp_tuple = await saver.aget_tuple(config)
        assert cp_tuple is not None
        assert cp_tuple.checkpoint["payload"] == 123
        # list
        results = []
        async for cpt in saver.alist(config):
            results.append(cpt)
        assert len(results) == 1
        assert results[0].checkpoint["id"] == "cp2"
