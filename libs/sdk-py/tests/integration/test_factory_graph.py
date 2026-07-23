"""Factory-graph execution regression test.

Unlike the other integration graphs (all pre-compiled), `factory_agent` is a
graph *factory*, so executing a run against it drives the server's graph-factory
code path. That path regressed in langgraph 1.2.3: a leaked `__pregel_runtime`
(an SDK `_ExecutionRuntime`) survived `ensure_config`'s configurable-merge into
`astream`, which then raised
`AttributeError: '_ExecutionRuntime' object has no attribute 'control'`. A
successful `runs.wait` here proves the factory path executes end to end.
"""

from __future__ import annotations

import pytest

from .conftest import FACTORY_ASSISTANT_ID

pytestmark = pytest.mark.integration


def _async_runs(raw):
    from langgraph_sdk._async.http import HttpClient
    from langgraph_sdk._async.runs import RunsClient

    return RunsClient(HttpClient(raw))


def _sync_runs(raw):
    from langgraph_sdk._sync.http import SyncHttpClient
    from langgraph_sdk._sync.runs import SyncRunsClient

    return SyncRunsClient(SyncHttpClient(raw))


async def test_factory_graph_executes_async(async_threads) -> None:
    """A run against a factory graph completes and echoes the input."""
    threads, raw = async_threads
    runs = _async_runs(raw)
    thread = await threads.create(
        metadata={"suite": "integration", "label": "factory-async"}
    )
    tid = thread["thread_id"]
    try:
        result = await runs.wait(tid, FACTORY_ASSISTANT_ID, input={"text": "hi"})
        assert isinstance(result, dict), result
        assert result.get("text") == "hi echoed"
        assert result.get("access_context") == "threads.create_run"
        assert result.get("is_for_execution") is True
    finally:
        await threads.delete(tid)


def test_factory_graph_executes_sync(sync_threads) -> None:
    threads, raw = sync_threads
    runs = _sync_runs(raw)
    thread = threads.create(metadata={"suite": "integration", "label": "factory-sync"})
    tid = thread["thread_id"]
    try:
        result = runs.wait(tid, FACTORY_ASSISTANT_ID, input={"text": "hi"})
        assert isinstance(result, dict), result
        assert result.get("text") == "hi echoed"
        assert result.get("access_context") == "threads.create_run"
        assert result.get("is_for_execution") is True
    finally:
        threads.delete(tid)
