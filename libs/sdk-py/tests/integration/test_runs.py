"""`RunsClient` non-streaming surface.

`cancel` is covered in `test_cancel.py`. This file covers create / get /
list / wait. The canonical `agent` graph interrupts at `ask_human`, so a
plain `runs.create` lands in the `interrupted` state. We use
`interrupt_before=["ask_human"]` so the run pauses before the interrupting
node and reaches a deterministic non-success terminal.
"""

from __future__ import annotations

import pytest

from .conftest import ASSISTANT_ID

pytestmark = pytest.mark.integration


def _async_runs(raw):
    from langgraph_sdk._async.http import HttpClient
    from langgraph_sdk._async.runs import RunsClient

    return RunsClient(HttpClient(raw))


def _sync_runs(raw):
    from langgraph_sdk._sync.http import SyncHttpClient
    from langgraph_sdk._sync.runs import SyncRunsClient

    return SyncRunsClient(SyncHttpClient(raw))


async def test_runs_create_get_list_async(async_threads) -> None:
    threads, raw = async_threads
    runs = _async_runs(raw)
    thread = await threads.create(
        metadata={"suite": "integration", "label": "runs-async"}
    )
    tid = thread["thread_id"]
    try:
        created = await runs.create(
            tid,
            ASSISTANT_ID,
            input={"messages": [], "value": "init", "items": []},
        )
        run_id = created["run_id"]
        assert created["thread_id"] == tid

        fetched = await runs.get(tid, run_id)
        assert fetched["run_id"] == run_id

        listed = await runs.list(tid, limit=10)
        assert any(r["run_id"] == run_id for r in listed)
    finally:
        await threads.delete(tid)


def test_runs_create_get_list_sync(sync_threads) -> None:
    threads, raw = sync_threads
    runs = _sync_runs(raw)
    thread = threads.create(metadata={"suite": "integration", "label": "runs-sync"})
    tid = thread["thread_id"]
    try:
        created = runs.create(
            tid,
            ASSISTANT_ID,
            input={"messages": [], "value": "init", "items": []},
        )
        run_id = created["run_id"]
        assert created["thread_id"] == tid

        fetched = runs.get(tid, run_id)
        assert fetched["run_id"] == run_id

        listed = runs.list(tid, limit=10)
        assert any(r["run_id"] == run_id for r in listed)
    finally:
        threads.delete(tid)


async def test_runs_wait_async(async_threads) -> None:
    """`wait` blocks until the run reaches a terminal state and returns its values."""
    threads, raw = async_threads
    runs = _async_runs(raw)
    thread = await threads.create(
        metadata={"suite": "integration", "label": "wait-async"}
    )
    tid = thread["thread_id"]
    try:
        # `interrupt_before` makes the run pause before `ask_human` rather
        # than running into the dynamic `interrupt(...)` inside it; the run
        # ends up in `interrupted` status with a deterministic terminal.
        result = await runs.wait(
            tid,
            ASSISTANT_ID,
            input={"messages": [], "value": "init", "items": []},
            interrupt_before=["ask_human"],
        )
        # The result is the terminal `values` payload for this run.
        assert isinstance(result, dict)
        assert "items" in result
        assert "streamed" in result["items"]
        assert "tool" in result["items"]
    finally:
        await threads.delete(tid)


def test_runs_wait_sync(sync_threads) -> None:
    threads, raw = sync_threads
    runs = _sync_runs(raw)
    thread = threads.create(metadata={"suite": "integration", "label": "wait-sync"})
    tid = thread["thread_id"]
    try:
        result = runs.wait(
            tid,
            ASSISTANT_ID,
            input={"messages": [], "value": "init", "items": []},
            interrupt_before=["ask_human"],
        )
        assert isinstance(result, dict)
        assert "items" in result
        assert "streamed" in result["items"]
        assert "tool" in result["items"]
    finally:
        threads.delete(tid)
