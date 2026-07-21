"""`ThreadsClient` non-streaming CRUD surface.

`stream` and `update_state` are covered elsewhere; this file covers
create / get / delete / search / copy / get_history.
"""

from __future__ import annotations

import pytest

from .conftest import ASSISTANT_ID

pytestmark = pytest.mark.integration


async def test_threads_create_get_delete_async(async_threads) -> None:
    threads, _ = async_threads
    created = await threads.create(
        metadata={"suite": "integration", "label": "crud-async"}
    )
    tid = created["thread_id"]
    try:
        fetched = await threads.get(tid)
        assert fetched["thread_id"] == tid
        assert fetched["metadata"]["label"] == "crud-async"
    finally:
        await threads.delete(tid)


def test_threads_create_get_delete_sync(sync_threads) -> None:
    threads, _ = sync_threads
    created = threads.create(metadata={"suite": "integration", "label": "crud-sync"})
    tid = created["thread_id"]
    try:
        fetched = threads.get(tid)
        assert fetched["thread_id"] == tid
        assert fetched["metadata"]["label"] == "crud-sync"
    finally:
        threads.delete(tid)


async def test_threads_search_async(async_threads) -> None:
    threads, _ = async_threads
    created = await threads.create(
        metadata={"suite": "integration", "label": "search-async"}
    )
    tid = created["thread_id"]
    try:
        results = await threads.search(metadata={"label": "search-async"}, limit=10)
        assert any(t["thread_id"] == tid for t in results)
    finally:
        await threads.delete(tid)


def test_threads_search_sync(sync_threads) -> None:
    threads, _ = sync_threads
    created = threads.create(metadata={"suite": "integration", "label": "search-sync"})
    tid = created["thread_id"]
    try:
        results = threads.search(metadata={"label": "search-sync"}, limit=10)
        assert any(t["thread_id"] == tid for t in results)
    finally:
        threads.delete(tid)


async def test_threads_copy_async(async_threads) -> None:
    threads, _ = async_threads
    src = await threads.create(
        metadata={"suite": "integration", "label": "copy-async-src"}
    )
    src_id = src["thread_id"]
    try:
        copied = await threads.copy(src_id)
        copy_id = copied["thread_id"]
        try:
            assert copy_id != src_id
        finally:
            await threads.delete(copy_id)
    finally:
        await threads.delete(src_id)


def test_threads_copy_sync(sync_threads) -> None:
    threads, _ = sync_threads
    src = threads.create(metadata={"suite": "integration", "label": "copy-sync-src"})
    src_id = src["thread_id"]
    try:
        copied = threads.copy(src_id)
        copy_id = copied["thread_id"]
        try:
            assert copy_id != src_id
        finally:
            threads.delete(copy_id)
    finally:
        threads.delete(src_id)


async def test_threads_history_after_run_async(async_threads) -> None:
    """A completed run produces at least one checkpoint in history."""
    threads, _ = async_threads
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        await thread.run.start(input={"messages": [], "value": "init", "items": []})
        async for _ in thread.values:
            if thread.interrupted:
                break
        if thread.interrupted:
            await thread.run.respond("yes")
        await thread.output
        history = await threads.get_history(thread.thread_id, limit=20)
    assert history, "expected at least one checkpoint after a completed run"


def test_threads_history_after_run_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        thread.run.start(input={"messages": [], "value": "init", "items": []})
        for _ in thread.values:
            if thread.interrupted:
                break
        if thread.interrupted:
            thread.run.respond("yes")
        _ = thread.output  # force terminal-state fetch; value unused
        history = threads.get_history(thread.thread_id, limit=20)
    assert history, "expected at least one checkpoint after a completed run"
