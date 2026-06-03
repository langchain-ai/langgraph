"""`thread.subgraphs` discovery against `agent` and `deep_agent`.

`deep_agent` uses `FakeMessagesListChatModel` for both supervisor and
researcher, so this suite is hermetic (no LLM API key required).
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from .conftest import ASSISTANT_ID, DEEP_AGENT_ASSISTANT_ID

pytestmark = pytest.mark.integration


async def _collect_subgraphs_async(thread) -> list:
    return [h async for h in thread.subgraphs]


def _collect_subgraphs_sync(thread) -> list:
    return list(thread.subgraphs)


async def test_subgraphs_agent_async(async_threads) -> None:
    """Plain nested `StateGraph.invoke` does not produce a scoped child handle."""
    threads, _ = async_threads
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        await thread.run.start(input={"messages": [], "value": "init", "items": []})
        handles = await _collect_subgraphs_async(thread)
        # Documented behavior: plain nested invokes do not show up as scoped
        # child handles; the canonical signal is `create_deep_agent`.
        assert handles == []


def test_subgraphs_agent_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        thread.run.start(input={"messages": [], "value": "init", "items": []})
        handles = _collect_subgraphs_sync(thread)
        assert handles == []


async def test_subgraphs_deep_agent_async(async_threads) -> None:
    threads, _ = async_threads
    async with threads.stream(assistant_id=DEEP_AGENT_ASSISTANT_ID) as thread:
        # Subscribe before / during the run so child-namespace lifecycle events
        # are not missed when the graph finishes quickly.
        collect = asyncio.create_task(_collect_subgraphs_async(thread))
        await thread.run.start(
            input={"messages": [{"role": "user", "content": "research the v3 spec"}]},
        )
        handles = await collect
        assert handles, "deep_agent should produce at least one direct-child handle"


def test_subgraphs_deep_agent_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=DEEP_AGENT_ASSISTANT_ID) as thread:
        handles: list = []
        error: list[BaseException] = []

        def collect() -> None:
            try:
                handles.extend(_collect_subgraphs_sync(thread))
            except BaseException as exc:
                error.append(exc)

        worker = threading.Thread(target=collect)
        worker.start()
        try:
            thread.run.start(
                input={
                    "messages": [{"role": "user", "content": "research the v3 spec"}]
                },
            )
            worker.join(timeout=30)
        finally:
            if worker.is_alive():
                worker.join(timeout=1)
        if error:
            raise error[0]
        assert handles, "deep_agent should produce at least one direct-child handle"
