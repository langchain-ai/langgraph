"""`thread.subgraphs` discovery against `agent` and `deep_agent`.

`deep_agent` uses `FakeMessagesListChatModel` for both supervisor and
researcher, so this suite is hermetic (no LLM API key required).
"""

from __future__ import annotations

import pytest

from .conftest import ASSISTANT_ID, DEEP_AGENT_ASSISTANT_ID

pytestmark = pytest.mark.integration


async def test_subgraphs_agent_async(async_threads) -> None:
    """Plain nested `StateGraph.invoke` does not produce a scoped child handle."""
    threads, _ = async_threads
    async with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        await thread.run.start(input={"messages": [], "value": "init", "items": []})
        handles = [h async for h in thread.subgraphs]
        # Documented behavior: plain nested invokes do not show up as scoped
        # child handles; the canonical signal is `create_deep_agent`.
        assert handles == []


def test_subgraphs_agent_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=ASSISTANT_ID) as thread:
        thread.run.start(input={"messages": [], "value": "init", "items": []})
        handles = list(thread.subgraphs)
        assert handles == []


async def test_subgraphs_deep_agent_async(async_threads) -> None:
    threads, _ = async_threads
    async with threads.stream(assistant_id=DEEP_AGENT_ASSISTANT_ID) as thread:
        await thread.run.start(
            input={"messages": [{"role": "user", "content": "research the v3 spec"}]},
        )
        handles = [h async for h in thread.subgraphs]
        assert handles, "deep_agent should produce at least one direct-child handle"


def test_subgraphs_deep_agent_sync(sync_threads) -> None:
    threads, _ = sync_threads
    with threads.stream(assistant_id=DEEP_AGENT_ASSISTANT_ID) as thread:
        thread.run.start(
            input={"messages": [{"role": "user", "content": "research the v3 spec"}]},
        )
        handles = list(thread.subgraphs)
        assert handles, "deep_agent should produce at least one direct-child handle"
