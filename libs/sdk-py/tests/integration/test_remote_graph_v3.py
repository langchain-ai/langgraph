"""Integration tests for RemoteGraph v3 streaming.

Tests the end-to-end wiring: RemoteGraph -> langgraph_sdk client.threads.stream(...) ->
docker-running langgraph-api -> SSE projections -> adapter classes.

Run with: pytest tests/integration/test_remote_graph_v3.py -m integration
"""

from __future__ import annotations

import pytest

from langgraph.pregel.remote import RemoteGraph

pytestmark = pytest.mark.integration

URL = "http://localhost:2024"

# Input shapes matching what the integration graphs expect.
# `agent` graph (streaming_graph.py): AgentState has messages, value, items.
# `tools_agent` graph (tools_agent.py): create_agent graph expects messages list.
_AGENT_INPUT = {"messages": [], "value": "init", "items": []}
_TOOLS_AGENT_INPUT = {"messages": [{"role": "user", "content": "search for v3"}]}


@pytest.fixture
def remote_agent() -> RemoteGraph:
    return RemoteGraph("agent", url=URL)


@pytest.fixture
def remote_tools_agent() -> RemoteGraph:
    return RemoteGraph("tools_agent", url=URL)


async def test_async_happy_path_yields_output(remote_tools_agent: RemoteGraph) -> None:
    """tools_agent completes without interrupt; output must be non-None."""
    async with remote_tools_agent.astream_events(
        _TOOLS_AGENT_INPUT,
        version="v3",
    ) as stream:
        async for _ in stream:
            pass
        output = await stream.output
        assert output is not None
        assert (await stream.interrupted) is False


async def test_async_interrupt_path_surfaces_interrupts(
    remote_agent: RemoteGraph,
) -> None:
    """agent graph hits ask_human; interrupted must be True with >= 1 interrupt."""
    async with remote_agent.astream_events(
        _AGENT_INPUT,
        version="v3",
    ) as stream:
        async for _ in stream:
            # The SSE stream ends when the run pauses at the interrupt; the
            # loop exits naturally at that point.
            pass
        assert (await stream.interrupted) is True
        interrupts = await stream.interrupts
        assert len(interrupts) >= 1


def test_sync_happy_path_yields_output(remote_tools_agent: RemoteGraph) -> None:
    """Sync stream: tools_agent completes; output must be non-None."""
    with remote_tools_agent.stream_events(
        _TOOLS_AGENT_INPUT,
        version="v3",
    ) as stream:
        for _ in stream:
            pass
        assert stream.output is not None
        assert stream.interrupted is False


async def test_abort_mid_run_cancels_server_side(remote_tools_agent: RemoteGraph) -> None:
    """Abort after first event; reaching end without exception = abort + cleanup worked."""
    async with remote_tools_agent.astream_events(
        _TOOLS_AGENT_INPUT,
        version="v3",
    ) as stream:
        async for _ in stream:
            break
        await stream.abort()
    # Reaching here without unhandled exceptions confirms abort + __aexit__ succeeded.
