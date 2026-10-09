"""Integration tests for RemoteGraph v3 streaming.

Tests the end-to-end wiring: RemoteGraph -> langgraph_sdk client.threads.stream(...) ->
docker-running langgraph-api -> SSE projections -> adapter classes.

Run with: pytest tests/integration/test_remote_graph_v3.py -m integration
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.pregel.remote import RemoteGraph
from langgraph.types import Command

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
    """tools_agent completes without interrupt; ``await stream.output`` drives
    the run to terminal via the lifecycle watcher (no explicit event iteration
    needed — the SSE subscription stays open by design after run completion)."""
    async with await remote_tools_agent.astream_events(
        _TOOLS_AGENT_INPUT,
        version="v3",
    ) as stream:
        output = await stream.output()
        assert output is not None
        assert (await stream.interrupted()) is False


async def test_async_interrupt_path_surfaces_interrupts(
    remote_agent: RemoteGraph,
) -> None:
    """agent graph hits ask_human; interrupted must be True with >= 1 interrupt.

    Note: interrupts pause the run but DON'T resolve `_run_done` (only
    `completed` / `failed` lifecycle phases do), so `await stream.output()`
    would hang. The adapter doesn't expose `interleave()` on the async
    side (mirrors local `AsyncGraphRunStream`), so drain the `values`
    projection directly until the run reports it is interrupted.
    """
    async with await remote_agent.astream_events(
        _AGENT_INPUT,
        version="v3",
    ) as stream:
        async for _ in stream.values:
            if await stream.interrupted():
                break
        assert (await stream.interrupted()) is True
        interrupts = await stream.interrupts()
        assert len(interrupts) >= 1


async def test_async_resume_after_interrupt(remote_agent: RemoteGraph) -> None:
    """Interrupt the agent at ask_human, then resume the SAME thread with
    `Command(resume=...)`.

    Validates the v3 resume path end-to-end. The client sends the raw resume
    value as `input` (not a serialized Command); the server detects the
    thread's pending interrupt from persisted state — which survives the first
    session's close — and wraps it as `Command(resume=...)`, driving the run
    past `ask_human` to completion (the graph interrupts only once).
    """
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # First session: drive until the agent pauses at the ask_human interrupt.
    async with await remote_agent.astream_events(
        _AGENT_INPUT,
        config=config,
        version="v3",
    ) as stream:
        async for _ in stream.values:
            if await stream.interrupted():
                break
        assert (await stream.interrupted()) is True

    # Second session on the same thread: resume with the human's answer. The
    # run continues past ask_human to completion with no further interrupt.
    async with await remote_agent.astream_events(
        Command(resume="yes"),
        config=config,
        version="v3",
    ) as stream:
        output = await stream.output()
        assert output is not None
        assert (await stream.interrupted()) is False


def test_sync_happy_path_yields_output(remote_tools_agent: RemoteGraph) -> None:
    """Sync stream: tools_agent completes; ``stream.output`` (sync property)
    blocks until terminal."""
    with remote_tools_agent.stream_events(
        _TOOLS_AGENT_INPUT,
        version="v3",
    ) as stream:
        output = stream.output
        assert output is not None
        assert stream.interrupted is False


async def test_abort_mid_run_cancels_server_side(
    remote_tools_agent: RemoteGraph,
) -> None:
    """Abort immediately after run.start; reaching the end without exception
    confirms abort + __aexit__ cleanup worked."""
    async with await remote_tools_agent.astream_events(
        _TOOLS_AGENT_INPUT,
        version="v3",
    ) as stream:
        await stream.abort()
    # Reaching here without unhandled exceptions confirms abort + __aexit__ succeeded.
