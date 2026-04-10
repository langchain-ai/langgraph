"""Tests for the bounded queue + single writer checkpoint mechanism in AsyncPregelLoop.

Covers: happy path, FIFO ordering, backpressure, error propagation,
deadlock-free shutdown, durability modes, and no-checkpointer fallback.
"""

import asyncio
import operator
from typing import Annotated, Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.graph import END, START, StateGraph

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingCheckpointer(InMemorySaver):
    """InMemorySaver that records aput calls, with optional delay and failure."""

    def __init__(
        self,
        *,
        delay: float = 0,
        fail_after: int | None = None,
    ) -> None:
        super().__init__()
        self.delay = delay
        self.fail_after = fail_after  # fail on the Nth call (0-indexed)
        self.put_calls: list[dict[str, Any]] = []
        self._put_started = asyncio.Event()
        self._call_count = 0

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._call_count += 1
        self._put_started.set()
        if self.fail_after is not None and self._call_count > self.fail_after:
            raise ValueError("Injected aput failure")
        if self.delay:
            await asyncio.sleep(self.delay)
        result = await super().aput(config, checkpoint, metadata, new_versions)
        self.put_calls.append(
            {
                "checkpoint_id": checkpoint["id"],
                "step": metadata.get("step"),
            }
        )
        return result


class DelayThenFailCheckpointer(InMemorySaver):
    """Succeeds for `succeed_count` puts with a delay, then fails."""

    def __init__(self, *, succeed_count: int = 3, delay: float = 0.2) -> None:
        super().__init__()
        self.succeed_count = succeed_count
        self.delay = delay
        self._call_count = 0

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._call_count += 1
        if self._call_count > self.succeed_count:
            raise ValueError("Injected aput failure")
        await asyncio.sleep(self.delay)
        return await super().aput(config, checkpoint, metadata, new_versions)


def build_linear_graph(checkpointer: InMemorySaver | None, num_steps: int = 1) -> Any:
    """Build a linear chain of `num_steps` nodes, each appending 'x' to state."""
    builder = StateGraph(Annotated[str, operator.add])
    prev = START
    for i in range(num_steps):
        name = f"step_{i}"
        # Each node appends "x"
        builder.add_node(name, lambda s: "x")
        builder.add_edge(prev, name)
        prev = name
    builder.add_edge(prev, END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_queue_writer_happy_path() -> None:
    """Basic sanity: checkpoints written correctly with durability='async'."""
    checkpointer = RecordingCheckpointer(delay=0)
    graph = build_linear_graph(checkpointer, num_steps=2)
    config = {"configurable": {"thread_id": "happy-1"}}

    result = await graph.ainvoke("a", config, durability="async")
    assert result == "axx"
    # N nodes produce N+2 checkpoints: initial input + N per-step + final tick
    assert len(checkpointer.put_calls) == 4

    # Final checkpoint is retrievable
    tup = await checkpointer.aget_tuple(config)
    assert tup is not None


async def test_queue_writer_fifo_ordering() -> None:
    """Multiple checkpoints arrive at checkpointer in strict step order."""
    checkpointer = RecordingCheckpointer(delay=0.02)
    graph = build_linear_graph(checkpointer, num_steps=6)
    config = {"configurable": {"thread_id": "fifo-1"}}

    result = await graph.ainvoke("a", config, durability="async")
    assert result == "a" + "x" * 6

    steps = [c["step"] for c in checkpointer.put_calls]
    assert steps == sorted(steps), f"Steps not in FIFO order: {steps}"
    assert len(steps) == 8  # 6 nodes + initial input + final tick


async def test_queue_writer_backpressure() -> None:
    """Slow writer + many steps: main loop throttled, completes without OOM."""
    checkpointer = RecordingCheckpointer(delay=0.1)
    graph = build_linear_graph(checkpointer, num_steps=8)
    config = {"configurable": {"thread_id": "backpressure-1"}}

    result = await asyncio.wait_for(
        graph.ainvoke("a", config, durability="async"), timeout=10
    )
    assert result == "a" + "x" * 8
    assert len(checkpointer.put_calls) == 10  # 8 nodes + initial input + final tick

    steps = [c["step"] for c in checkpointer.put_calls]
    assert steps == sorted(steps)


async def test_queue_writer_aput_error_propagation() -> None:
    """Writer aput failure surfaces as exception from ainvoke and astream."""
    checkpointer = RecordingCheckpointer(fail_after=1)
    graph = build_linear_graph(checkpointer, num_steps=3)

    # ainvoke
    with pytest.raises(ValueError, match="Injected aput failure"):
        await asyncio.wait_for(
            graph.ainvoke(
                "a",
                {"configurable": {"thread_id": "err-invoke"}},
                durability="async",
            ),
            timeout=5,
        )

    # astream
    checkpointer2 = RecordingCheckpointer(fail_after=1)
    graph2 = build_linear_graph(checkpointer2, num_steps=3)
    with pytest.raises(ValueError, match="Injected aput failure"):
        async for _ in graph2.astream(
            "a",
            {"configurable": {"thread_id": "err-stream"}},
            durability="async",
        ):
            pass


async def test_queue_writer_shutdown_no_deadlock_after_writer_death() -> None:
    """When writer dies on very first aput, __aexit__ shutdown completes.

    Regression test for code review finding #1 (shutdown deadlock after writer error).
    """
    checkpointer = RecordingCheckpointer(fail_after=0)
    graph = build_linear_graph(checkpointer, num_steps=2)
    config = {"configurable": {"thread_id": "deadlock-1"}}

    with pytest.raises(ValueError, match="Injected aput failure"):
        await asyncio.wait_for(
            graph.ainvoke("a", config, durability="async"),
            timeout=5,
        )


async def test_queue_writer_error_during_backpressure() -> None:
    """Writer fails while main loop is blocked on queue.put() (backpressure).

    Regression test for code review finding #2 (race in backpressure path).
    """
    checkpointer = DelayThenFailCheckpointer(succeed_count=3, delay=0.2)
    graph = build_linear_graph(checkpointer, num_steps=8)
    config = {"configurable": {"thread_id": "bp-error-1"}}

    with pytest.raises(ValueError, match="Injected aput failure"):
        await asyncio.wait_for(
            graph.ainvoke("a", config, durability="async"),
            timeout=10,
        )


async def test_queue_writer_sync_durability() -> None:
    """durability='sync': writes complete before next step proceeds."""
    checkpointer = RecordingCheckpointer(delay=0.05)
    graph = build_linear_graph(checkpointer, num_steps=3)
    config = {"configurable": {"thread_id": "sync-1"}}

    result = await graph.ainvoke("a", config, durability="sync")
    assert result == "a" + "x" * 3
    assert len(checkpointer.put_calls) == 5  # 3 nodes + initial input + final tick


async def test_queue_writer_durability_exit() -> None:
    """durability='exit': checkpoint written only at graph exit."""
    checkpointer = RecordingCheckpointer(delay=0)
    graph = build_linear_graph(checkpointer, num_steps=2)
    config = {"configurable": {"thread_id": "exit-1"}}

    result = await graph.ainvoke("a", config, durability="exit")
    assert result == "axx"
    # Exit durability: only the final exit checkpoint is written
    assert len(checkpointer.put_calls) == 1

    # Final state is retrievable
    tup = await checkpointer.aget_tuple(config)
    assert tup is not None


async def test_queue_writer_no_checkpointer() -> None:
    """Without checkpointer, graph runs fine with no crash."""
    graph = build_linear_graph(checkpointer=None, num_steps=2)

    result = await graph.ainvoke("a")
    assert result == "axx"
