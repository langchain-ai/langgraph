"""Tests verifying that the bounded checkpoint queue limits live checkpoint items.

Complements test_checkpoint_queue.py by directly asserting that the number of
live _CheckpointWriteItem instances stays bounded during execution with many
steps, using gc.get_objects() counting.
"""

import asyncio
import gc
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
from langgraph.pregel._loop import CHECKPOINT_WRITE_BUFFER_SIZE, _CheckpointWriteItem

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SamplingCheckpointer(InMemorySaver):
    """InMemorySaver that periodically samples live _CheckpointWriteItem count."""

    def __init__(self, *, delay: float = 0.05) -> None:
        super().__init__()
        self.delay = delay
        self._call_count = 0
        self.peak_live_items = 0

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._call_count += 1
        if self._call_count % 5 == 0:
            gc.collect()
            live = sum(
                1 for obj in gc.get_objects() if type(obj) is _CheckpointWriteItem
            )
            if live > self.peak_live_items:
                self.peak_live_items = live
        if self.delay:
            await asyncio.sleep(self.delay)
        return await super().aput(config, checkpoint, metadata, new_versions)


def build_linear_graph(checkpointer: InMemorySaver | None, num_steps: int = 1) -> Any:
    """Build a linear chain of `num_steps` nodes, each appending 'x' to state."""
    builder = StateGraph(Annotated[str, operator.add])
    prev = START
    for i in range(num_steps):
        name = f"step_{i}"
        builder.add_node(name, lambda s: "x")
        builder.add_edge(prev, name)
        prev = name
    builder.add_edge(prev, END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_bounded_queue_limits_live_checkpoint_items() -> None:
    """Peak live _CheckpointWriteItem count stays bounded with 50 steps."""
    checkpointer = SamplingCheckpointer(delay=0.05)
    graph = build_linear_graph(checkpointer, num_steps=50)
    config = {"configurable": {"thread_id": "memory-bound-1"}}

    result = await asyncio.wait_for(
        graph.ainvoke("a", config, durability="async"), timeout=15
    )

    assert result == "a" + "x" * 50
    # 50 nodes + initial input + final tick
    assert checkpointer._call_count == 52
    # Key assertion: queue is bounded
    assert checkpointer.peak_live_items <= CHECKPOINT_WRITE_BUFFER_SIZE + 2
    # Sanity: we actually measured something
    assert checkpointer.peak_live_items >= 1


async def test_bounded_queue_post_execution_cleanup() -> None:
    """After graph execution, zero _CheckpointWriteItem instances remain alive."""
    checkpointer = SamplingCheckpointer(delay=0.02)
    graph = build_linear_graph(checkpointer, num_steps=20)
    config = {"configurable": {"thread_id": "cleanup-1"}}

    result = await graph.ainvoke("a", config, durability="async")
    assert result == "a" + "x" * 20

    # Allow GC to clean up
    gc.collect()
    gc.collect()

    live = sum(1 for obj in gc.get_objects() if type(obj) is _CheckpointWriteItem)
    assert live == 0, f"Expected 0 live _CheckpointWriteItem instances, found {live}"
