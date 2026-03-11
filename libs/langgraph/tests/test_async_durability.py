from dataclasses import dataclass, field
import asyncio
from operator import add
from typing import Any, Annotated, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class DurabilityRunStats:
    pending_checkpoint_tasks: list[int] = field(default_factory=list)
    active_put_counts: list[int] = field(default_factory=list)
    saw_active_put_during_node: bool = False
    started_puts: int = 0
    finished_puts: int = 0
    max_concurrent_puts: int = 0
    final_counter: int | None = None


class RecordingSlowInMemorySaver(InMemorySaver):
    """In-memory saver that records overlapping async checkpoint writes."""

    def __init__(self, delay_s: float, stats: DurabilityRunStats) -> None:
        super().__init__()
        self.delay_s = delay_s
        self.stats = stats
        self.active_puts = 0

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        self.stats.started_puts += 1
        self.active_puts += 1
        self.stats.max_concurrent_puts = max(
            self.stats.max_concurrent_puts, self.active_puts
        )
        try:
            await asyncio.sleep(self.delay_s)
            return await super().aput(config, checkpoint, metadata, new_versions)
        finally:
            self.active_puts -= 1
            self.stats.finished_puts += 1


class LoopState(TypedDict):
    counter: int
    samples: Annotated[list[int], add]


def count_checkpoint_tasks() -> int:
    return sum(
        1
        for task in asyncio.all_tasks()
        if "_checkpointer_put_after_previous" in task.get_coro().__qualname__
        and not task.done()
    )


async def run_counter_graph(
    *,
    durability: str,
    delay_s: float = 0.05,
    steps: int = 60,
    sample_every: int = 5,
) -> DurabilityRunStats:
    stats = DurabilityRunStats()
    saver = RecordingSlowInMemorySaver(delay_s=delay_s, stats=stats)
    config = {"configurable": {"thread_id": f"durability-{durability}-{steps}-{delay_s}"}}

    async def increment(state: LoopState) -> LoopState:
        step = state["counter"]
        if saver.active_puts > 0:
            stats.saw_active_put_during_node = True
        if step > 0 and step % sample_every == 0:
            stats.pending_checkpoint_tasks.append(count_checkpoint_tasks())
            stats.active_put_counts.append(saver.active_puts)
        return {"counter": step + 1, "samples": []}

    def route(state: LoopState) -> str:
        return "loop" if state["counter"] < steps else "done"

    graph = StateGraph(LoopState)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_conditional_edges("increment", route, {"loop": "increment", "done": END})
    app = graph.compile(checkpointer=saver)

    async for _ in app.astream(
        {"counter": 0, "samples": []},
        config,
        durability=durability,
    ):
        pass

    snapshot = await app.aget_state(config)
    stats.final_counter = snapshot.values["counter"]
    return stats


def assert_checkpoint_run_flushed(
    stats: DurabilityRunStats, *, expected_counter: int
) -> None:
    assert stats.started_puts > 0
    assert stats.started_puts == stats.finished_puts
    assert stats.final_counter == expected_counter


async def test_async_durability_caps_checkpoint_task_backlog() -> None:
    stats = await run_counter_graph(
        durability="async",
        delay_s=0.05,
        steps=80,
        sample_every=5,
    )

    assert stats.pending_checkpoint_tasks
    assert max(stats.pending_checkpoint_tasks) <= 2, stats.pending_checkpoint_tasks
    assert_checkpoint_run_flushed(stats, expected_counter=80)


async def test_async_durability_keeps_single_superstep_overlap() -> None:
    stats = await run_counter_graph(
        durability="async",
        delay_s=0.05,
        steps=50,
        sample_every=5,
    )

    assert stats.saw_active_put_during_node
    assert max(stats.active_put_counts, default=0) == 1
    assert stats.max_concurrent_puts == 1


async def test_sync_durability_waits_for_checkpoint_completion() -> None:
    stats = await run_counter_graph(
        durability="sync",
        delay_s=0.03,
        steps=30,
        sample_every=3,
    )

    assert stats.pending_checkpoint_tasks
    assert max(stats.pending_checkpoint_tasks) <= 1, stats.pending_checkpoint_tasks
    assert not stats.saw_active_put_during_node
    assert max(stats.active_put_counts, default=0) == 0
    assert stats.max_concurrent_puts == 1
    assert_checkpoint_run_flushed(stats, expected_counter=30)


async def test_async_durability_flushes_checkpoint_before_returning() -> None:
    stats = await run_counter_graph(
        durability="async",
        delay_s=0.04,
        steps=40,
        sample_every=4,
    )

    assert_checkpoint_run_flushed(stats, expected_counter=40)
    assert stats.pending_checkpoint_tasks[-1] <= 2


async def test_exit_durability_defers_checkpoint_work_until_completion() -> None:
    stats = await run_counter_graph(
        durability="exit",
        delay_s=0.03,
        steps=30,
        sample_every=3,
    )

    assert stats.pending_checkpoint_tasks
    assert max(stats.pending_checkpoint_tasks) == 0, stats.pending_checkpoint_tasks
    assert not stats.saw_active_put_during_node
    assert max(stats.active_put_counts, default=0) == 0
    assert_checkpoint_run_flushed(stats, expected_counter=30)
