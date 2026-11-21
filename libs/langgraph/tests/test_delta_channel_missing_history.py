"""A `DeltaChannel` load warns about history its checkpointer no longer holds."""

import logging
from typing import Annotated, Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import _checkpoint
from langgraph.types import Command

pytestmark = pytest.mark.anyio

THREAD: RunnableConfig = {"configurable": {"thread_id": "1"}}
WARNING = "DeltaChannel 'items' of thread '1' at checkpoint"
# What the update counter in the checkpoint metadata adds to the message.
COUNTED = "supersteps wrote to it since its last snapshot"


class _KeepsNewest(InMemorySaver):
    """Keeps a thread's newest `keep` checkpoints and their writes, as a TTL
    does. With `keep=1` it also reports no parent, as a shallow saver does.
    Walks history with the base class, as a third-party saver does."""

    get_delta_channel_history = BaseCheckpointSaver.get_delta_channel_history  # type: ignore[assignment]
    aget_delta_channel_history = BaseCheckpointSaver.aget_delta_channel_history  # type: ignore[assignment]

    def __init__(self, keep: int) -> None:
        super().__init__()
        self.keep = keep

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        out = super().put(config, checkpoint, metadata, new_versions)
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        stored = self.storage[thread_id][checkpoint_ns]
        for checkpoint_id in sorted(stored)[: -self.keep]:
            del stored[checkpoint_id]
            self.writes.pop((thread_id, checkpoint_ns, checkpoint_id), None)
        return out

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        saved = super().get_tuple(config)
        if saved is not None and self.keep == 1:
            return saved._replace(parent_config=None)
        return saved


def _extend(state: list | None, writes: list[Any]) -> list:
    return [*(state or []), *(item for write in writes for item in write)]


def _graph(checkpointer: BaseCheckpointSaver) -> Any:
    class State(TypedDict):
        items: Annotated[list, DeltaChannel(_extend, snapshot_frequency=50)]

    builder = StateGraph(State)
    builder.add_node("a", lambda state: {"items": ["a"]})
    builder.add_node("b", lambda state: {"items": ["b"]})
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    return builder.compile(checkpointer=checkpointer)


@pytest.fixture
def warnings(caplog: pytest.LogCaptureFixture) -> Any:
    """Returns the messages the loader has logged so far."""
    caplog.set_level(logging.WARNING, logger=_checkpoint.logger.name)
    return lambda: [
        record.getMessage()
        for record in caplog.records
        if record.name == _checkpoint.logger.name
    ]


def test_load_from_a_saver_that_keeps_one_checkpoint_warns(
    warnings: Any,
) -> None:
    graph = _graph(_KeepsNewest(keep=1))
    graph.invoke({"items": ["in"]}, THREAD)

    values = graph.get_state(THREAD).values

    assert values == {"items": []}
    [message] = warnings()
    assert WARNING in message
    # The saver reports no parent, so no checkpoint is named as gone.
    assert "3 supersteps wrote to it since its last snapshot and 0 writes" in message
    assert "no snapshot and no write of it is stored" in message
    assert "parent chain" not in message


def test_load_after_older_checkpoints_expired_warns(warnings: Any) -> None:
    graph = _graph(_KeepsNewest(keep=2))
    graph.invoke({"items": ["in"]}, THREAD)

    values = graph.get_state(THREAD).values

    assert values == {"items": ["b"]}
    [message] = warnings()
    assert WARNING in message
    assert "on its parent chain is gone" in message
    assert "3 supersteps wrote to it since its last snapshot and 1 writes" in message


def test_run_on_a_thread_with_expired_checkpoints_warns_and_continues(
    warnings: Any,
) -> None:
    graph = _graph(_KeepsNewest(keep=2))
    graph.invoke({"items": ["in"]}, THREAD)

    out = graph.invoke({"items": ["in"]}, THREAD)

    assert out == {"items": ["b", "in", "a", "b"]}
    assert any(WARNING in message and COUNTED in message for message in warnings())


def test_update_on_a_thread_with_expired_checkpoints_warns(warnings: Any) -> None:
    graph = _graph(_KeepsNewest(keep=2))
    graph.invoke({"items": ["in"]}, THREAD)

    graph.update_state(THREAD, {"items": ["u"]})

    assert any(WARNING in message and COUNTED in message for message in warnings())


async def test_async_load_run_and_update_warn(warnings: Any) -> None:
    graph = _graph(_KeepsNewest(keep=2))
    await graph.ainvoke({"items": ["in"]}, THREAD)

    state = await graph.aget_state(THREAD)
    assert state.values == {"items": ["b"]}
    [message] = warnings()
    assert "on its parent chain is gone" in message
    assert COUNTED in message

    await graph.aupdate_state(THREAD, {"items": ["u"]})
    assert len(warnings()) == 2
    assert COUNTED in warnings()[1]

    await graph.ainvoke({"items": ["in"]}, THREAD)
    assert len(warnings()) > 2
    assert all(COUNTED in message for message in warnings())


class _OddCounter(_KeepsNewest):
    """Returns a counter in a shape the loop never writes."""

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        saved = super().get_tuple(config)
        if saved is None:
            return None
        counters: Any = {"items": None}
        return saved._replace(
            metadata={**saved.metadata, "counters_since_delta_snapshot": counters}
        )


def test_a_counter_the_check_cannot_read_does_not_fail_the_load(
    warnings: Any,
) -> None:
    graph = _graph(_OddCounter(keep=2))
    graph.invoke({"items": ["in"]}, THREAD)

    assert graph.get_state(THREAD).values == {"items": ["b"]}
    [message] = warnings()
    assert "on its parent chain is gone" in message
    assert COUNTED not in message


def test_first_input_as_a_command_update_does_not_warn(warnings: Any) -> None:
    graph = _graph(_KeepsNewest(keep=100))

    graph.invoke(Command(update={"items": ["in"]}, goto="a"), THREAD)

    # The input checkpoint has a version for `items` and nothing before it.
    assert [state.values for state in graph.get_state_history(THREAD)][-1] == {
        "items": []
    }
    assert warnings() == []


def test_complete_history_loads_without_a_warning(warnings: Any) -> None:
    graph = _graph(_KeepsNewest(keep=100))

    graph.invoke({"items": ["in"]}, THREAD)
    graph.invoke({"items": ["in"]}, THREAD)

    assert graph.get_state(THREAD).values == {"items": ["in", "a", "b", "in", "a", "b"]}
    assert [*graph.get_state_history(THREAD)]
    assert warnings() == []
