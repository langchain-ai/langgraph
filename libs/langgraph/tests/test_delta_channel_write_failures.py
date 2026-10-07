"""A checkpoint must never be saved without the `DeltaChannel` writes it reads."""

import operator
import threading
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import START, StateGraph
from langgraph.types import Durability

pytestmark = pytest.mark.anyio

INPUT = {"log": [], "plain": []}
FINAL = {"log": ["a", "b", "c"], "plain": ["a", "b", "c"]}

# Exit mode saves nothing before the failed write, so its retry starts over.
RETRIES = [
    pytest.param("sync", None, id="sync"),
    pytest.param("async", None, id="async"),
    pytest.param("exit", INPUT, id="exit"),
]


def _append(current: list, writes: list) -> list:
    return [*current, *(item for write in writes for item in write)]


class _State(TypedDict):
    log: Annotated[list, DeltaChannel(_append)]
    plain: Annotated[list, operator.add]


class _FailsTheWriteOfBOnce(InMemorySaver):
    failed = False

    def _fail_once(self, writes: Any) -> None:
        if not self.failed and ("log", ["b"]) in writes:
            self.failed = True
            raise ConnectionError("b's write was not saved")

    def put_writes(
        self, config: Any, writes: Any, task_id: str, task_path: str = ""
    ) -> None:
        self._fail_once(writes)
        super().put_writes(config, writes, task_id, task_path)

    async def aput_writes(
        self, config: Any, writes: Any, task_id: str, task_path: str = ""
    ) -> None:
        self._fail_once(writes)
        await super().aput_writes(config, writes, task_id, task_path)


def _a_then_b_then_c(saver: InMemorySaver) -> Any:
    builder = StateGraph(_State)
    for name in "abc":
        builder.add_node(
            name, lambda state, name=name: {"log": [name], "plain": [name]}
        )
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    return builder.compile(checkpointer=saver)


@pytest.mark.parametrize(("durability", "retry_input"), RETRIES)
def test_a_failed_delta_write_is_rerun_not_lost(
    durability: Durability, retry_input: dict | None
) -> None:
    graph = _a_then_b_then_c(_FailsTheWriteOfBOnce())
    config = {"configurable": {"thread_id": "t"}}

    with pytest.raises(ConnectionError):
        graph.invoke(INPUT, config, durability=durability)
    for state in graph.get_state_history(config):
        assert state.values.get("log", []) == state.values.get("plain", [])

    graph.invoke(retry_input, config, durability=durability)
    assert graph.get_state(config).values == FINAL


@pytest.mark.parametrize(("durability", "retry_input"), RETRIES)
async def test_a_failed_delta_write_is_rerun_not_lost_async(
    durability: Durability, retry_input: dict | None
) -> None:
    graph = _a_then_b_then_c(_FailsTheWriteOfBOnce())
    config = {"configurable": {"thread_id": "t"}}

    with pytest.raises(ConnectionError):
        await graph.ainvoke(INPUT, config, durability=durability)
    async for state in graph.aget_state_history(config):
        assert state.values.get("log", []) == state.values.get("plain", [])

    await graph.ainvoke(retry_input, config, durability=durability)
    assert (await graph.aget_state(config)).values == FINAL


def test_a_delta_graph_finishes_on_a_single_background_thread() -> None:
    graph = _a_then_b_then_c(InMemorySaver())
    config = {"configurable": {"thread_id": "t"}, "max_concurrency": 1}
    result: dict = {}
    run = threading.Thread(
        target=lambda: result.update(graph.invoke(INPUT, config, durability="async")),
        daemon=True,
    )

    run.start()
    run.join(timeout=10)

    assert not run.is_alive(), "invoke hung"
    assert result == FINAL
