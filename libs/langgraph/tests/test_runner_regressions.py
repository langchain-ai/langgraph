"""Regression tests for Pregel runner/main engine fixes.

Covers:
- sync `PregelRunner.tick` waiting over a snapshot of the futures dict, so
  worker threads inserting new futures via `call()` cannot trigger
  `RuntimeError: dictionary changed size during iteration` inside
  `concurrent.futures.wait`.
- `durability="sync"` arriving via config on a graph compiled without a
  checkpointer no longer raises `AttributeError` for `_put_checkpoint_fut`.
"""

import pytest
from typing_extensions import TypedDict

from langgraph._internal._constants import CONFIG_KEY_DURABILITY
from langgraph.func import entrypoint, task
from langgraph.graph import START, StateGraph

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_call_fanout_from_worker_threads() -> None:
    """Stress: `call()` inserts futures from worker threads while the driver
    thread is blocked in `concurrent.futures.wait` over the same FuturesDict.

    Pre-fix this was timing-dependent (could raise `RuntimeError: dictionary
    changed size during iteration`); post-fix it must pass deterministically.
    """

    @task
    def leaf(i: int) -> int:
        return i

    @task
    def branch(i: int) -> int:
        # tasks spawning tasks: more insertions from more worker threads
        futs = [leaf(i * 10 + j) for j in range(5)]
        return sum(f.result() for f in futs)

    @entrypoint()
    def graph(n: int) -> int:
        futs = [branch(i) for i in range(n)]
        return sum(f.result() for f in futs)

    # NOTE: keep the nested fan-out narrow: each `branch` (and the
    # entrypoint) blocks a pool thread in `.result()`, so a wide nested
    # fan-out can exhaust the executor pool and deadlock regardless of
    # the wait-snapshot fix
    expected = sum(i * 10 + j for i in range(3) for j in range(5))
    for _ in range(5):
        assert graph.invoke(3) == expected

    @task
    def double(i: int) -> int:
        return i * 2

    @entrypoint()
    def wide(n: int) -> int:
        # 50 children spawned in quick succession from the entrypoint's
        # worker thread while the driver waits
        futs = [double(i) for i in range(n)]
        return sum(f.result() for f in futs)

    for _ in range(5):
        assert wide.invoke(50) == sum(i * 2 for i in range(50))


class _State(TypedDict):
    foo: str


def _node(state: _State) -> _State:
    return {"foo": "bar"}


def _compile_without_checkpointer():
    builder = StateGraph(_State)
    builder.add_node("node", _node)
    builder.add_edge(START, "node")
    return builder.compile()


def test_durability_sync_without_checkpointer() -> None:
    """`durability="sync"` injected via config (as a parent graph does for
    subgraphs) on a checkpointer-less graph used to raise AttributeError on
    `loop._put_checkpoint_fut`."""
    graph = _compile_without_checkpointer()
    result = graph.invoke(
        {"foo": ""}, config={"configurable": {CONFIG_KEY_DURABILITY: "sync"}}
    )
    assert result == {"foo": "bar"}


async def test_durability_sync_without_checkpointer_async() -> None:
    graph = _compile_without_checkpointer()
    result = await graph.ainvoke(
        {"foo": ""}, config={"configurable": {CONFIG_KEY_DURABILITY: "sync"}}
    )
    assert result == {"foo": "bar"}
