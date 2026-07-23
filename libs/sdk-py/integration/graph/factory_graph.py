"""Integration fixture: a graph *factory* (not a pre-compiled graph).

Registered as `factory_agent`. The other integration graphs are all
pre-compiled objects, so this is the only fixture that drives the server's
graph-factory code path on a run. That path is where langgraph-api seeds a
`ServerRuntime` into the run config; executing a run against this graph is the
end-to-end regression guard for the langgraph 1.2.3 `ensure_config`
configurable-merge surfacing a leaked `__pregel_runtime`
(`AttributeError: '_ExecutionRuntime' object has no attribute 'control'`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph.state import END, CompiledStateGraph, StateGraph
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langgraph.types import RunnableConfig

    from langgraph_sdk.runtime import ServerRuntime


class State(TypedDict, total=False):
    text: str
    access_context: str
    is_for_execution: bool


async def make_graph(
    config: RunnableConfig, runtime: ServerRuntime
) -> CompiledStateGraph:
    """2-arg factory (config + runtime) returning a compiled echo graph.

    Validates that the server injected a real `ServerRuntime`, then builds a
    one-node graph whose output echoes the input and reports the runtime's
    access context (so the test can confirm the factory ran under execution).
    """
    from langgraph_sdk.runtime import (
        _ExecutionRuntime,
        _ReadRuntime,
    )

    # `if/raise` rather than `assert` so the contract holds under `python -O`.
    if not isinstance(runtime, (_ExecutionRuntime, _ReadRuntime)):
        raise TypeError(
            f"factory must receive a ServerRuntime, got {type(runtime).__name__}"
        )
    if not isinstance(runtime.store, BaseStore):
        raise TypeError(
            f"runtime.store must be a BaseStore, got {type(runtime.store).__name__}"
        )

    access_context = runtime.access_context
    is_for_execution = runtime.execution_runtime is not None

    def echo(state: State) -> State:
        return {
            "text": (state.get("text") or "") + " echoed",
            "access_context": access_context,
            "is_for_execution": is_for_execution,
        }

    workflow = StateGraph(State)
    workflow.add_node("echo", echo)
    workflow.set_entry_point("echo")
    workflow.add_edge("echo", END)
    return workflow.compile()
