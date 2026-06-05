"""Reproduction for deepagents#3774.

`update_state` / `aupdate_state` on a *fresh* thread drops the first write to a
`DeltaChannel`-backed channel (e.g. `DeepAgentState.messages`).

Root cause lives in `Pregel.{bulk_update_state,abulk_update_state}`: channel
writes are only persisted when a previous checkpoint exists. On a brand-new thread
there is no previous checkpoint.

Writes
"""

from typing import Annotated, Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import START, StateGraph
from langgraph.graph.message import _messages_delta_reducer

pytestmark = pytest.mark.anyio


def _build_graph(checkpointer: InMemorySaver) -> Any:
    channel = DeltaChannel(_messages_delta_reducer)
    State = TypedDict("State", {"messages": Annotated[list, channel]})  # type: ignore[call-overload]  # noqa: UP013

    def model(state: dict) -> dict:
        return {}

    builder = StateGraph(State)
    builder.add_node("model", model)
    builder.add_edge(START, "model")
    builder.set_finish_point("model")
    return builder.compile(checkpointer=checkpointer)


def test_update_state_fresh_thread_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "fresh-sync"}}
    message = HumanMessage(content="hello", id="m1")

    graph.update_state(config, {"messages": [message]}, as_node="model")

    state = graph.get_state(config)
    assert [m.content for m in state.values["messages"]] == ["hello"]


async def test_aupdate_state_fresh_thread_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "fresh-async"}}
    message = HumanMessage(content="hello", id="m1")

    await graph.aupdate_state(config, {"messages": [message]}, as_node="model")

    state = await graph.aget_state(config)
    assert [m.content for m in state.values["messages"]] == ["hello"]
