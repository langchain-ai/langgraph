"""Regression test for #9089: a deferred node listed in `next` must run on resume.

When `update_state` snapshots a DeltaChannel-backed key, the checkpoint used to
record only the channels the update tasks wrote. `apply_writes` additionally
finishes every channel when the writes trigger no node — that finish is what
makes a deferred node's barrier channel available — so the recorded
`updated_channels` was too small and the deferred node never ran on resume.
"""

from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import _messages_delta_reducer


class _State(TypedDict):
    messages: Annotated[
        list, DeltaChannel(_messages_delta_reducer, snapshot_frequency=1)
    ]


def test_deferred_node_runs_on_resume_after_update_state_snapshots() -> None:
    def a(state):
        return {"messages": [HumanMessage("a1", id="a1")]}

    def b(state):
        return {"messages": [HumanMessage("b", id="b")]}

    def c(state):
        return {}

    builder = StateGraph(_State)
    builder.add_node("a", a)
    builder.add_node("b", b, defer=True)
    builder.add_node("c", c)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", END)
    builder.add_edge("c", END)
    graph = builder.compile(checkpointer=InMemorySaver(), interrupt_after=["a"])

    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"messages": [HumanMessage("s0", id="s0")]}, config)
    graph.update_state(config, {"messages": [HumanMessage("u1", id="u1")]}, as_node="c")

    assert graph.get_state(config).next == ("b",)

    final = graph.invoke(None, config)
    contents = [m.content for m in final["messages"]]
    assert "b" in contents, f"deferred node b never ran on resume: {contents}"
    assert graph.get_state(config).next == ()
