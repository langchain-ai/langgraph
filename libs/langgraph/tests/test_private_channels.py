from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    visible: str
    _secret: bytes


def _build(private):
    def node(state: State) -> State:
        return {"visible": "v", "_secret": b"x" * 8}

    return (
        StateGraph(State)
        .add_node("n", node)
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile(checkpointer=InMemorySaver(), private_channels=private)
    )


def test_private_channel_hidden_from_get_state():
    graph = _build(["_secret"])
    config = {"configurable": {"thread_id": "p1"}}
    graph.invoke({}, config)
    values = graph.get_state(config).values
    assert "visible" in values
    assert "_secret" not in values


def test_without_private_channels_everything_is_visible():
    graph = _build(None)
    config = {"configurable": {"thread_id": "p2"}}
    graph.invoke({}, config)
    assert "_secret" in graph.get_state(config).values


def test_private_channel_still_persisted_for_later_turns():
    graph = _build(["_secret"])
    config = {"configurable": {"thread_id": "p3"}}
    graph.invoke({}, config)
    # hidden in public reads...
    assert "_secret" not in graph.get_state(config).values
    # ...but still present in the persisted checkpoint
    saved = graph.checkpointer.get(config)
    assert saved is not None
    assert "_secret" in saved["channel_values"]
