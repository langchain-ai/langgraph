"""Tests for Graph API pub-sub (topics, Publish, node modes)."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Annotated

import pytest
from typing_extensions import TypedDict

from langgraph.channels import Topic
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Publish


class BusState(TypedDict):
    n: int
    bus: Annotated[Sequence[str], Topic(str)]
    sink: Annotated[list[str], operator.add]


class DualSinkState(TypedDict):
    n: int
    bus: Annotated[Sequence[str], Topic(str)]
    a: Annotated[list[str], operator.add]
    b: Annotated[list[str], operator.add]


def test_fan_out_one_publisher_two_subscribers() -> None:
    def pub(state: DualSinkState):
        return {"n": state["n"] + 1}, Publish("bus", "hello")

    def sub_a(state: DualSinkState):
        return {"a": [f"A:{list(state['bus'])}"]}

    def sub_b(state: DualSinkState):
        return {"b": [f"B:{list(state['bus'])}"]}

    builder = StateGraph(DualSinkState)
    builder.add_node("pub", pub, publishes=["bus"])
    builder.add_node("a", sub_a, mode="pubsub", subscribes=["bus"])
    builder.add_node("b", sub_b, mode="pubsub", subscribes=["bus"])
    builder.add_edge(START, "pub")
    builder.add_edge("pub", END)
    graph = builder.compile()

    result = graph.invoke({"n": 0, "a": [], "b": []})
    assert result["n"] == 1
    assert result["a"] == ["A:['hello']"]
    assert result["b"] == ["B:['hello']"]


def test_fan_in_two_publishers_one_subscriber() -> None:
    def pub1(state: BusState):
        return Publish("bus", "from-1")

    def pub2(state: BusState):
        return Publish("bus", "from-2")

    def sub(state: BusState):
        return {"sink": [f"got:{sorted(state['bus'])}"]}

    builder = StateGraph(BusState)
    builder.add_node("pub1", pub1, publishes=["bus"])
    builder.add_node("pub2", pub2, publishes=["bus"])
    builder.add_node("sub", sub, mode="pubsub", subscribes=["bus"])
    builder.add_edge(START, "pub1")
    builder.add_edge(START, "pub2")
    builder.add_edge("pub1", END)
    builder.add_edge("pub2", END)
    graph = builder.compile()

    result = graph.invoke({"n": 0, "sink": []})
    assert result["sink"] == ["got:['from-1', 'from-2']"]


def test_workflow_and_pubsub_coexist() -> None:
    """Workflow spine continues while a side subscriber audits."""

    class State(TypedDict):
        steps: Annotated[list[str], operator.add]
        events: Annotated[Sequence[str], Topic(str)]

    def step_a(state: State):
        return {"steps": ["a"]}, Publish("events", "after-a")

    def step_b(state: State):
        return {"steps": ["b"]}

    def audit(state: State):
        return {"steps": [f"audit:{list(state['events'])}"]}

    builder = StateGraph(State)
    builder.add_node("a", step_a, publishes=["events"])
    builder.add_node("b", step_b)
    builder.add_node("audit", audit, mode="pubsub", subscribes=["events"])
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    graph = builder.compile()

    result = graph.invoke({"steps": []})
    assert "a" in result["steps"]
    assert "b" in result["steps"]
    assert any(s.startswith("audit:") for s in result["steps"])


def test_mode_pubsub_not_activated_by_edge() -> None:
    class State(TypedDict):
        t: Annotated[Sequence[int], Topic(int)]
        hits: Annotated[list[str], operator.add]

    def entry(state: State):
        return Publish("t", 1)

    def listener(state: State):
        return {"hits": ["listener"]}

    # Edge to pubsub-only node is illegal at compile time
    builder = StateGraph(State)
    builder.add_node("entry", entry, publishes=["t"])
    builder.add_node("listener", listener, mode="pubsub", subscribes=["t"])
    builder.add_edge(START, "entry")
    builder.add_edge("entry", "listener")
    with pytest.raises(ValueError, match="without 'workflow'"):
        builder.compile()

    # Without an edge, the subscriber is only woken by the topic
    builder2 = StateGraph(State)
    builder2.add_node("entry", entry, publishes=["t"])
    builder2.add_node("listener", listener, mode="pubsub", subscribes=["t"])
    builder2.add_edge(START, "entry")
    builder2.add_edge("entry", END)
    graph = builder2.compile()
    result = graph.invoke({"hits": []})
    assert result["hits"] == ["listener"]


def test_mode_workflow_not_activated_by_topic_alone() -> None:
    class State(TypedDict):
        t: Annotated[Sequence[int], Topic(int)]
        hits: Annotated[list[str], operator.add]

    calls: list[str] = []

    def pub(state: State):
        return Publish("t", 7)

    def workflow_only(state: State):
        calls.append("wf")
        return {"hits": ["wf"]}

    builder = StateGraph(State)
    builder.add_node("pub", pub, publishes=["t"])
    builder.add_node("wf", workflow_only)  # no subscription
    builder.add_edge(START, "pub")
    builder.add_edge("pub", END)
    graph = builder.compile()
    result = graph.invoke({"hits": []})
    assert calls == []
    assert result.get("hits", []) == []


def test_mode_both_accepts_edge_and_topic() -> None:
    class State(TypedDict):
        t: Annotated[Sequence[str], Topic(str)]
        hits: Annotated[list[str], operator.add]

    def pub(state: State):
        return Publish("t", "ping")

    def hybrid(state: State):
        note = f"t={list(state.get('t') or [])}"
        return {"hits": [note]}

    builder = StateGraph(State)
    builder.add_node("pub", pub, publishes=["t"])
    builder.add_node("hybrid", hybrid, mode="both", subscribes=["t"])
    builder.add_edge(START, "pub")
    builder.add_edge("pub", "hybrid")
    builder.add_edge("hybrid", END)
    graph = builder.compile()
    result = graph.invoke({"hits": []})
    assert result["hits"]
    # Topic payload visible when woken (edge and/or topic in same step)
    assert any("ping" in h for h in result["hits"])


def test_publish_return_shapes() -> None:
    class State(TypedDict):
        n: int
        bus: Annotated[Sequence[str], Topic(str)]
        sink: Annotated[list[str], operator.add]

    def alone(state: State):
        return Publish("bus", "alone")

    def with_dict(state: State):
        return {"n": 1}, Publish("bus", "dict")

    def with_list(state: State):
        return [Command(update={"n": 2}), Publish("bus", "list")]

    def sub(state: State):
        return {"sink": list(state["bus"])}

    for producer, expected_n, expected_msg in [
        (alone, 0, "alone"),
        (with_dict, 1, "dict"),
        (with_list, 2, "list"),
    ]:
        builder = StateGraph(State)
        builder.add_node("p", producer, publishes=["bus"])
        builder.add_node("s", sub, mode="pubsub", subscribes=["bus"])
        builder.add_edge(START, "p")
        builder.add_edge("p", END)
        graph = builder.compile()
        result = graph.invoke({"n": 0, "sink": []})
        assert result["n"] == expected_n
        assert expected_msg in result["sink"]


def test_state_key_topic_dict_write_triggers_subscriber() -> None:
    """Writing the Topic state key (without Publish) still wakes subscribers."""

    class State(TypedDict):
        bus: Annotated[Sequence[str], Topic(str)]
        sink: Annotated[list[str], operator.add]

    def pub(state: State):
        return {"bus": "via-dict"}

    def sub(state: State):
        return {"sink": list(state["bus"])}

    builder = StateGraph(State)
    builder.add_node("pub", pub, publishes=["bus"])
    builder.add_node("sub", sub, mode="pubsub", subscribes=["bus"])
    builder.add_edge(START, "pub")
    builder.add_edge("pub", END)
    graph = builder.compile()
    result = graph.invoke({"sink": []})
    assert "via-dict" in result["sink"]


def test_add_topic_side_channel() -> None:
    """Topics need not be on the state schema; subscribers still read payloads."""

    class State(TypedDict):
        n: int
        sink: Annotated[list[str], operator.add]

    def pub(state: State):
        return {"n": state["n"] + 1}, Publish("side", "ping")

    def sub(state: State):
        # side topic is injected into the input dict for subscribers
        return {"sink": [str(state.get("side"))]}  # type: ignore[typeddict-item]

    builder = StateGraph(State)
    builder.add_topic("side", typ=str)
    builder.add_node("pub", pub, publishes=["side"])
    builder.add_node("sub", sub, mode="pubsub", subscribes=["side"])
    builder.add_edge(START, "pub")
    builder.add_edge("pub", END)
    graph = builder.compile()
    result = graph.invoke({"n": 0, "sink": []})
    assert result["n"] == 1
    assert result["sink"] == ["['ping']"]


def test_validation_errors() -> None:
    class State(TypedDict):
        x: int

    def noop(state: State):
        return {}

    # mode workflow + subscribes
    b = StateGraph(State)
    with pytest.raises(ValueError, match="requires mode to include 'pubsub'"):
        b.add_node("n", noop, mode="workflow", subscribes=["missing"])

    # pubsub without subscriptions
    b2 = StateGraph(State)
    b2.add_node("n", noop, mode="pubsub")
    b2.add_edge(START, "n")
    with pytest.raises(ValueError, match="no subscriptions"):
        b2.compile()

    # edge to pubsub-only
    class S2(TypedDict):
        t: Annotated[Sequence[int], Topic(int)]

    def pub(state: S2):
        return Publish("t", 1)

    def sub(state: S2):
        return {}

    b3 = StateGraph(S2)
    b3.add_node("pub", pub, publishes=["t"])
    b3.add_node("sub", sub, mode="pubsub", subscribes=["t"])
    b3.add_edge(START, "pub")
    b3.add_edge("pub", "sub")
    with pytest.raises(ValueError, match="without 'workflow'"):
        b3.compile()


def test_publish_and_subscribe_fluent_api() -> None:
    class State(TypedDict):
        bus: Annotated[Sequence[str], Topic(str)]
        sink: Annotated[list[str], operator.add]

    def pub(state: State):
        return Publish("bus", "fluent")

    def sub(state: State):
        return {"sink": list(state["bus"])}

    builder = StateGraph(State)
    builder.add_node("pub", pub)
    builder.add_node("sub", sub)
    builder.publish("pub", "bus")
    builder.subscribe("sub", "bus")
    builder.add_edge(START, "pub")
    builder.add_edge("pub", END)
    graph = builder.compile()
    result = graph.invoke({"sink": []})
    assert result["sink"] == ["fluent"]
    assert "pubsub" in builder.nodes["sub"].modes


def test_accumulate_topic() -> None:
    class AccState(TypedDict):
        bus: Annotated[Sequence[str], Topic(str, accumulate=True)]
        rounds: int
        sink: Annotated[list[str], operator.add]

    def tick(state: AccState):
        return (
            {"rounds": state["rounds"] + 1},
            Publish("bus", f"r{state['rounds']}"),
        )

    def collect(state: AccState):
        return {"sink": [",".join(state["bus"])]}

    builder = StateGraph(AccState)
    builder.add_topic("bus", typ=str, accumulate=True)
    builder.add_node("tick1", tick, publishes=["bus"])
    builder.add_node("tick2", tick, publishes=["bus"])
    builder.add_node("collect", collect, mode="both", subscribes=["bus"])
    builder.add_edge(START, "tick1")
    builder.add_edge("tick1", "tick2")
    builder.add_edge("tick2", "collect")
    builder.add_edge("collect", END)
    graph = builder.compile()
    result = graph.invoke({"rounds": 0, "sink": []})
    # accumulate=True keeps messages across supersteps
    assert "r0" in result["bus"] and "r1" in result["bus"]
    assert result["rounds"] == 2


def test_diagram_shows_mode_and_topic_edges() -> None:
    class State(TypedDict):
        bus: Annotated[Sequence[str], Topic(str)]

    def pub(state: State):
        return Publish("bus", "x")

    def sub(state: State):
        return {}

    builder = StateGraph(State)
    builder.add_node("pub", pub, publishes=["bus"])
    builder.add_node("sub", sub, mode="pubsub", subscribes=["bus"])
    builder.add_edge(START, "pub")
    builder.add_edge("pub", END)
    graph = builder.compile()

    drawable = graph.get_graph()
    assert drawable.nodes["pub"].metadata is not None
    assert drawable.nodes["pub"].metadata["mode"] == "workflow"
    assert drawable.nodes["pub"].metadata["publishes"] == "bus"
    assert drawable.nodes["sub"].metadata is not None
    assert drawable.nodes["sub"].metadata["mode"] == "pubsub"
    assert drawable.nodes["sub"].metadata["subscribes"] == "bus"

    mermaid = drawable.draw_mermaid()
    assert "mode = workflow" in mermaid
    assert "mode = pubsub" in mermaid
    assert "topic:bus" in mermaid


@pytest.mark.anyio
async def test_pubsub_async() -> None:
    class State(TypedDict):
        bus: Annotated[Sequence[str], Topic(str)]
        sink: Annotated[list[str], operator.add]

    async def pub(state: State):
        return Publish("bus", "async")

    async def sub(state: State):
        return {"sink": list(state["bus"])}

    builder = StateGraph(State)
    builder.add_node("pub", pub, publishes=["bus"])
    builder.add_node("sub", sub, mode="pubsub", subscribes=["bus"])
    builder.add_edge(START, "pub")
    builder.add_edge("pub", END)
    graph = builder.compile()
    result = await graph.ainvoke({"sink": []})
    assert result["sink"] == ["async"]
