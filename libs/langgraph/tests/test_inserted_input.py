import operator
from typing import Annotated

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import RunControl
from langgraph.types import Command, Send, interrupt


class State(TypedDict, total=False):
    log: Annotated[list[str], operator.add]
    seen: list[str]


def _tools_then_model(control: RunControl) -> StateGraph:
    """`tools` queues input while it runs; `model` records what it sees."""

    def tools(state: State) -> State:
        control.submit_input("insert-1", {"log": ["inserted"]})
        return {"log": ["tools"]}

    def model(state: State) -> State:
        return {"seen": list(state["log"])}

    graph = StateGraph(State)
    graph.add_node("tools", tools)
    graph.add_node("model", model)
    graph.add_edge(START, "tools")
    graph.add_edge("tools", "model")
    graph.add_edge("model", END)
    return graph


def test_inserted_input_is_applied_before_an_allowed_node() -> None:
    control = RunControl()
    checkpointer = InMemorySaver()
    graph = _tools_then_model(control).compile(
        checkpointer=checkpointer, insert_before=["model"]
    )
    config = {"configurable": {"thread_id": "1"}}

    result = graph.invoke({"log": ["user"]}, config, control=control)

    assert result["seen"] == ["user", "tools", "inserted"]
    assert not control.has_inputs
    inserted = [
        snapshot.metadata
        for snapshot in graph.get_state_history(config)
        if snapshot.metadata.get("inserted_inputs")
    ]
    assert len(inserted) == 1
    assert inserted[0]["source"] == "input"
    assert inserted[0]["inserted_inputs"] == ["insert-1"]


@pytest.mark.anyio
async def test_inserted_input_is_applied_before_an_allowed_node_async() -> None:
    control = RunControl()
    graph = _tools_then_model(control).compile(
        checkpointer=InMemorySaver(), insert_before=["model"]
    )

    result = await graph.ainvoke(
        {"log": ["user"]}, {"configurable": {"thread_id": "1"}}, control=control
    )

    assert result["seen"] == ["user", "tools", "inserted"]
    assert not control.has_inputs


def test_inserted_input_is_applied_without_a_checkpointer() -> None:
    control = RunControl()
    graph = _tools_then_model(control).compile(insert_before=["model"])

    result = graph.invoke({"log": ["user"]}, control=control)

    assert result["seen"] == ["user", "tools", "inserted"]


@pytest.mark.parametrize("insert_before", [None, ["tools"]])
def test_inserted_input_waits_when_the_next_node_is_not_allowed(
    insert_before: list[str] | None,
) -> None:
    control = RunControl()
    graph = _tools_then_model(control).compile(insert_before=insert_before)

    result = graph.invoke({"log": ["user"]}, control=control)

    assert result["seen"] == ["user", "tools"]
    assert control.take_inputs() == [("insert-1", {"log": ["inserted"]})]


def _outer(node: object) -> StateGraph:
    """Runs `node` (which runs a subgraph), then `finish`."""

    def finish(state: State) -> State:
        return {"log": ["finished"]}

    outer = StateGraph(State)
    outer.add_node("outer", node)
    outer.add_node("finish", finish)
    outer.add_edge(START, "outer")
    outer.add_edge("outer", "finish")
    outer.add_edge("finish", END)
    return outer


def test_inserted_input_is_not_applied_inside_a_subgraph() -> None:
    control = RunControl()
    inner = _tools_then_model(control).compile(insert_before=["model"])

    def outer_node(state: State) -> State:
        return inner.invoke({"log": state["log"]})

    result = (
        _outer(outer_node)
        .compile(insert_before=["finish"])
        .invoke({"log": ["user"]}, control=control)
    )

    # the subgraph's model didn't see it; the parent applied it before `finish`
    assert result["seen"] == ["user", "tools"]
    assert result["log"][-2:] == ["inserted", "finished"]


@pytest.mark.anyio
async def test_inserted_input_is_not_applied_inside_a_subgraph_async() -> None:
    control = RunControl()
    inner = _tools_then_model(control).compile(insert_before=["model"])

    async def outer_node(state: State) -> State:
        return await inner.ainvoke({"log": state["log"]})

    result = await (
        _outer(outer_node)
        .compile(insert_before=["finish"])
        .ainvoke({"log": ["user"]}, control=control)
    )

    assert result["seen"] == ["user", "tools"]
    assert result["log"][-2:] == ["inserted", "finished"]


def test_inserted_input_is_streamed_as_values() -> None:
    control = RunControl()
    graph = _tools_then_model(control).compile(insert_before=["model"])

    chunks = list(
        graph.stream({"log": ["user"]}, control=control, stream_mode="values")
    )

    assert {"log": ["user", "tools", "inserted"]} in chunks


@pytest.mark.parametrize(
    "value", ["not a dict", {"seen": ["forged"]}, {"branch:to:model": None}]
)
def test_invalid_inserted_input_is_skipped(value: object) -> None:
    class InputState(TypedDict):
        log: Annotated[list[str], operator.add]

    control = RunControl()

    def tools(state: State) -> State:
        control.submit_input("bad", value)
        return {"log": ["tools"]}

    def model(state: State) -> State:
        return {"seen": list(state["log"])}

    builder = StateGraph(State, input_schema=InputState)
    builder.add_node("tools", tools)
    builder.add_node("model", model)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", "model")
    builder.add_edge("model", END)
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer, insert_before=["model"])
    config = {"configurable": {"thread_id": "1"}}

    result = graph.invoke({"log": ["user"]}, config, control=control)

    assert result["seen"] == ["user", "tools"]
    assert not any(
        snapshot.metadata.get("inserted_inputs")
        for snapshot in graph.get_state_history(config)
    )


@pytest.mark.parametrize("by_id", [False, True])
def test_inserted_input_waits_while_resuming(by_id: bool) -> None:
    def model(state: State) -> State:
        answer = interrupt("approve?")
        return {"log": [f"answered {answer}"], "seen": list(state["log"])}

    builder = StateGraph(State)
    builder.add_node("model", model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    graph = builder.compile(checkpointer=InMemorySaver(), insert_before=["model"])
    config = {"configurable": {"thread_id": "1"}}
    paused = graph.invoke({"log": ["user"]}, config)
    interrupt_id = paused["__interrupt__"][0].id
    control = RunControl()
    control.submit_input("insert-1", {"log": ["inserted"]})

    result = graph.invoke(
        Command(resume={interrupt_id: "yes"} if by_id else "yes"),
        config,
        control=control,
    )

    # the resume is answered as asked; the input waits for a later boundary
    assert result["seen"] == ["user"]
    assert result["log"] == ["user", "answered yes"]
    assert control.take_inputs() == [("insert-1", {"log": ["inserted"]})]


def test_insert_before_rejects_unknown_nodes() -> None:
    graph = _tools_then_model(RunControl())

    with pytest.raises(ValueError, match="missing"):
        graph.compile(insert_before=["missing"])


def test_inserts_are_applied_one_by_one_and_bad_values_skipped() -> None:
    class InputState(TypedDict, total=False):
        log: Annotated[list[str], operator.add]
        note: str

    class FullState(InputState, total=False):
        seen: list[str]

    control = RunControl()

    def tools(state: FullState) -> FullState:
        control.submit_input("first", {"note": "a", "log": ["one"]})
        control.submit_input("bad", {"log": "not a list"})
        control.submit_input("second", {"note": "b", "log": ["two"]})
        return {"log": ["tools"]}

    def model(state: FullState) -> FullState:
        return {"seen": [*state["log"], state.get("note", "")]}

    builder = StateGraph(FullState, input_schema=InputState)
    builder.add_node("tools", tools)
    builder.add_node("model", model)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", "model")
    builder.add_edge("model", END)
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer, insert_before=["model"])
    config = {"configurable": {"thread_id": "1"}}

    result = graph.invoke({"log": ["user"]}, config, control=control)

    assert result["seen"] == ["user", "tools", "one", "two", "b"]
    inserted = [
        snapshot.metadata["inserted_inputs"]
        for snapshot in graph.get_state_history(config)
        if snapshot.metadata.get("inserted_inputs")
    ]
    assert inserted == [["first", "second"]]


def test_inserted_input_waits_while_resuming_from_a_breakpoint() -> None:
    control = RunControl()
    graph = _tools_then_model(control).compile(
        checkpointer=InMemorySaver(),
        insert_before=["model"],
        interrupt_before=["model"],
    )
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"log": ["user"]}, config)
    assert graph.get_state(config).next == ("model",)

    result = graph.invoke(None, config, control=control)

    # the breakpoint is passed once; the input waits for a later boundary
    assert result["seen"] == ["user", "tools"]
    assert graph.get_state(config).next == ()


def test_inserted_input_is_not_applied_without_steps_left() -> None:
    control = RunControl()
    graph = _tools_then_model(control).compile(insert_before=["model"])

    with pytest.raises(GraphRecursionError):
        graph.invoke({"log": ["user"]}, {"recursion_limit": 1}, control=control)

    assert control.take_inputs() == [("insert-1", {"log": ["inserted"]})]


def test_inserted_input_keeps_the_recursion_budget() -> None:
    control = RunControl()
    graph = _tools_then_model(control).compile(insert_before=["model"])
    # the smallest limit the graph runs within without inserted input
    config = {"recursion_limit": 3}
    _tools_then_model(RunControl()).compile().invoke({"log": ["user"]}, config)

    result = graph.invoke({"log": ["user"]}, config, control=control)

    assert result["seen"] == ["user", "tools", "inserted"]


def test_inserted_input_waits_for_a_node_started_by_send() -> None:
    control = RunControl()

    def tools(state: State) -> State:
        control.submit_input("insert-1", {"log": ["inserted"]})
        return {"log": ["tools"]}

    def model(state: State) -> State:
        return {"seen": list(state["log"])}

    builder = StateGraph(State)
    builder.add_node("tools", tools)
    builder.add_node("model", model)
    builder.add_edge(START, "tools")
    builder.add_conditional_edges("tools", lambda _: [Send("model", {"log": ["sent"]})])
    builder.add_edge("model", END)

    result = builder.compile(insert_before=["model"]).invoke(
        {"log": ["user"]}, control=control
    )

    # `model` reads its `Send` argument, so the input would never reach it
    assert result["seen"] == ["sent"]
    assert control.take_inputs() == [("insert-1", {"log": ["inserted"]})]
