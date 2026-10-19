import operator
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels import EphemeralValue, LastValue
from langgraph.func import entrypoint
from langgraph.graph import START, StateGraph
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.types import Command, Durability, Send, interrupt

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.parametrize("asynchronous", [False, True]),
]


@pytest.fixture(params=[None, "sync", "async", "exit"])
def command_durability(request: pytest.FixtureRequest) -> Durability | None:
    return request.param


async def _invoke(
    graph: Pregel,
    value: Any,
    asynchronous: bool,
    durability: Durability | None,
) -> Any:
    config = {"configurable": {"thread_id": "command"}}
    if asynchronous:
        return await graph.ainvoke(value, config, durability=durability)
    return graph.invoke(value, config, durability=durability)


class _State(TypedDict):
    items: Annotated[list[str], operator.add]


@pytest.mark.parametrize("use_command", [False, True])
async def test_fresh_command_update_applied_once(
    asynchronous: bool, command_durability: Durability | None, use_command: bool
) -> None:
    builder = StateGraph(_State)
    builder.add_node("node", lambda state: Command(update={"items": ["node"]}))
    builder.add_edge(START, "node")
    graph = builder.compile(
        checkpointer=InMemorySaver() if command_durability else None
    )
    value = {"items": ["input"]}

    result = await _invoke(
        graph,
        Command(update=value) if use_command else value,
        asynchronous,
        command_durability,
    )

    assert result == {"items": ["input", "node"]}


@pytest.mark.parametrize("use_command", [False, True])
async def test_fresh_command_updates_fields_outside_input_schema(
    asynchronous: bool, command_durability: Durability | None, use_command: bool
) -> None:
    class Input(TypedDict):
        question: str

    class State(Input):
        private: Annotated[list[str], operator.add]
        observed: list[str]

    builder = StateGraph(State, input_schema=Input)
    builder.add_node("node", lambda state: {"observed": state["private"]})
    builder.add_edge(START, "node")
    graph = builder.compile(
        checkpointer=InMemorySaver() if command_durability else None
    )
    value = {"question": "q", "private": ["keep-me"]}

    result = await _invoke(
        graph,
        Command(update=value) if use_command else value,
        asynchronous,
        command_durability,
    )

    expected = ["keep-me"] if use_command else []
    assert result == {"question": "q", "private": expected, "observed": expected}


@pytest.mark.parametrize("input_channel", ["input", START])
async def test_fresh_command_updates_direct_pregel(
    asynchronous: bool, command_durability: Durability | None, input_channel: str
) -> None:
    node = (
        NodeBuilder()
        .subscribe_to(input_channel, read=False)
        .read_from("x")
        .do(lambda state: state.get("x", -1))
        .write_to("output")
    )
    graph = Pregel(
        nodes={input_channel: node},
        channels={
            input_channel: EphemeralValue(object),
            "x": LastValue(int),
            "output": LastValue(int),
        },
        input_channels=input_channel,
        output_channels="output",
        checkpointer=InMemorySaver() if command_durability else None,
    )

    assert (
        await _invoke(
            graph, Command(update={"x": 42}), asynchronous, command_durability
        )
        == 42
    )


async def test_fresh_command_root_update_applied_once(
    asynchronous: bool, command_durability: Durability | None
) -> None:
    builder = StateGraph(Annotated[list[str], operator.add])
    builder.add_node("node", lambda state: Command(update=["node"]))
    builder.add_edge(START, "node")
    graph = builder.compile(
        checkpointer=InMemorySaver() if command_durability else None
    )

    assert await _invoke(
        graph, Command(update=["input"]), asynchronous, command_durability
    ) == ["input", "node"]


@pytest.mark.parametrize("value", [0, False, ""])
async def test_fresh_command_falsy_root_update(
    asynchronous: bool, command_durability: Durability | None, value: Any
) -> None:
    def node(state: Any) -> Any:
        assert state == value
        return state

    builder = StateGraph(type(value))
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    graph = builder.compile(
        checkpointer=InMemorySaver() if command_durability else None
    )

    assert (
        await _invoke(
            graph, Command(update=value, goto="node"), asynchronous, command_durability
        )
        == value
    )


@pytest.mark.parametrize("use_send", [False, True])
async def test_fresh_command_routing_runs_once(
    asynchronous: bool, command_durability: Durability | None, use_send: bool
) -> None:
    seen = []

    def node(state: _State) -> dict:
        seen.append(state["items"])
        return {"items": ["node"]}

    builder = StateGraph(_State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    graph = builder.compile(
        checkpointer=InMemorySaver() if command_durability else None
    )
    # A Send is an additional task, independent of the static START edge.
    goto = Send("node", {"items": ["send"]}) if use_send else "node"

    result = await _invoke(
        graph,
        Command(update={"items": ["input"]}, goto=goto),
        asynchronous,
        command_durability,
    )

    assert result == {
        "items": ["input", "node", "node"] if use_send else ["input", "node"]
    }
    assert sorted(seen) == ([["input"], ["send"]] if use_send else [["input"]])


async def test_fresh_command_functional_entrypoint_receives_input(
    asynchronous: bool, command_durability: Durability | None
) -> None:
    @entrypoint(checkpointer=InMemorySaver() if command_durability else None)
    def __start__(value: Command) -> Any:
        return value.update

    assert await _invoke(
        __start__,
        Command(update={"items": ["input"]}),
        asynchronous,
        command_durability,
    ) == {"items": ["input"]}


@pytest.mark.parametrize("as_node", [START, "__input__"])
async def test_command_staged_by_update_state(
    asynchronous: bool, durability: Durability, as_node: str
) -> None:
    builder = StateGraph(_State)
    builder.add_node("node", lambda state: {"items": ["node"]})
    builder.add_edge(START, "node")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "command"}}
    value = Command(update={"items": ["input"]})
    if asynchronous:
        await graph.aupdate_state(config, value, as_node=as_node)
    else:
        graph.update_state(config, value, as_node=as_node)

    assert await _invoke(graph, None, asynchronous, durability) == {
        "items": ["input", "node"]
    }


@pytest.mark.parametrize("fresh", [False, True])
async def test_command_resume_and_update_applied_once(
    asynchronous: bool, durability: Durability, fresh: bool
) -> None:
    def node(state: _State) -> dict:
        answer = interrupt("Continue?")
        return {"items": [answer]}

    builder = StateGraph(_State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    graph = builder.compile(checkpointer=InMemorySaver())
    if not fresh:
        await _invoke(graph, {"items": ["initial"]}, asynchronous, durability)

    result = await _invoke(
        graph,
        Command(update={"items": ["command"]}, resume="answer"),
        asynchronous,
        durability,
    )

    if fresh:
        # A resume supplied before an interrupt exists does not answer it.
        assert result["items"] == ["command"]
        assert len(result["__interrupt__"]) == 1
        result = await _invoke(
            graph, Command(resume="answer"), asynchronous, durability
        )

    assert result == {
        "items": ["command", "answer"] if fresh else ["initial", "command", "answer"]
    }
