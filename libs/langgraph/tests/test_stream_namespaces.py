import sys
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langgraph.func import entrypoint, task
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import push_message
from langgraph.graph.state import CompiledStateGraph

pytestmark = pytest.mark.anyio

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


@tool
def echo(text: str) -> str:
    """Echo the text."""
    return text


def _one_node_graph(node: Any) -> CompiledStateGraph:
    return (
        StateGraph(MessagesState)
        .add_node(node)
        .add_edge(START, node.__name__)
        .compile()
    )


def _chat_and_echo(state: MessagesState) -> dict:
    echo.invoke({"text": "done"})
    return {"messages": [GenericFakeChatModel(messages=iter(["one two"])).invoke("hi")]}


async def _achat_and_echo(state: MessagesState) -> dict:
    await echo.ainvoke({"text": "done"})
    model = GenericFakeChatModel(messages=iter(["one two"]))
    return {"messages": [await model.ainvoke("hi")]}


def _segment_names(ns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(segment.split(":")[0] for segment in ns)


def _summary(part: dict) -> tuple[tuple[str, ...], str]:
    if part["type"] == "messages":
        return _segment_names(part["ns"]), part["data"][0].content
    return _segment_names(part["ns"]), part["data"]["event"]


_CHAT_AND_ECHO_EVENTS = [
    ((), "tool-started"),
    ((), "tool-finished"),
    ((), "one"),
    ((), " "),
    ((), "two"),
]

_CHILD_CONFIGS = pytest.mark.parametrize(
    "child_config",
    [
        None,
        {"configurable": {"thread_id": "child"}},
        {"configurable": {"thread_id": "parent"}},
    ],
    ids=["no_config", "own_thread", "caller_thread"],
)


_TASK_CALLERS = pytest.mark.parametrize(
    "nested", [False, True], ids=["from_entrypoint", "from_task_in_task"]
)


@_TASK_CALLERS
@pytest.mark.parametrize("subgraphs", [False, True])
def test_task_events_stream_under_the_graph_namespace(
    nested: bool, subgraphs: bool
) -> None:
    @task
    def chat_and_echo() -> None:
        _chat_and_echo({"messages": []})

    @task
    def outer() -> None:
        chat_and_echo().result()

    @entrypoint()
    def workflow(inputs: dict) -> None:
        (outer() if nested else chat_and_echo()).result()

    parts = workflow.stream(
        {}, stream_mode=["messages", "tools"], subgraphs=subgraphs, version="v2"
    )

    assert [_summary(part) for part in parts] == _CHAT_AND_ECHO_EVENTS


@NEEDS_CONTEXTVARS
@_TASK_CALLERS
@pytest.mark.parametrize("subgraphs", [False, True])
async def test_task_events_stream_under_the_graph_namespace_async(
    nested: bool, subgraphs: bool
) -> None:
    @task
    async def chat_and_echo() -> None:
        await _achat_and_echo({"messages": []})

    @task
    async def outer() -> None:
        await chat_and_echo()

    @entrypoint()
    async def workflow(inputs: dict) -> None:
        await (outer() if nested else chat_and_echo())

    parts = workflow.astream(
        {}, stream_mode=["messages", "tools"], subgraphs=subgraphs, version="v2"
    )

    assert [_summary(part) async for part in parts] == _CHAT_AND_ECHO_EVENTS


def test_task_events_stream_under_the_namespace_of_their_subgraph() -> None:
    @task
    def chat_and_echo() -> None:
        _chat_and_echo({"messages": []})

    def node(state: MessagesState) -> dict:
        chat_and_echo().result()
        return {}

    sub = _one_node_graph(node)
    parent = StateGraph(MessagesState).add_node("sub", sub).add_edge(START, "sub")

    parts = parent.compile().stream(
        {"messages": []},
        stream_mode=["messages", "tools"],
        subgraphs=True,
        version="v2",
    )

    assert [_summary(part) for part in parts] == [
        (("sub",), event) for _, event in _CHAT_AND_ECHO_EVENTS
    ]


@_CHILD_CONFIGS
def test_child_graph_streams_only_to_the_node_streaming_it(
    child_config: dict | None,
) -> None:
    child = _one_node_graph(_chat_and_echo)
    received = []

    def node(state: MessagesState) -> dict:
        received.extend(
            child.stream(
                {"messages": []}, child_config, stream_mode=["messages", "tools"]
            )
        )
        return {}

    parts = _one_node_graph(node).stream(
        {"messages": []},
        {"configurable": {"thread_id": "parent"}},
        stream_mode=["messages", "tools"],
    )

    assert list(parts) == []
    assert len(received) == len(_CHAT_AND_ECHO_EVENTS)


@NEEDS_CONTEXTVARS
@_CHILD_CONFIGS
async def test_child_graph_streams_only_to_the_node_streaming_it_async(
    child_config: dict | None,
) -> None:
    child = _one_node_graph(_achat_and_echo)
    received = []

    async def node(state: MessagesState) -> dict:
        stream = child.astream(
            {"messages": []}, child_config, stream_mode=["messages", "tools"]
        )
        received.extend([part async for part in stream])
        return {}

    parts = _one_node_graph(node).astream(
        {"messages": []},
        {"configurable": {"thread_id": "parent"}},
        stream_mode=["messages", "tools"],
    )

    assert [part async for part in parts] == []
    assert len(received) == len(_CHAT_AND_ECHO_EVENTS)


@pytest.mark.parametrize(
    "child_config",
    [
        {"configurable": {"thread_id": "child"}},
        {"configurable": {"thread_id": "parent"}},
    ],
    ids=["own_thread", "caller_thread"],
)
def test_child_graph_on_its_own_thread_skips_a_caller_streaming_subgraphs(
    child_config: dict,
) -> None:
    child = _one_node_graph(_chat_and_echo)

    def node(state: MessagesState) -> dict:
        child.invoke({"messages": []}, child_config)
        return {}

    parts = _one_node_graph(node).stream(
        {"messages": []},
        {"configurable": {"thread_id": "parent"}},
        stream_mode=["messages", "tools"],
        subgraphs=True,
    )

    assert list(parts) == []


def test_node_gets_the_events_of_every_subgraph_it_streams() -> None:
    sub = _one_node_graph(_chat_and_echo)
    counts = []

    def node(state: MessagesState) -> dict:
        for _ in range(2):
            stream = sub.stream({"messages": []}, stream_mode=["messages", "tools"])
            counts.append(len(list(stream)))
        return {}

    _one_node_graph(node).invoke({"messages": []})

    assert counts == [len(_CHAT_AND_ECHO_EVENTS)] * 2


@NEEDS_CONTEXTVARS
async def test_node_gets_the_events_of_every_subgraph_it_streams_async() -> None:
    sub = _one_node_graph(_achat_and_echo)
    counts = []

    async def node(state: MessagesState) -> dict:
        for _ in range(2):
            stream = sub.astream({"messages": []}, stream_mode=["messages", "tools"])
            counts.append(len([part async for part in stream]))
        return {}

    await _one_node_graph(node).ainvoke({"messages": []})

    assert counts == [len(_CHAT_AND_ECHO_EVENTS)] * 2


def _push(state: MessagesState) -> dict:
    push_message(AIMessage(id="pushed", content="hi"), state_key=None)
    return {}


def _namespaces(graph: CompiledStateGraph, subgraphs: bool) -> list[tuple[str, ...]]:
    parts = graph.stream(
        {"messages": []}, stream_mode="messages", subgraphs=subgraphs, version="v2"
    )
    return [_segment_names(part["ns"]) for part in parts]


def _streams_pushing_graph(received: list) -> Any:
    pushing = _one_node_graph(_push)

    def node(state: MessagesState) -> dict:
        for message, _ in pushing.stream({"messages": []}, stream_mode="messages"):
            received.append(message.id)
        return {}

    return node


def test_pushed_message_streams_under_the_namespace_of_its_graph() -> None:
    pushing = _one_node_graph(_push)
    parent = StateGraph(MessagesState).add_node("sub", pushing).add_edge(START, "sub")

    assert _namespaces(pushing, subgraphs=True) == [()]
    assert _namespaces(parent.compile(), subgraphs=False) == []
    assert _namespaces(parent.compile(), subgraphs=True) == [("sub",)]


def test_pushed_message_streams_to_the_node_that_streams_its_graph() -> None:
    received: list[str] = []
    graph = _one_node_graph(_streams_pushing_graph(received))

    assert _namespaces(graph, subgraphs=False) == []
    assert received == ["pushed"]


def test_pushed_message_streams_to_every_stream_that_takes_it() -> None:
    received: list[str] = []
    graph = _one_node_graph(_streams_pushing_graph(received))

    assert _namespaces(graph, subgraphs=True) == [("node",)]
    assert received == ["pushed"]
