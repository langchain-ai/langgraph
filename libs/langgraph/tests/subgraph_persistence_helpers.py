"""Shared helpers for subgraph persistence tests."""

from langchain_core.messages import AIMessage
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import interrupt


class ParentState(TypedDict):
    result: str


def _contents(messages: list) -> list[str]:
    """Extract message content strings for readable assertions."""
    return [m.content for m in messages]


def _echo_graph(prefix: str = "Echo", *, checkpointer=None):
    """Single-node subgraph: echoes '<prefix>: <last message content>'."""

    def echo(state: MessagesState):
        content = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"{prefix}: {content}")]}

    return (
        StateGraph(MessagesState)
        .add_node("echo", echo)
        .add_edge(START, "echo")
        .compile(checkpointer=checkpointer)
    )


def _interrupt_echo_graph(prefix: str = "Processing"):
    """Two-node subgraph with interrupt: echoes '<prefix>: <input>' then 'Done'."""

    def process(state: MessagesState):
        interrupt("continue?")
        content = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"{prefix}: {content}")]}

    def respond(state: MessagesState):
        return {"messages": [AIMessage(content="Done")]}

    return (
        StateGraph(MessagesState)
        .add_node("process", process)
        .add_node("respond", respond)
        .add_edge(START, "process")
        .add_edge("process", "respond")
        .compile()
    )


def _wrap_session_scope(inner, name: str):
    """Wrap a compiled subgraph for session scope (checkpointer=True)."""
    return (
        StateGraph(MessagesState)
        .add_node(name, inner)
        .add_edge(START, name)
        .compile(checkpointer=True)
    )


def _wrap_session_scope_interrupt(name: str, prefix: str = "Processing"):
    """Session-scope subgraph with interrupt."""
    return _wrap_session_scope(_interrupt_echo_graph(prefix), name)
