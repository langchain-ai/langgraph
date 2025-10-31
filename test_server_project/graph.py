"""Minimal graph for server testing."""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    """Graph state with messages."""
    messages: Annotated[list, add_messages]


def chat_node(state: State) -> State:
    """Simple echo node."""
    return {"messages": [{"role": "assistant", "content": "Echo: received your message"}]}


# Build the graph
builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# Compile the graph
graph = builder.compile()

