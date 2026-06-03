"""Subgraph fixture exercising v3 `thread.subgraphs`.

This graph is intentionally small and first-party: the SDK integration suite
only needs a graph that reliably emits a direct child namespace when streamed.
Using a compiled child `StateGraph` keeps the fixture independent of external
agent-package internals while still covering the server and SDK projection that
turns subgraph lifecycle events into `ScopedStreamHandle`s.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def research(state: AgentState) -> dict[str, list[AIMessage]]:
    """Single child-graph node that emits a deterministic message."""
    return {
        "messages": [
            AIMessage(
                content="v3 streaming is event-typed and thread-centric.",
                id="subgraph-msg-1",
            )
        ]
    }


_child_builder = StateGraph(AgentState)
_child_builder.add_node("research", research)
_child_builder.set_entry_point("research")
_child_builder.set_finish_point("research")
child_graph = _child_builder.compile()


_builder = StateGraph(AgentState)
_builder.add_node("researcher", child_graph)
_builder.set_entry_point("researcher")
_builder.set_finish_point("researcher")
graph = _builder.compile(name="v3_subgraph_agent")
