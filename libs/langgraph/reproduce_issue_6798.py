"""Repro for #6798: message streaming key filtering."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    agent_messages: Annotated[list[BaseMessage], add_messages]


def public_node(state: State) -> State:
    return {"messages": [AIMessage(content="Public response")]}


def internal_node(state: State) -> State:
    return {"agent_messages": [AIMessage(content="SECRET internal reasoning")]}


def extract_contents(events: list[tuple[BaseMessage, dict]]) -> list[str]:
    return [message.content for message, _ in events]


def main() -> None:
    workflow = StateGraph(State)
    workflow.add_node("public", public_node)
    workflow.add_node("internal", internal_node)
    workflow.add_edge(START, "public")
    workflow.add_edge("public", "internal")
    workflow.add_edge("internal", END)

    app = workflow.compile()
    app_with_filter = workflow.compile(messages_key="messages")
    input_state: State = {
        "messages": [HumanMessage(content="hi")],
        "agent_messages": [],
    }

    all_contents = extract_contents(
        list(app.stream(input_state, stream_mode="messages"))
    )
    runtime_filtered = extract_contents(
        list(app.stream(input_state, stream_mode="messages", messages_key="messages"))
    )
    compile_filtered = extract_contents(
        list(app_with_filter.stream(input_state, stream_mode="messages"))
    )

    assert all_contents == ["Public response", "SECRET internal reasoning"], (
        all_contents
    )
    assert runtime_filtered == ["Public response"], runtime_filtered
    assert compile_filtered == ["Public response"], compile_filtered
    print("Repro passed:", runtime_filtered)


if __name__ == "__main__":
    main()
