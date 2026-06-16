from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from shared import get_dummy_message


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_model(state: State) -> dict:
    dummy_message = get_dummy_message()
    message = AIMessage(content=f"Agent says: {dummy_message}")
    return {"messages": [message]}


def should_continue(state: State):
    if len(state["messages"]) > 0:
        return END
    return "call_model"


workflow = StateGraph(State)
workflow.add_node("call_model", call_model)
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", should_continue)

graph = workflow.compile()
