# graphs/agent.py
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class State(TypedDict):
    messages: list


def agent_node(state: State):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_edge("agent", END)
graph = graph.compile()
