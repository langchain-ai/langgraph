from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import TypedDict


# Subgraph
class MessagesState(TypedDict):
    messages: list[dict[str, str]]


def call_model(state: MessagesState):
    response = {"role": "assistant", "content": "Hello, how can I help you today?"}
    # return {"messages": [state['messages'][-1], response]}
    return Command(goto=END, update={"messages": [state["messages"][-1], response]})


class State(TypedDict):
    messages: list[dict[str, str]]
    user_name: str


subgraph_builder = StateGraph(MessagesState)
subgraph_builder.add_node("call_model", call_model)
subgraph_builder.add_edge(START, "call_model")
subgraph_builder.add_edge("call_model", END)
subgraph = subgraph_builder.compile()

# Parent graph

builder = StateGraph(State)
builder.add_node("subgraph_node", subgraph)
builder.add_edge(START, "subgraph_node")
builder.add_edge("subgraph_node", END)
graph = builder.compile()

response = graph.invoke({"messages": [{"role": "user", "content": "hi!"}], "user_name": "John"})
#> Task call_model with path ('__pregel_pull', 'call_model') wrote to unknown channel branch:to:__end__, ignoring it.
print(response)