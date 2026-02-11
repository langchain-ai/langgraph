import operator
from typing import Annotated, TypedDict
import uuid

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.postgres import PostgresSaver


class State(TypedDict):
    parent: str
    subgraph: str
    messages: Annotated[list, add_messages]


def subgraph_node(state: State):
    val = interrupt("Interrupting from subgraph_node")
    return {"subgraph": "jump", "messages": [AIMessage(content="hop")]}


subgraph = StateGraph(State)
subgraph.add_node("subgraph_node", subgraph_node)
subgraph.add_edge(START, "subgraph_node")
subgraph_app = subgraph.compile(name="subgraph", checkpointer=True)


def parent_node(state: State):
    # val = interrupt("Interrupting from parent_node")
    # print(f"KAWHIIII {val}")
    return {"parent": "sprint"}


graph = StateGraph(State)
graph.add_node("parent_node", parent_node)
graph.add_node("subgraph", subgraph_app)
graph.add_edge(START, "parent_node")
graph.add_edge("parent_node", "subgraph")

with PostgresSaver.from_conn_string(
    "postgresql://postgres:postgres@127.0.0.1:5433/postgres"
) as checkpointer:
    checkpointer.setup()

    app = graph.compile(checkpointer=checkpointer)
    # import pdb
    #
    # pdb.set_trace()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": str(thread_id)}}
    for chunk in app.stream(
        input={}, config=config, subgraphs=True, stream_mode=["debug", "values"]
    ):
        print("\n")
        print(chunk)

    # for chunk in app.stream(
    #     Command(resume="yolo"),
    #     config,
    #     subgraphs=True,
    #     stream_mode=["debug", "values"],
    # ):
    #     print("\n")
    #     print(chunk)

    # assert [*app.stream({}, config, subgraphs=True, stream_mode="values")] == [
    #     ((), {"parent": "sprint", "messages": []}),
    #     ((AnyStr("subgraph:"),), {"parent": "sprint", "messages": []}),
    #     (
    #         (AnyStr("subgraph:"),),
    #         {
    #             "parent": "sprint",
    #             "subgraph": "jump",
    #             "messages": [_AnyIdAIMessage(content="hop")],
    #         },
    #     ),
    #     (
    #         (),
    #         {
    #             "parent": "sprint",
    #             "subgraph": "jump",
    #             "messages": [_AnyIdAIMessage(content="hop")],
    #         },
    #     ),
    # ]
