from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict
from typing import Any
from langgraph.types import GraphRuntime


@dataclass
class Context:
    api_key: str


class State(TypedDict):
    message: str


def print_context(state: State, runtime: GraphRuntime[Context]) -> dict[str, Any]:
    print(runtime)
    print(f"Using API key: {runtime.context.api_key}")
    return {"message": "done"}


graph = StateGraph(state_schema=State, context_schema=Context)

graph.add_node("print_key", print_context)
graph.add_edge(START, "print_key")
graph.add_edge("print_key", END)

compiled = graph.compile(store=InMemoryStore())
compiled.invoke({"message": "hello world"}, context=Context(api_key="sk_123456"))
# > Using API key: sk_123456
