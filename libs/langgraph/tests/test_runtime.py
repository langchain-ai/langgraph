from dataclasses import dataclass
from typing import Any

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime, get_runtime


def test_injected_runtime() -> None:
    @dataclass
    class Context:
        api_key: str

    class State(TypedDict):
        message: str

    def injected_runtime(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        return {"message": f"api key: {runtime.context.api_key}"}

    graph = StateGraph(state_schema=State, context_schema=Context)
    graph.add_node("injected_runtime", injected_runtime)
    graph.add_edge(START, "injected_runtime")
    graph.add_edge("injected_runtime", END)
    compiled = graph.compile()
    result = compiled.invoke(
        {"message": "hello world"}, context=Context(api_key="sk_123456")
    )
    assert result == {"message": "api key: sk_123456"}


def test_context_runtime() -> None:
    @dataclass
    class Context:
        api_key: str

    class State(TypedDict):
        message: str

    def context_runtime(state: State) -> dict[str, Any]:
        runtime = get_runtime(Context)
        return {"message": f"api key: {runtime.context.api_key}"}

    graph = StateGraph(state_schema=State, context_schema=Context)
    graph.add_node("context_runtime", context_runtime)
    graph.add_edge(START, "context_runtime")
    graph.add_edge("context_runtime", END)
    compiled = graph.compile()
    result = compiled.invoke(
        {"message": "hello world"}, context=Context(api_key="sk_123456")
    )
    assert result == {"message": "api key: sk_123456"}


def test_override_runtime() -> None:
    @dataclass
    class Context:
        api_key: str

    prev = Runtime(context=Context(api_key="abc"))
    new = prev.override(context=Context(api_key="def"))
    assert new.override(context=Context(api_key="def")).context.api_key == "def"


def test_merge_runtime() -> None:
    @dataclass
    class Context:
        api_key: str

    runtime1 = Runtime(context=Context(api_key="abc"))
    runtime2 = Runtime(context=Context(api_key="def"))
    runtime3 = Runtime(context=None)

    assert runtime1.merge(runtime2).context.api_key == "def"
    # override only applies to non-falsy values
    assert runtime1.merge(runtime3).context.api_key == "abc"  # type: ignore


def test_runtime_propogated_to_subgraph() -> None:
    @dataclass
    class Context:
        username: str

    class State(TypedDict, total=False):
        subgraph: str
        main: str

    def subgraph_node_1(state: State, runtime: Runtime[Context]):
        return {"subgraph": f"{runtime.context.username}!"}

    subgraph_builder = StateGraph(State, context_schema=Context)
    subgraph_builder.add_node(subgraph_node_1)
    subgraph_builder.set_entry_point("subgraph_node_1")
    subgraph = subgraph_builder.compile()

    def main_node(state: State, runtime: Runtime[Context]):
        return {"main": f"{runtime.context.username}!"}

    builder = StateGraph(State, context_schema=Context)
    builder.add_node(main_node)
    builder.add_node("node_1", subgraph)
    builder.set_entry_point("main_node")
    builder.add_edge("main_node", "node_1")
    graph = builder.compile()

    context = Context(username="Alice")
    result = graph.invoke({}, context=context)
    assert result == {"subgraph": "Alice!", "main": "Alice!"}
