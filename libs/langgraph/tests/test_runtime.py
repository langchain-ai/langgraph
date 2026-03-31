from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import ExecutionInfo, Runtime, ServerInfo, UserInfo, get_runtime


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


def test_context_coercion_dataclass() -> None:
    """Test that dict context is coerced to dataclass."""

    @dataclass
    class Context:
        api_key: str
        timeout: int = 30

    class State(TypedDict):
        message: str

    def node_with_context(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        return {
            "message": f"api_key: {runtime.context.api_key}, timeout: {runtime.context.timeout}"
        }

    graph = StateGraph(state_schema=State, context_schema=Context)
    graph.add_node("node", node_with_context)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    compiled = graph.compile()

    # Test dict coercion with all fields
    result = compiled.invoke(
        {"message": "test"}, context={"api_key": "sk_test", "timeout": 60}
    )
    assert result == {"message": "api_key: sk_test, timeout: 60"}

    # Test dict coercion with default field
    result = compiled.invoke({"message": "test"}, context={"api_key": "sk_test2"})
    assert result == {"message": "api_key: sk_test2, timeout: 30"}

    # Test with actual dataclass instance (should still work)
    result = compiled.invoke(
        {"message": "test"}, context=Context(api_key="sk_test3", timeout=90)
    )
    assert result == {"message": "api_key: sk_test3, timeout: 90"}


def test_context_coercion_pydantic() -> None:
    """Test that dict context is coerced to Pydantic model."""

    class Context(BaseModel):
        api_key: str
        timeout: int = 30
        tags: list[str] = []

    class State(TypedDict):
        message: str

    def node_with_context(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        return {
            "message": f"api_key: {runtime.context.api_key}, timeout: {runtime.context.timeout}, tags: {runtime.context.tags}"
        }

    graph = StateGraph(state_schema=State, context_schema=Context)
    graph.add_node("node", node_with_context)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    compiled = graph.compile()

    # Test dict coercion with all fields
    result = compiled.invoke(
        {"message": "test"},
        context={"api_key": "sk_test", "timeout": 60, "tags": ["prod", "v2"]},
    )
    assert result == {"message": "api_key: sk_test, timeout: 60, tags: ['prod', 'v2']"}

    # Test dict coercion with defaults
    result = compiled.invoke({"message": "test"}, context={"api_key": "sk_test2"})
    assert result == {"message": "api_key: sk_test2, timeout: 30, tags: []"}

    # Test with actual Pydantic instance (should still work)
    result = compiled.invoke(
        {"message": "test"},
        context=Context(api_key="sk_test3", timeout=90, tags=["test"]),
    )
    assert result == {"message": "api_key: sk_test3, timeout: 90, tags: ['test']"}


def test_context_coercion_typeddict() -> None:
    """Test that dict context with TypedDict schema passes through as-is."""

    class Context(TypedDict):
        api_key: str
        timeout: int

    class State(TypedDict):
        message: str

    def node_with_context(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        # TypedDict context is just a dict at runtime
        return {
            "message": f"api_key: {runtime.context['api_key']}, timeout: {runtime.context['timeout']}"
        }

    graph = StateGraph(state_schema=State, context_schema=Context)
    graph.add_node("node", node_with_context)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    compiled = graph.compile()

    # Test dict passes through for TypedDict
    result = compiled.invoke(
        {"message": "test"}, context={"api_key": "sk_test", "timeout": 60}
    )
    assert result == {"message": "api_key: sk_test, timeout: 60"}


def test_context_coercion_none() -> None:
    """Test that None context is handled properly."""

    @dataclass
    class Context:
        api_key: str

    class State(TypedDict):
        message: str

    def node_without_context(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        # Should be None when no context provided
        return {"message": f"context is None: {runtime.context is None}"}

    graph = StateGraph(state_schema=State, context_schema=Context)
    graph.add_node("node", node_without_context)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    compiled = graph.compile()

    # Test with None context
    result = compiled.invoke({"message": "test"}, context=None)
    assert result == {"message": "context is None: True"}

    # Test without context parameter (defaults to None)
    result = compiled.invoke({"message": "test"})
    assert result == {"message": "context is None: True"}


def test_context_coercion_errors() -> None:
    """Test error handling for invalid context."""

    @dataclass
    class Context:
        api_key: str  # Required field

    class State(TypedDict):
        message: str

    def node_with_context(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        return {"message": "should not reach here"}

    graph = StateGraph(state_schema=State, context_schema=Context)
    graph.add_node("node", node_with_context)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    compiled = graph.compile()

    # Test missing required field
    with pytest.raises(TypeError):
        compiled.invoke({"message": "test"}, context={"timeout": 60})

    # Test invalid dict keys
    with pytest.raises(TypeError):
        compiled.invoke(
            {"message": "test"}, context={"api_key": "test", "invalid_field": "value"}
        )


@pytest.mark.anyio
async def test_context_coercion_async() -> None:
    """Test context coercion with async methods."""

    @dataclass
    class Context:
        api_key: str
        async_mode: bool = True

    class State(TypedDict):
        message: str

    async def async_node(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        return {
            "message": f"async api_key: {runtime.context.api_key}, async_mode: {runtime.context.async_mode}"
        }

    graph = StateGraph(state_schema=State, context_schema=Context)
    graph.add_node("node", async_node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    compiled = graph.compile()

    # Test dict coercion with ainvoke
    result = await compiled.ainvoke(
        {"message": "test"}, context={"api_key": "sk_async", "async_mode": False}
    )
    assert result == {"message": "async api_key: sk_async, async_mode: False"}

    # Test dict coercion with astream
    chunks = []
    async for chunk in compiled.astream(
        {"message": "test"}, context={"api_key": "sk_stream"}
    ):
        chunks.append(chunk)

    # Find the chunk with our node output
    node_output = None
    for chunk in chunks:
        if "node" in chunk:
            node_output = chunk["node"]
            break

    assert node_output == {"message": "async api_key: sk_stream, async_mode: True"}


def test_context_coercion_stream() -> None:
    """Test context coercion with sync stream method."""

    @dataclass
    class Context:
        api_key: str
        stream_mode: str = "default"

    class State(TypedDict):
        message: str

    def node_with_context(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        return {
            "message": f"stream api_key: {runtime.context.api_key}, mode: {runtime.context.stream_mode}"
        }

    graph = StateGraph(state_schema=State, context_schema=Context)
    graph.add_node("node", node_with_context)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    compiled = graph.compile()

    # Test dict coercion with stream
    chunks = []
    for chunk in compiled.stream(
        {"message": "test"}, context={"api_key": "sk_stream", "stream_mode": "fast"}
    ):
        chunks.append(chunk)

    # Find the chunk with our node output
    node_output = None
    for chunk in chunks:
        if "node" in chunk:
            node_output = chunk["node"]
            break

    assert node_output == {"message": "stream api_key: sk_stream, mode: fast"}


def test_context_coercion_pydantic_validation_errors() -> None:
    """Test that Pydantic validation errors are raised."""

    class Context(BaseModel):
        api_key: str
        timeout: int

    class State(TypedDict):
        message: str

    def node_with_context(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        return {
            "message": f"api_key: {runtime.context.api_key}, timeout: {runtime.context.timeout}"
        }

    graph = StateGraph(state_schema=State, context_schema=Context)
    graph.add_node("node", node_with_context)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)

    compiled = graph.compile()

    with pytest.raises(ValidationError):
        compiled.invoke(
            {"message": "test"}, context={"api_key": "sk_test", "timeout": "not_an_int"}
        )


# --- ExecutionInfo tests ---


def test_execution_info_defaults() -> None:
    info = ExecutionInfo()
    assert info.thread_id is None
    assert info.checkpoint_id is None
    assert info.checkpoint_ns == ""
    assert info.task_id is None
    assert info.run_id is None
    assert info.node_attempt == 1
    assert info.node_first_attempt_time is None


def test_execution_info_patch() -> None:
    info = ExecutionInfo(thread_id="t1", run_id="r1")
    patched = info.patch(node_attempt=3, task_id="tk1")
    assert patched.thread_id == "t1"
    assert patched.run_id == "r1"
    assert patched.node_attempt == 3
    assert patched.task_id == "tk1"
    # original unchanged
    assert info.node_attempt == 1
    assert info.task_id is None


def test_execution_info_frozen() -> None:
    info = ExecutionInfo(thread_id="t1")
    with pytest.raises(AttributeError):
        info.thread_id = "t2"  # type: ignore[misc]


def test_execution_info_populated_in_node() -> None:
    """ExecutionInfo fields are populated from config in node execution."""

    class State(TypedDict):
        result: str

    captured: dict[str, Any] = {}

    def my_node(state: State, runtime: Runtime) -> dict[str, Any]:
        info = runtime.execution_info
        captured["thread_id"] = info.thread_id
        captured["run_id"] = info.run_id
        captured["node_attempt"] = info.node_attempt
        captured["node_first_attempt_time"] = info.node_first_attempt_time
        return {"result": "ok"}

    from langgraph.checkpoint.memory import MemorySaver

    graph = (
        StateGraph(State)
        .add_node("my_node", my_node)
        .add_edge(START, "my_node")
        .compile(checkpointer=MemorySaver())
    )
    config = {"configurable": {"thread_id": "thread-abc"}}
    result = graph.invoke({"result": ""}, config=config)

    assert result["result"] == "ok"
    assert captured["thread_id"] == "thread-abc"
    assert captured["node_attempt"] == 1
    assert isinstance(captured["node_first_attempt_time"], float)


@pytest.mark.anyio
async def test_execution_info_populated_in_node_async() -> None:
    """ExecutionInfo fields are populated from config in async node execution."""

    class State(TypedDict):
        result: str

    captured: dict[str, Any] = {}

    async def my_node(state: State, runtime: Runtime) -> dict[str, Any]:
        info = runtime.execution_info
        captured["thread_id"] = info.thread_id
        captured["node_attempt"] = info.node_attempt
        captured["node_first_attempt_time"] = info.node_first_attempt_time
        return {"result": "ok"}

    from langgraph.checkpoint.memory import MemorySaver

    graph = (
        StateGraph(State)
        .add_node("my_node", my_node)
        .add_edge(START, "my_node")
        .compile(checkpointer=MemorySaver())
    )
    config = {"configurable": {"thread_id": "thread-xyz"}}
    result = await graph.ainvoke({"result": ""}, config=config)

    assert result["result"] == "ok"
    assert captured["thread_id"] == "thread-xyz"
    assert captured["node_attempt"] == 1
    assert isinstance(captured["node_first_attempt_time"], float)


# --- UserInfo protocol tests ---


def test_user_info_protocol() -> None:
    """BaseUser from SDK satisfies UserInfo protocol structurally."""

    class MyUser:
        @property
        def identity(self) -> str:
            return "user-123"

        @property
        def display_name(self) -> str:
            return "Alice"

        @property
        def is_authenticated(self) -> bool:
            return True

        @property
        def permissions(self) -> Sequence[str]:
            return ["read", "write"]

    user = MyUser()
    assert isinstance(user, UserInfo)
    assert user.identity == "user-123"
    assert user.display_name == "Alice"
    assert user.is_authenticated is True
    assert user.permissions == ["read", "write"]


# --- ServerInfo tests ---


def test_server_info() -> None:
    info = ServerInfo(assistant_id="asst-1", graph_id="graph-1")
    assert info.assistant_id == "asst-1"
    assert info.graph_id == "graph-1"
    assert info.user is None


def test_server_info_with_user() -> None:
    class MyUser:
        @property
        def identity(self) -> str:
            return "user-1"

        @property
        def display_name(self) -> str:
            return "Bob"

        @property
        def is_authenticated(self) -> bool:
            return True

        @property
        def permissions(self) -> Sequence[str]:
            return []

    user = MyUser()
    info = ServerInfo(assistant_id="asst-1", graph_id="graph-1", user=user)
    assert info.user is not None
    assert info.user.identity == "user-1"


def test_server_info_frozen() -> None:
    info = ServerInfo(assistant_id="asst-1", graph_id="graph-1")
    with pytest.raises(AttributeError):
        info.assistant_id = "asst-2"  # type: ignore[misc]


# --- Runtime server_info tests ---


def test_runtime_server_info_default_none() -> None:
    runtime = Runtime()
    assert runtime.server_info is None


def test_runtime_server_info_set() -> None:
    si = ServerInfo(assistant_id="asst-1", graph_id="graph-1")
    runtime = Runtime(server_info=si)
    assert runtime.server_info is si


def test_runtime_merge_preserves_server_info() -> None:
    si = ServerInfo(assistant_id="asst-1", graph_id="graph-1")
    runtime1 = Runtime(server_info=si)
    runtime2 = Runtime()

    # server_info from self is preserved when other has None
    merged = runtime1.merge(runtime2)
    assert merged.server_info is si

    # server_info from other takes precedence
    si2 = ServerInfo(assistant_id="asst-2", graph_id="graph-2")
    runtime3 = Runtime(server_info=si2)
    merged2 = runtime1.merge(runtime3)
    assert merged2.server_info is si2
