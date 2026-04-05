from dataclasses import dataclass
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import ExecutionInfo, Runtime, ServerInfo, get_runtime


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


# --- ExecutionInfo unit tests ---


def test_execution_info_defaults_and_patch() -> None:
    info = ExecutionInfo(checkpoint_id="c1", checkpoint_ns="ns1", task_id="t1")
    assert info.checkpoint_id == "c1"
    assert info.checkpoint_ns == "ns1"
    assert info.task_id == "t1"
    assert info.thread_id is None
    assert info.run_id is None
    assert info.node_attempt == 1
    assert info.node_first_attempt_time is None

    # patch returns new instance, original unchanged
    patched = info.patch(thread_id="th1", node_attempt=3, task_id="tk1")
    assert patched.thread_id == "th1"
    assert patched.node_attempt == 3
    assert patched.task_id == "tk1"
    assert info.node_attempt == 1
    assert info.task_id == "t1"

    # frozen
    with pytest.raises(AttributeError):
        info.thread_id = "t2"  # type: ignore[misc]


# --- ServerInfo / Runtime unit tests ---


def test_server_info_and_runtime_merge() -> None:
    si = ServerInfo(assistant_id="asst-1", graph_id="graph-1")
    assert si.assistant_id == "asst-1"
    assert si.user is None

    # frozen
    with pytest.raises(AttributeError):
        si.assistant_id = "asst-2"  # type: ignore[misc]

    # runtime default is None
    assert Runtime().server_info is None

    # merge preserves server_info from self when other has None
    r1 = Runtime(server_info=si)
    merged = r1.merge(Runtime())
    assert merged.server_info is si

    # merge takes server_info from other when present
    si2 = ServerInfo(assistant_id="asst-2", graph_id="graph-2")
    merged2 = r1.merge(Runtime(server_info=si2))
    assert merged2.server_info is si2


# --- Integration tests ---


def _make_capture_graph(
    capture: dict[str, Any],
    *,
    checkpointer: Any = None,
) -> Any:
    """Helper: build a simple graph that captures runtime info."""

    class State(TypedDict):
        message: str

    def capture_node(state: State, runtime: Runtime) -> dict[str, Any]:
        capture["execution_info"] = runtime.execution_info
        capture["server_info"] = runtime.server_info
        return {"message": "done"}

    graph = StateGraph(state_schema=State)
    graph.add_node("capture", capture_node)
    graph.add_edge(START, "capture")
    graph.add_edge("capture", END)
    return graph.compile(checkpointer=checkpointer)


def test_execution_info_populated_in_graph() -> None:
    """execution_info fields are populated when running with a checkpointer."""
    captured: dict[str, Any] = {}
    compiled = _make_capture_graph(captured, checkpointer=MemorySaver())
    compiled.invoke(
        {"message": "hi"},
        config={"configurable": {"thread_id": "t-123"}},
    )
    info = captured["execution_info"]
    assert info.thread_id == "t-123"
    assert info.task_id is not None
    assert info.checkpoint_id is not None
    assert info.checkpoint_ns is not None
    assert info.node_attempt == 1
    assert isinstance(info.node_first_attempt_time, float)


@pytest.mark.anyio
async def test_execution_info_populated_in_graph_async() -> None:
    """execution_info fields are populated in async execution."""
    captured: dict[str, Any] = {}
    compiled = _make_capture_graph(captured, checkpointer=MemorySaver())
    await compiled.ainvoke(
        {"message": "hi"},
        config={"configurable": {"thread_id": "t-xyz"}},
    )
    info = captured["execution_info"]
    assert info.thread_id == "t-xyz"
    assert info.node_attempt == 1
    assert isinstance(info.node_first_attempt_time, float)


def test_server_info_from_metadata() -> None:
    """server_info is built from assistant_id/graph_id in config metadata."""
    captured: dict[str, Any] = {}
    compiled = _make_capture_graph(captured)
    compiled.invoke(
        {"message": "hi"},
        config={"metadata": {"assistant_id": "asst-abc", "graph_id": "my-graph"}},
    )
    si = captured["server_info"]
    assert si is not None
    assert si.assistant_id == "asst-abc"
    assert si.graph_id == "my-graph"
    assert si.user is None


def test_server_info_none_without_metadata() -> None:
    """server_info is None when no assistant_id/graph_id in metadata."""
    captured: dict[str, Any] = {}
    compiled = _make_capture_graph(captured)
    compiled.invoke({"message": "hi"})
    assert captured["server_info"] is None


def test_server_info_user_from_auth_user() -> None:
    """server_info.user is populated from configurable['langgraph_auth_user'].

    Tests both a proper BaseUser protocol object and a starlette-style proxy
    that provides `permissions` via __getattr__ (which the Protocol isinstance
    check may not see).
    """

    class _ProxyUser:
        """Mimics langgraph_api's ProxyUser: identity/display_name as properties,
        permissions via __getattr__."""

        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        @property
        def identity(self) -> str:
            return self._data["identity"]

        @property
        def display_name(self) -> str:
            return self._data.get("display_name", self.identity)

        @property
        def is_authenticated(self) -> bool:
            return True

        def __getattr__(self, name: str) -> Any:
            return self._data[name]

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def __contains__(self, key: str) -> bool:
            return key in self._data

        def __iter__(self) -> Any:
            return iter(self._data)

    proxy = _ProxyUser(
        {
            "identity": "proxy-user",
            "display_name": "Proxy User",
            "is_authenticated": True,
            "permissions": ["read"],
        }
    )
    assert not isinstance(proxy, dict)
    assert hasattr(proxy, "identity")

    captured: dict[str, Any] = {}
    compiled = _make_capture_graph(captured)
    compiled.invoke(
        {"message": "hi"},
        config={
            "configurable": {"langgraph_auth_user": proxy},
            "metadata": {"assistant_id": "asst-proxy", "graph_id": "graph-proxy"},
        },
    )
    si = captured["server_info"]
    assert si is not None
    assert si.assistant_id == "asst-proxy"
    assert si.user is not None
    assert si.user.identity == "proxy-user"
    assert si.user["display_name"] == "Proxy User"


def test_execution_info_inherited_by_subgraph() -> None:
    """execution_info is correctly populated for subgraph nodes, including namespace."""
    captured_main: dict[str, Any] = {}
    captured_sub: dict[str, Any] = {}

    class State(TypedDict, total=False):
        message: str

    def subgraph_node(state: State, runtime: Runtime) -> dict[str, str]:
        captured_sub["execution_info"] = runtime.execution_info
        return {"message": "from_sub"}

    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node("sub_node", subgraph_node)
    subgraph_builder.add_edge(START, "sub_node")
    subgraph = subgraph_builder.compile()

    def main_node(state: State, runtime: Runtime) -> dict[str, str]:
        captured_main["execution_info"] = runtime.execution_info
        return {"message": "from_main"}

    builder = StateGraph(State)
    builder.add_node("main_node", main_node)
    builder.add_node("subgraph", subgraph)
    builder.add_edge(START, "main_node")
    builder.add_edge("main_node", "subgraph")
    graph = builder.compile(checkpointer=MemorySaver())

    graph.invoke(
        {"message": "hi"},
        config={"configurable": {"thread_id": "sub-thread"}},
    )

    main_info = captured_main["execution_info"]
    sub_info = captured_sub["execution_info"]

    # Both share the same thread_id
    assert main_info.thread_id == "sub-thread"
    assert sub_info.thread_id == "sub-thread"

    # Both have node_attempt = 1
    assert main_info.node_attempt == 1
    assert sub_info.node_attempt == 1

    # Main namespace is "main_node:<task_id>" (top-level, no separator)
    assert main_info.checkpoint_ns.startswith("main_node:")
    assert "|" not in main_info.checkpoint_ns

    # Subgraph namespace is "subgraph:<task_id>|sub_node:<task_id>" (nested)
    assert sub_info.checkpoint_ns.startswith("subgraph:")
    assert "|sub_node:" in sub_info.checkpoint_ns

    # task_id appears in its own namespace segment
    assert main_info.task_id in main_info.checkpoint_ns
    assert sub_info.task_id in sub_info.checkpoint_ns
