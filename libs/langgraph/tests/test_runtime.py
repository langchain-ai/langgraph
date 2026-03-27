import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import RunControl, Runtime, get_runtime


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


def test_merge_runtime_preserves_run_control() -> None:
    control = RunControl()
    runtime1 = Runtime(control=control)
    runtime2 = Runtime(context=None)

    assert runtime1.merge(runtime2).control is control


def test_runtime_request_drain_stops_future_steps() -> None:
    class State(TypedDict, total=False):
        first: str
        second: str

    def first_node(state: State, runtime: Runtime) -> dict[str, str]:
        runtime.request_drain()
        return {"first": "done"}

    def second_node(state: State) -> dict[str, str]:
        return {"second": "should-not-run"}

    graph = StateGraph(State)
    graph.add_node("first", first_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    with pytest.raises(GraphDrained, match="shutdown"):
        graph.compile().invoke({})


@pytest.mark.anyio
async def test_runtime_request_drain_stops_future_steps_async() -> None:
    class State(TypedDict, total=False):
        first: str
        second: str

    async def first_node(state: State, runtime: Runtime) -> dict[str, str]:
        runtime.request_drain()
        return {"first": "done"}

    async def second_node(state: State) -> dict[str, str]:
        return {"second": "should-not-run"}

    graph = StateGraph(State)
    graph.add_node("first", first_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    with pytest.raises(GraphDrained, match="shutdown"):
        await graph.compile().ainvoke({})


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


def test_external_drain_concurrent_sync() -> None:
    """External thread calls request_drain() while graph is mid-execution."""

    class State(TypedDict, total=False):
        first: str
        second: str

    started = threading.Event()

    def first_node(state: State) -> dict[str, str]:
        started.set()
        time.sleep(0.5)
        return {"first": "done"}

    def second_node(state: State) -> dict[str, str]:
        return {"second": "should-not-run"}

    graph = StateGraph(State)
    graph.add_node("first", first_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    control = RunControl()
    compiled = graph.compile()

    exc_holder: list[BaseException | None] = [None]

    def run_graph() -> None:
        try:
            compiled.invoke({}, control=control)
        except GraphDrained as e:
            exc_holder[0] = e

    t = threading.Thread(target=run_graph)
    t.start()

    started.wait(timeout=5)
    control.request_drain("sigterm")

    t.join(timeout=10)

    exc = exc_holder[0]
    assert isinstance(exc, GraphDrained)
    assert exc.reason == "sigterm"


@pytest.mark.anyio
async def test_external_drain_concurrent_async() -> None:
    """External task calls request_drain() while graph is mid-execution."""

    class State(TypedDict, total=False):
        first: str
        second: str

    started = asyncio.Event()

    async def first_node(state: State) -> dict[str, str]:
        started.set()
        await asyncio.sleep(0.5)
        return {"first": "done"}

    async def second_node(state: State) -> dict[str, str]:
        return {"second": "should-not-run"}

    graph = StateGraph(State)
    graph.add_node("first", first_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    control = RunControl()
    compiled = graph.compile()

    async def drain_after_start() -> None:
        await started.wait()
        control.request_drain("sigterm")

    drain_task = asyncio.create_task(drain_after_start())

    with pytest.raises(GraphDrained, match="sigterm"):
        await compiled.ainvoke({}, control=control)

    await drain_task


@pytest.mark.anyio
async def test_drain_then_cancel_after_graceful_timeout() -> None:
    """Simulate: drain requested -> node still running -> graceful timeout -> cancel.

    This shows what happens when a long-running node doesn't finish within
    the graceful period after drain is requested.
    """

    class State(TypedDict, total=False):
        first: str
        second: str

    node_started = asyncio.Event()
    node_cancelled = asyncio.Event()
    node_finished = asyncio.Event()

    async def slow_node(state: State) -> dict[str, str]:
        node_started.set()
        try:
            await asyncio.sleep(30)  # very long operation
        except asyncio.CancelledError:
            node_cancelled.set()
            raise
        node_finished.set()
        return {"first": "done"}

    async def second_node(state: State) -> dict[str, str]:
        return {"second": "should-not-run"}

    graph = StateGraph(State)
    graph.add_node("first", slow_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    control = RunControl()
    compiled = graph.compile()

    # Phase 1: start graph
    graph_task = asyncio.create_task(compiled.ainvoke({}, control=control))

    # Phase 2: wait for node to start, then request drain
    await node_started.wait()
    control.request_drain("sigterm")

    # Phase 3: graceful timeout — node is still running, cancel after 1s
    graceful_timeout = 1.0
    await asyncio.sleep(graceful_timeout)

    assert not node_finished.is_set(), "node should still be running"
    assert not node_cancelled.is_set(), "node should not be cancelled yet"

    # Phase 4: force cancel
    graph_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await graph_task

    # The node received CancelledError at the await point
    assert node_cancelled.is_set(), "node should have received CancelledError"
    assert not node_finished.is_set(), "node should NOT have finished normally"


@pytest.mark.anyio
async def test_cancel_ainvoke_with_async_node() -> None:
    """Cancel ainvoke running an async node: CancelledError is delivered
    at the await point and the node stops immediately."""

    class State(TypedDict, total=False):
        first: str
        second: str

    node_started = asyncio.Event()
    node_cancelled = False
    node_finished = False

    async def slow_async_node(state: State) -> dict[str, str]:
        nonlocal node_cancelled, node_finished
        node_started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            node_cancelled = True
            raise
        node_finished = True
        return {"first": "done"}

    async def second_node(state: State) -> dict[str, str]:
        return {"second": "should-not-run"}

    graph = StateGraph(State)
    graph.add_node("first", slow_async_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    compiled = graph.compile()
    graph_task = asyncio.create_task(compiled.ainvoke({}))

    await node_started.wait()
    graph_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await graph_task

    assert node_cancelled, "async node received CancelledError at await point"
    assert not node_finished, "async node did NOT run to completion"


@pytest.mark.anyio
async def test_cancel_ainvoke_with_sync_node() -> None:
    """Cancel ainvoke running a sync node.

    Sync nodes in ainvoke run on a separate thread (via run_coroutine_threadsafe),
    NOT on the event loop thread. Cancelling the asyncio task disconnects from
    the thread, but the thread keeps running as an orphan and completes on its own.

    Key difference from async nodes:
    - async node: CancelledError stops the coroutine at an await point
    - sync node: cancel only disconnects asyncio; the thread runs to completion

    In shutdown case, we will ignore this because the instance will be destroyed soon.
    """

    class State(TypedDict, total=False):
        first: str
        second: str

    node_started = threading.Event()
    node_finished = threading.Event()

    def slow_sync_node(state: State) -> dict[str, str]:
        node_started.set()
        time.sleep(1)
        node_finished.set()
        return {"first": "done"}

    def second_node(state: State) -> dict[str, str]:
        return {"second": "should-not-run"}

    graph = StateGraph(State)
    graph.add_node("first", slow_sync_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    control = RunControl()
    compiled = graph.compile()

    graph_task = asyncio.create_task(compiled.ainvoke({}, control=control))

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, node_started.wait, 5)

    # Node is now running time.sleep(1) on a background thread.
    # Cancel disconnects the asyncio side; drain is set for good measure.
    graph_task.cancel()
    control.request_drain("sigterm")

    with pytest.raises(asyncio.CancelledError):
        await graph_task

    # At this point the asyncio task is done, but the sync node's thread
    # is still running (orphaned). It hasn't finished yet.
    assert not node_finished.is_set(), (
        "sync node should still be running in its background thread"
    )

    # Wait for the orphaned thread to complete on its own.
    await loop.run_in_executor(None, node_finished.wait, 5)

    assert node_finished.is_set(), (
        "orphaned thread eventually completed — cancel() only disconnected "
        "asyncio, the thread itself was never interrupted"
    )


def test_drain_with_control_parameter_sync() -> None:
    """Control parameter is wired through invoke -> stream."""

    class State(TypedDict, total=False):
        value: str

    def node(state: State) -> dict[str, str]:
        return {"value": "done"}

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)

    # Pre-drained control should immediately stop after first step
    control = RunControl()
    control.request_drain("pre-drained")

    # Graph has only one node in first step, so it completes that step
    # then drain prevents any further steps
    with pytest.raises(GraphDrained, match="pre-drained"):
        graph.compile().invoke({}, control=control)
