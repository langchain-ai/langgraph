import asyncio
import sys
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Annotated, Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from typing_extensions import TypedDict

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_RUNTIME,
    CONFIG_KEY_SEND,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_THREAD_ID,
    CONFIG_KEY_TIMED_ATTEMPT_OBSERVER,
    ERROR,
    GRAPH_ERROR_INFO,
)
from langgraph._internal._runnable import RunnableCallable
from langgraph._internal._timeout import coerce_timeout_policy
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.errors import GraphInterrupt, NodeTimeoutError, ParentCommand
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.pregel._algo import (
    _read_errors_for_task_ids_from_pending_writes,
    _read_failed_node_names_from_pending_writes,
)
from langgraph.pregel._read import PregelNode
from langgraph.pregel._retry import (
    _checkpoint_ns_for_parent_command,
    _ensure_execution_info,
    _should_retry_on,
    _TimedAttemptScope,
    arun_with_retry,
    run_with_retry,
)
from langgraph.pregel.protocol import StreamProtocol
from langgraph.runtime import DEFAULT_RUNTIME, ExecutionInfo, Runtime
from langgraph.types import (
    Command,
    PregelExecutableTask,
    RetryPolicy,
    Send,
    TimeoutPolicy,
)

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


def test_should_retry_on_single_exception():
    """Test retry with a single exception type."""
    policy = RetryPolicy(retry_on=ValueError)

    # Should retry on ValueError
    assert _should_retry_on(policy, ValueError("test error")) is True

    # Should not retry on other exceptions
    assert _should_retry_on(policy, TypeError("test error")) is False
    assert _should_retry_on(policy, Exception("test error")) is False


def test_should_retry_on_sequence_of_exceptions():
    """Test retry with a sequence of exception types."""
    policy = RetryPolicy(retry_on=(ValueError, KeyError))

    # Should retry on listed exceptions
    assert _should_retry_on(policy, ValueError("test error")) is True
    assert _should_retry_on(policy, KeyError("test error")) is True

    # Should not retry on other exceptions
    assert _should_retry_on(policy, TypeError("test error")) is False
    assert _should_retry_on(policy, Exception("test error")) is False


def test_should_retry_on_subclass_of_exception():
    """Test retry on subclass of specified exception."""

    class CustomError(ValueError):
        pass

    policy = RetryPolicy(retry_on=ValueError)

    # Should retry on subclass of specified exception
    assert _should_retry_on(policy, CustomError("test error")) is True


def test_should_retry_on_callable():
    """Test retry with a callable predicate."""

    # Only retry on ValueError with message containing 'retry'
    def should_retry(exc: Exception) -> bool:
        return isinstance(exc, ValueError) and "retry" in str(exc)

    policy = RetryPolicy(retry_on=should_retry)

    # Should retry when predicate returns True
    assert _should_retry_on(policy, ValueError("please retry this")) is True

    # Should not retry when predicate returns False
    assert _should_retry_on(policy, ValueError("other error")) is False
    assert _should_retry_on(policy, TypeError("please retry this")) is False


def test_should_retry_on_invalid_type():
    """Test retry with an invalid retry_on type."""
    policy = RetryPolicy(retry_on=123)  # type: ignore

    with pytest.raises(TypeError, match="retry_on must be an Exception class"):
        _should_retry_on(policy, ValueError("test error"))


def test_should_retry_on_empty_sequence():
    """Test retry with an empty sequence."""
    policy = RetryPolicy(retry_on=())

    # Should not retry when sequence is empty
    assert _should_retry_on(policy, ValueError("test error")) is False


def test_checkpoint_ns_for_parent_command() -> None:
    assert _checkpoint_ns_for_parent_command("") == ""
    assert _checkpoint_ns_for_parent_command("node:1") == ""
    assert _checkpoint_ns_for_parent_command("node:1|child:2") == "node:1"
    assert _checkpoint_ns_for_parent_command("node:1|1|child:2") == "node:1"
    assert _checkpoint_ns_for_parent_command("node:1|1|child:2|1") == "node:1"
    assert (
        _checkpoint_ns_for_parent_command("parent:1|1|child:1|1|node:1|1")
        == "parent:1|1|child:1"
    )
    assert (
        _checkpoint_ns_for_parent_command("parent:1|1|child:1|1|node:1")
        == "parent:1|1|child:1"
    )


def test_should_retry_default_retry_on():
    """Test the default retry_on function."""
    import httpx
    import requests

    # Create a RetryPolicy with default_retry_on
    policy = RetryPolicy()

    # Should retry on ConnectionError
    assert _should_retry_on(policy, ConnectionError("connection refused")) is True

    # Should not retry on common programming errors
    assert _should_retry_on(policy, ValueError("invalid value")) is False
    assert _should_retry_on(policy, TypeError("invalid type")) is False
    assert _should_retry_on(policy, ArithmeticError("division by zero")) is False
    assert _should_retry_on(policy, ImportError("module not found")) is False
    assert _should_retry_on(policy, LookupError("key not found")) is False
    assert _should_retry_on(policy, NameError("name not defined")) is False
    assert _should_retry_on(policy, SyntaxError("invalid syntax")) is False
    assert _should_retry_on(policy, RuntimeError("runtime error")) is False
    assert _should_retry_on(policy, ReferenceError("weak reference")) is False
    assert _should_retry_on(policy, StopIteration()) is False
    assert _should_retry_on(policy, StopAsyncIteration()) is False
    assert _should_retry_on(policy, OSError("file not found")) is False

    # Should retry on httpx.HTTPStatusError with 5xx status code
    response_5xx = Mock()
    response_5xx.status_code = 503
    http_error_5xx = httpx.HTTPStatusError(
        "server error", request=Mock(), response=response_5xx
    )
    assert _should_retry_on(policy, http_error_5xx) is True

    # Should not retry on httpx.HTTPStatusError with 4xx status code
    response_4xx = Mock()
    response_4xx.status_code = 404
    http_error_4xx = httpx.HTTPStatusError(
        "not found", request=Mock(), response=response_4xx
    )
    assert _should_retry_on(policy, http_error_4xx) is False

    # Should retry on requests.HTTPError with 5xx status code
    response_req_5xx = Mock()
    response_req_5xx.status_code = 502
    req_error_5xx = requests.HTTPError("bad gateway")
    req_error_5xx.response = response_req_5xx
    assert _should_retry_on(policy, req_error_5xx) is True

    # Should not retry on requests.HTTPError with 4xx status code
    response_req_4xx = Mock()
    response_req_4xx.status_code = 400
    req_error_4xx = requests.HTTPError("bad request")
    req_error_4xx.response = response_req_4xx
    assert _should_retry_on(policy, req_error_4xx) is False

    # Should retry on requests.HTTPError with no response
    req_error_no_resp = requests.HTTPError("connection error")
    req_error_no_resp.response = None
    assert _should_retry_on(policy, req_error_no_resp) is True

    # Should retry on other exceptions by default
    class CustomException(Exception):
        pass

    assert _should_retry_on(policy, CustomException("custom error")) is True


def test_graph_with_single_retry_policy():
    """Test a simple graph with a single RetryPolicy for a node."""

    class State(TypedDict):
        foo: str

    attempt_count = 0
    attempt_numbers: list[int] = []
    first_attempt_times: list[float | None] = []

    def failing_node(state: State, runtime: Runtime):
        nonlocal attempt_count
        attempt_count += 1
        assert runtime.execution_info.node_attempt == attempt_count
        attempt_numbers.append(runtime.execution_info.node_attempt)
        first_attempt_times.append(runtime.execution_info.node_first_attempt_time)
        if attempt_count < 3:  # Fail the first two attempts
            raise ValueError("Intentional failure")
        return {"foo": "success"}

    def other_node(state: State):
        return {"foo": "other_node"}

    # Create a retry policy with specific parameters
    retry_policy = RetryPolicy(
        max_attempts=3,
        initial_interval=0.01,  # Short interval for tests
        backoff_factor=2.0,
        jitter=False,  # Disable jitter for predictable timing
        retry_on=ValueError,
    )

    # Create and compile the graph
    graph = (
        StateGraph(State)
        .add_node("failing_node", failing_node, retry_policy=retry_policy)
        .add_node("other_node", other_node)
        .add_edge(START, "failing_node")
        .add_edge("failing_node", "other_node")
        .compile()
    )

    with patch("time.sleep") as mock_sleep:
        result = graph.invoke({"foo": ""})

    # Verify retry behavior
    assert attempt_count == 3  # The node should have been tried 3 times
    assert attempt_numbers == [1, 2, 3]
    assert len(first_attempt_times) == 3
    assert first_attempt_times[0] is not None
    assert first_attempt_times[1] == first_attempt_times[0]
    assert first_attempt_times[2] == first_attempt_times[0]
    assert result["foo"] == "other_node"  # Final result should be from other_node

    # Verify the sleep intervals
    call_args_list = [args[0][0] for args in mock_sleep.call_args_list]
    assert call_args_list == [0.01, 0.02]


def test_runtime_execution_info_defaults_without_retry():
    """Test execution_info defaults when no retry and no config are provided."""

    class State(TypedDict):
        foo: str

    captured = {}

    def node(state: State, runtime: Runtime):
        captured["node_attempt"] = runtime.execution_info.node_attempt
        captured["node_first_attempt_time"] = (
            runtime.execution_info.node_first_attempt_time
        )
        return {"foo": "ok"}

    graph = StateGraph(State).add_node("node", node).add_edge(START, "node").compile()

    result = graph.invoke({"foo": ""})

    assert result["foo"] == "ok"
    assert captured["node_attempt"] == 1
    assert isinstance(captured["node_first_attempt_time"], float)


def test_graph_with_jitter_retry_policy():
    """Test a graph with a RetryPolicy that uses jitter."""

    class State(TypedDict):
        foo: str

    attempt_count = 0

    def failing_node(state):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:  # Fail the first attempt
            raise ValueError("Intentional failure")
        return {"foo": "success"}

    # Create a retry policy with jitter enabled
    retry_policy = RetryPolicy(
        max_attempts=3,
        initial_interval=0.01,
        jitter=True,  # Enable jitter for randomized backoff
        retry_on=ValueError,
    )

    # Create and compile the graph
    graph = (
        StateGraph(State)
        .add_node("failing_node", failing_node, retry_policy=retry_policy)
        .add_edge(START, "failing_node")
        .compile()
    )

    # Test graph execution with mocked random and sleep
    with (
        patch("random.uniform", return_value=0.05) as mock_random,
        patch("time.sleep") as mock_sleep,
    ):
        result = graph.invoke({"foo": ""})

    # Verify retry behavior
    assert attempt_count == 2  # The node should have been tried twice
    assert result["foo"] == "success"

    # Verify jitter was applied
    mock_random.assert_called_with(0, 1)  # Jitter should use random.uniform(0, 1)
    mock_sleep.assert_called_with(0.01 + 0.05)  # Sleep should include jitter


def test_graph_with_multiple_retry_policies():
    """Test a graph with multiple retry policies for a node."""

    class State(TypedDict):
        foo: str
        error_type: str

    attempt_counts = {"value_error": 0, "key_error": 0}

    def failing_node(state):
        error_type = state["error_type"]

        if error_type == "value_error":
            attempt_counts["value_error"] += 1
            if attempt_counts["value_error"] < 2:
                raise ValueError("Value error")
        elif error_type == "key_error":
            attempt_counts["key_error"] += 1
            if attempt_counts["key_error"] < 3:
                raise KeyError("Key error")

        return {"foo": f"recovered_from_{error_type}"}

    # Create multiple retry policies
    value_error_policy = RetryPolicy(
        max_attempts=2,
        initial_interval=0.01,
        jitter=False,
        retry_on=ValueError,
    )

    key_error_policy = RetryPolicy(
        max_attempts=3,
        initial_interval=0.02,
        jitter=False,
        retry_on=KeyError,
    )

    # Create and compile the graph with a list of retry policies
    graph = (
        StateGraph(State)
        .add_node(
            "failing_node",
            failing_node,
            retry_policy=(value_error_policy, key_error_policy),
        )
        .add_edge(START, "failing_node")
        .compile()
    )

    # Test ValueError scenario
    with patch("time.sleep"):
        result_value_error = graph.invoke({"foo": "", "error_type": "value_error"})

    assert attempt_counts["value_error"] == 2
    assert result_value_error["foo"] == "recovered_from_value_error"

    # Reset attempt counts
    attempt_counts = {"value_error": 0, "key_error": 0}

    # Test KeyError scenario
    with patch("time.sleep"):
        result_key_error = graph.invoke({"foo": "", "error_type": "key_error"})

    assert attempt_counts["key_error"] == 3
    assert result_key_error["foo"] == "recovered_from_key_error"


def test_graph_with_max_attempts_exceeded():
    """Test a graph where max_attempts is exceeded."""

    class State(TypedDict):
        foo: str

    def always_failing_node(state):
        raise ValueError("Always fails")

    # Create a retry policy with limited attempts
    retry_policy = RetryPolicy(
        max_attempts=2,
        initial_interval=0.01,
        jitter=False,
        retry_on=ValueError,
    )

    # Create and compile the graph
    graph = (
        StateGraph(State)
        .add_node("always_failing", always_failing_node, retry_policy=retry_policy)
        .add_edge(START, "always_failing")
        .compile()
    )

    # Test graph execution
    with (
        patch("time.sleep") as mock_sleep,
        pytest.raises(ValueError, match="Always fails"),
    ):
        graph.invoke({"foo": ""})

    mock_sleep.assert_called_with(0.01)


def test_execution_info_identity_fields_populated_on_retry():
    """Test that thread_id, task_id, run_id, etc. are populated in execution_info during retries."""

    class State(TypedDict):
        foo: str

    attempt_count = 0
    captured_infos: list[dict] = []

    def failing_node(state: State, runtime: Runtime):
        nonlocal attempt_count
        attempt_count += 1
        info = runtime.execution_info
        captured_infos.append(
            {
                "thread_id": info.thread_id,
                "run_id": info.run_id,
                "node_attempt": info.node_attempt,
                "node_first_attempt_time": info.node_first_attempt_time,
                "checkpoint_ns": info.checkpoint_ns,
            }
        )
        if attempt_count < 2:
            raise ValueError("Intentional failure")
        return {"foo": "success"}

    retry_policy = RetryPolicy(
        max_attempts=3,
        initial_interval=0.01,
        jitter=False,
        retry_on=ValueError,
    )

    graph = (
        StateGraph(State)
        .add_node("failing_node", failing_node, retry_policy=retry_policy)
        .add_edge(START, "failing_node")
        .compile(checkpointer=MemorySaver())
    )

    with patch("time.sleep"):
        result = graph.invoke(
            {"foo": ""},
            config={"configurable": {"thread_id": "retry-thread"}},
        )

    assert result["foo"] == "success"
    assert len(captured_infos) == 2

    # Both attempts should have the same thread_id and first_attempt_time
    assert captured_infos[0]["thread_id"] == "retry-thread"
    assert captured_infos[1]["thread_id"] == "retry-thread"
    assert (
        captured_infos[0]["node_first_attempt_time"]
        == captured_infos[1]["node_first_attempt_time"]
    )

    # node_attempt should increment
    assert captured_infos[0]["node_attempt"] == 1
    assert captured_infos[1]["node_attempt"] == 2


def test_ensure_execution_info_noop_when_already_set():
    """Test that _ensure_execution_info is a no-op when execution_info exists."""
    existing_info = ExecutionInfo(
        checkpoint_id="cp-1", checkpoint_ns="ns-1", task_id="task-1"
    )
    runtime = DEFAULT_RUNTIME.override(execution_info=existing_info)
    config = {CONF: {CONFIG_KEY_THREAD_ID: "thread-1"}}
    task = Mock(id="task-2")

    result = _ensure_execution_info(runtime, config, task)
    assert result is runtime
    assert result.execution_info is existing_info


def test_ensure_execution_info_creates_from_config():
    """Test that _ensure_execution_info creates ExecutionInfo from config when missing."""
    runtime = DEFAULT_RUNTIME.override(execution_info=None)
    config = {
        "run_id": "run-123",
        CONF: {
            CONFIG_KEY_CHECKPOINT_ID: "cp-42",
            CONFIG_KEY_CHECKPOINT_NS: "ns-42",
            CONFIG_KEY_TASK_ID: "task-42",
            CONFIG_KEY_THREAD_ID: "thread-42",
        },
    }
    task = Mock(id="fallback-task-id")

    result = _ensure_execution_info(runtime, config, task)
    assert result.execution_info is not None
    assert result.execution_info.checkpoint_id == "cp-42"
    assert result.execution_info.checkpoint_ns == "ns-42"
    assert result.execution_info.task_id == "task-42"
    assert result.execution_info.thread_id == "thread-42"
    assert result.execution_info.run_id == "run-123"


def test_ensure_execution_info_falls_back_to_task_id():
    """Test that _ensure_execution_info uses task.id when CONFIG_KEY_TASK_ID is missing."""
    runtime = DEFAULT_RUNTIME.override(execution_info=None)
    config = {CONF: {}}
    task = Mock(id="fallback-task-id")

    result = _ensure_execution_info(runtime, config, task)
    assert result.execution_info.task_id == "fallback-task-id"
    assert result.execution_info.checkpoint_id == ""
    assert result.execution_info.checkpoint_ns == ""


def test_run_with_retry_creates_execution_info_when_missing():
    """Test that run_with_retry works when runtime has no execution_info (distributed runtime scenario)."""
    captured_infos: list[ExecutionInfo] = []

    class FakeProc:
        def invoke(self, input, config):
            runtime = config[CONF][CONFIG_KEY_RUNTIME]
            captured_infos.append(runtime.execution_info)
            return input

    runtime = DEFAULT_RUNTIME.override(execution_info=None)
    config = {
        "run_id": "run-abc",
        CONF: {
            CONFIG_KEY_RUNTIME: runtime,
            CONFIG_KEY_CHECKPOINT_ID: "cp-99",
            CONFIG_KEY_CHECKPOINT_NS: "__start__:task123",
            CONFIG_KEY_TASK_ID: "task123",
            CONFIG_KEY_THREAD_ID: "thread-xyz",
        },
    }

    task = PregelExecutableTask(
        name="__start__",
        input={"messages": []},
        proc=FakeProc(),
        writes=deque(),
        config=config,
        triggers=["__start__"],
        retry_policy=[],
        cache_key=None,
        id="task123",
        path=("__pregel_pull", "__start__"),
    )

    run_with_retry(task, retry_policy=None)

    assert len(captured_infos) == 1
    info = captured_infos[0]
    assert info.checkpoint_id == "cp-99"
    assert info.checkpoint_ns == "__start__:task123"
    assert info.task_id == "task123"
    assert info.thread_id == "thread-xyz"
    assert info.run_id == "run-abc"
    assert info.node_attempt == 1
    assert info.node_first_attempt_time is not None


def _make_task(
    proc,
    *,
    timeout=None,
    retry_policy=(),
    name="timed",
    task_id="tid",
    writers=(),
):
    runtime = DEFAULT_RUNTIME.override(execution_info=None)
    writes = deque()
    config = {
        "run_id": "run-x",
        CONF: {
            CONFIG_KEY_RUNTIME: runtime,
            CONFIG_KEY_CHECKPOINT_ID: "cp",
            CONFIG_KEY_CHECKPOINT_NS: f"{name}:{task_id}",
            CONFIG_KEY_SEND: writes.extend,
            CONFIG_KEY_TASK_ID: task_id,
            CONFIG_KEY_THREAD_ID: "thr",
        },
    }
    return PregelExecutableTask(
        name=name,
        input=None,
        proc=proc,
        writes=writes,
        config=config,
        triggers=[name],
        retry_policy=retry_policy,
        cache_key=None,
        id=task_id,
        path=("__pregel_pull", name),
        writers=writers,
        timeout=coerce_timeout_policy(timeout),
    )


def _idle_timeout(value: float | timedelta) -> TimeoutPolicy:
    return TimeoutPolicy(idle_timeout=value)


def test_coerce_timeout_policy_scalar_is_run_timeout():
    assert coerce_timeout_policy(None) is None
    policy = coerce_timeout_policy(timedelta(milliseconds=250))
    assert policy == TimeoutPolicy(run_timeout=0.25)
    assert Send("node", None, timeout=timedelta(milliseconds=250)).timeout == policy

    idle_policy = coerce_timeout_policy(TimeoutPolicy(idle_timeout=1.5))
    assert idle_policy == TimeoutPolicy(idle_timeout=1.5)

    with pytest.raises(ValueError, match="run_timeout must be greater than 0"):
        coerce_timeout_policy(0)


def test_coerce_timeout_policy_returns_same_instance_for_already_coerced():
    policy = coerce_timeout_policy(TimeoutPolicy(run_timeout=1.0, idle_timeout=2.0))
    assert coerce_timeout_policy(policy) is policy
    assert TimeoutPolicy.coerce(policy) is policy


def test_send_timeout_round_trips_through_msgpack_serde():
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    packet = Send(
        "worker",
        {"x": 1},
        timeout=TimeoutPolicy(run_timeout=1, idle_timeout=2),
    )

    assert serde.loads_typed(serde.dumps_typed(packet)) == packet


def test_send_without_timeout_round_trips_through_msgpack_serde():
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    packet = Send("worker", {"x": 1})

    assert serde.loads_typed(serde.dumps_typed(packet)) == packet


def test_run_with_retry_rejects_sync_timeout_without_starting_proc():
    started = False

    class Proc:
        def invoke(self, input, config):
            nonlocal started
            started = True
            return input

    task = _make_task(Proc(), timeout=_idle_timeout(0.05), name="sync")

    with pytest.raises(ValueError, match="only supported for async nodes"):
        run_with_retry(task, retry_policy=None)
    assert not started


def test_run_with_retry_without_timeout_runs_sync_directly():
    class FastProc:
        def invoke(self, input, config):
            return "ok"

    task = _make_task(FastProc(), timeout=None)
    assert run_with_retry(task, retry_policy=None) == "ok"


def test_idle_timeout_guard_call_does_not_hold_scope_lock():
    scope = _TimedAttemptScope()

    def call():
        assert not scope._lock.locked()
        return "ok"

    assert scope._guard_call(call)() == "ok"


def test_idle_timeout_guard_stream_does_not_hold_scope_lock():
    scope = _TimedAttemptScope()

    def stream(chunk):
        assert not scope._lock.locked()
        assert chunk == ((), "custom", "ok")

    scope._guard_stream(StreamProtocol(stream, {"custom"}))(((), "custom", "ok"))


def test_idle_timeout_guard_stream_writer_does_not_hold_scope_lock():
    scope = _TimedAttemptScope()

    def stream_writer(chunk):
        assert not scope._lock.locked()
        assert chunk == "ok"

    scope._guard_stream_writer(stream_writer)("ok")


@pytest.mark.anyio
async def test_arun_with_retry_timeout_ok_when_fast():
    class FastProc:
        async def ainvoke(self, input, config):
            return "ok"

    task = _make_task(FastProc(), timeout=_idle_timeout(1.0))
    assert await arun_with_retry(task, retry_policy=None) == "ok"


@pytest.mark.anyio
async def test_arun_with_retry_timeout_retries_when_retry_on_timeout():
    calls: list[float] = []

    class FlakyProc:
        async def ainvoke(self, input, config):
            calls.append(time.monotonic())
            if len(calls) < 2:
                await asyncio.sleep(0.5)
                return "late"
            return "ok"

    policy = RetryPolicy(
        max_attempts=3,
        initial_interval=0.0,
        jitter=False,
        retry_on=NodeTimeoutError,
    )
    task = _make_task(FlakyProc(), timeout=_idle_timeout(0.05), retry_policy=(policy,))
    assert await arun_with_retry(task, retry_policy=None) == "ok"
    assert len(calls) == 2


@pytest.mark.anyio
async def test_entrypoint_timeout_allows_pre_timeout_child_task_to_run():
    child_started = threading.Event()

    @task()
    def child(value: int) -> int:
        child_started.set()
        return value + 1

    @entrypoint(timeout=TimeoutPolicy(idle_timeout=0.05))
    async def parent(value: int) -> int:
        child(value)
        await asyncio.sleep(0.2)
        return value

    with pytest.raises(NodeTimeoutError):
        await parent.ainvoke(1)
    assert child_started.wait(timeout=1.0)


@pytest.mark.anyio
async def test_arun_with_retry_timeout_accepts_timedelta():
    class SlowProc:
        async def ainvoke(self, input, config):
            await asyncio.sleep(0.5)
            return input

    task = _make_task(SlowProc(), timeout=_idle_timeout(timedelta(milliseconds=50)))
    with pytest.raises(NodeTimeoutError):
        await arun_with_retry(task, retry_policy=None)


@pytest.mark.anyio
async def test_arun_with_retry_timeout_fires_async():
    class SlowProc:
        async def ainvoke(self, input, config):
            await asyncio.sleep(1.0)
            return input

    task = _make_task(SlowProc(), timeout=_idle_timeout(0.05), name="aslow")
    with pytest.raises(NodeTimeoutError) as excinfo:
        await arun_with_retry(task, retry_policy=None)
    assert excinfo.value.node == "aslow"
    assert excinfo.value.idle_timeout == 0.05


@pytest.mark.anyio
async def test_arun_with_retry_run_timeout_is_not_refreshed_by_heartbeat():
    class HeartbeatingProc:
        async def ainvoke(self, input, config):
            runtime = config[CONF][CONFIG_KEY_RUNTIME]
            while True:
                runtime.heartbeat()
                await asyncio.sleep(0.01)

    task = _make_task(HeartbeatingProc(), timeout=0.05, name="run-timeout")
    with pytest.raises(NodeTimeoutError) as excinfo:
        await arun_with_retry(task, retry_policy=None)
    assert excinfo.value.kind == "run"
    assert excinfo.value.run_timeout == 0.05
    assert excinfo.value.idle_timeout is None


@pytest.mark.anyio
async def test_node_timeout_error_carries_both_configured_timeouts():
    """Both `idle_timeout` and `run_timeout` reflect the configured policy
    even when only one of them fires."""

    class SlowProc:
        async def ainvoke(self, input, config):
            await asyncio.sleep(1.0)

    task = _make_task(
        SlowProc(),
        timeout=TimeoutPolicy(run_timeout=0.05, idle_timeout=0.5),
        name="both",
    )
    with pytest.raises(NodeTimeoutError) as excinfo:
        await arun_with_retry(task, retry_policy=None)
    assert excinfo.value.kind == "run"
    assert excinfo.value.run_timeout == 0.05
    assert excinfo.value.idle_timeout == 0.5
    # `timeout` is the one that fired.
    assert excinfo.value.timeout == 0.05


@pytest.mark.anyio
async def test_arun_with_retry_does_not_swallow_proc_asyncio_timeout():
    calls = 0

    class InnerTimeoutProc:
        async def ainvoke(self, input, config):
            nonlocal calls
            calls += 1
            raise asyncio.TimeoutError("inner")

    # `retry_on=NodeTimeoutError` + `calls == 1` is the load-bearing assertion:
    # if the proc's TimeoutError were misclassified as NodeTimeoutError it
    # would be retried, and `calls` would be 2.
    policy = RetryPolicy(
        max_attempts=2,
        initial_interval=0.0,
        jitter=False,
        retry_on=NodeTimeoutError,
    )
    task = _make_task(
        InnerTimeoutProc(),
        timeout=_idle_timeout(1.0),
        retry_policy=(policy,),
        name="parent",
    )
    with pytest.raises(asyncio.TimeoutError, match="inner"):
        await arun_with_retry(task, retry_policy=None)
    assert calls == 1


@pytest.mark.anyio
async def test_arun_with_retry_does_not_swallow_proc_node_timeout():
    child_timeout = NodeTimeoutError("child", 0.2, kind="idle", idle_timeout=0.1)

    class ChildTimeoutProc:
        async def ainvoke(self, input, config):
            raise child_timeout

    task = _make_task(ChildTimeoutProc(), timeout=_idle_timeout(1.0), name="parent")
    with pytest.raises(NodeTimeoutError) as excinfo:
        await arun_with_retry(task, retry_policy=None)
    assert excinfo.value is child_timeout
    assert excinfo.value.node == "child"


@pytest.mark.anyio
async def test_arun_with_retry_idle_timeout_resets_on_stream_event():
    events = []

    class StreamingProc:
        async def ainvoke(self, input, config):
            for _ in range(3):
                await asyncio.sleep(0.08)
                config[CONF][CONFIG_KEY_STREAM](((), "custom", "tick"))
            return "ok"

    task = _make_task(StreamingProc(), timeout=_idle_timeout(0.2), name="streaming")
    task.config[CONF][CONFIG_KEY_STREAM] = StreamProtocol(events.append, {"custom"})
    assert await arun_with_retry(task, retry_policy=None) == "ok"
    assert len(events) == 3


@pytest.mark.anyio
async def test_arun_with_retry_idle_timeout_resets_on_runtime_stream_writer():
    events = []

    class WriterProc:
        async def ainvoke(self, input, config):
            runtime = config[CONF][CONFIG_KEY_RUNTIME]
            for _ in range(3):
                await asyncio.sleep(0.08)
                runtime.stream_writer("tick")
            return "ok"

    task = _make_task(WriterProc(), timeout=_idle_timeout(0.2), name="writer")
    runtime = task.config[CONF][CONFIG_KEY_RUNTIME]
    task.config[CONF][CONFIG_KEY_RUNTIME] = runtime.override(
        stream_writer=events.append
    )
    assert await arun_with_retry(task, retry_policy=None) == "ok"
    assert events == ["tick", "tick", "tick"]


@pytest.mark.anyio
async def test_astream_with_retry_idle_timeout_resets_on_yielded_chunks():
    class StreamingProc:
        async def astream(self, input, config):
            for i in range(3):
                await asyncio.sleep(0.08)
                yield i

    task = _make_task(StreamingProc(), timeout=_idle_timeout(0.2), name="astream")
    await arun_with_retry(task, retry_policy=None, stream=True)


class _HandlerEmittingProc:
    """Proc that fires `on_llm_new_token` on every handler attached to its config."""

    def __init__(self, iterations: int = 1, sleep_s: float = 0.0) -> None:
        self.iterations = iterations
        self.sleep_s = sleep_s

    async def ainvoke(self, input, config):
        run_id = uuid4()
        for _ in range(self.iterations):
            if self.sleep_s:
                await asyncio.sleep(self.sleep_s)
            for handler in config["callbacks"]:
                handler.on_llm_new_token("tok", run_id=run_id)
        return "ok"


@pytest.mark.anyio
async def test_arun_with_retry_idle_timeout_resets_on_runtime_heartbeat():
    class HeartbeatProc:
        async def ainvoke(self, input, config):
            runtime = config[CONF][CONFIG_KEY_RUNTIME]
            for _ in range(3):
                await asyncio.sleep(0.08)
                runtime.heartbeat()
            return "ok"

    task = _make_task(HeartbeatProc(), timeout=_idle_timeout(0.15), name="heartbeat")
    assert await arun_with_retry(task, retry_policy=None) == "ok"


@pytest.mark.anyio
async def test_arun_with_retry_heartbeat_refresh_mode_ignores_stream_events():
    events = []

    class StreamingProc:
        async def ainvoke(self, input, config):
            while True:
                await asyncio.sleep(0.01)
                config[CONF][CONFIG_KEY_STREAM](((), "custom", "tick"))

    task = _make_task(
        StreamingProc(),
        timeout=TimeoutPolicy(idle_timeout=0.05, refresh_on="heartbeat"),
        name="heartbeat-only",
    )
    task.config[CONF][CONFIG_KEY_STREAM] = StreamProtocol(events.append, {"custom"})
    with pytest.raises(NodeTimeoutError) as excinfo:
        await arun_with_retry(task, retry_policy=None)
    assert excinfo.value.kind == "idle"
    assert events


@pytest.mark.anyio
async def test_arun_with_retry_heartbeat_refresh_mode_accepts_heartbeat():
    class HeartbeatProc:
        async def ainvoke(self, input, config):
            runtime = config[CONF][CONFIG_KEY_RUNTIME]
            for _ in range(3):
                await asyncio.sleep(0.03)
                runtime.heartbeat()
            return "ok"

    task = _make_task(
        HeartbeatProc(),
        timeout=TimeoutPolicy(idle_timeout=0.08, refresh_on="heartbeat"),
        name="heartbeat-only",
    )
    assert await arun_with_retry(task, retry_policy=None) == "ok"


def test_runtime_heartbeat_outside_idle_attempt_is_no_op():
    DEFAULT_RUNTIME.heartbeat()


@pytest.mark.anyio
async def test_arun_with_retry_idle_timeout_resets_on_callback_event():
    task = _make_task(
        _HandlerEmittingProc(iterations=3, sleep_s=0.08),
        timeout=_idle_timeout(0.15),
        name="cb",
    )
    assert await arun_with_retry(task, retry_policy=None) == "ok"


@pytest.mark.anyio
async def test_arun_with_retry_idle_timeout_preserves_existing_callbacks():
    seen: list[str] = []

    class RecordingHandler(BaseCallbackHandler):
        run_inline = True

        def on_llm_new_token(self, token, *, run_id, **kwargs):
            seen.append(token)

    task = _make_task(_HandlerEmittingProc(), timeout=_idle_timeout(0.5), name="cb-pre")
    task.config["callbacks"] = [RecordingHandler()]
    assert await arun_with_retry(task, retry_policy=None) == "ok"
    assert seen == ["tok"]


@pytest.mark.anyio
async def test_arun_with_retry_timeout_discards_stale_executor_writes():
    release_first_attempt = threading.Event()

    class FlakyAsyncProc:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, input, config):
            self.calls += 1
            if self.calls == 1:

                def late_write() -> str:
                    release_first_attempt.wait(timeout=1.0)
                    config[CONF][CONFIG_KEY_SEND]([("value", "stale")])
                    return "late"

                return await asyncio.to_thread(late_write)
            release_first_attempt.set()
            config[CONF][CONFIG_KEY_SEND]([("value", "fresh")])
            return "ok"

    policy = RetryPolicy(
        max_attempts=2,
        initial_interval=0.0,
        jitter=False,
        retry_on=NodeTimeoutError,
    )
    task = _make_task(
        FlakyAsyncProc(), timeout=_idle_timeout(0.05), retry_policy=(policy,)
    )
    assert await arun_with_retry(task, retry_policy=None) == "ok"
    await asyncio.sleep(0.05)
    assert task.writes == deque([("value", "fresh")])


@pytest.mark.anyio
async def test_arun_with_retry_timeout_discards_pre_timeout_writes():
    class SlowAsyncWriterProc:
        async def ainvoke(self, input, config):
            config[CONF][CONFIG_KEY_SEND]([("value", "stale-before-idle-timeout")])
            await asyncio.sleep(0.2)
            return "late"

    task = _make_task(
        SlowAsyncWriterProc(), timeout=_idle_timeout(0.05), name="aslow-writer"
    )
    with pytest.raises(NodeTimeoutError):
        await arun_with_retry(task, retry_policy=None)
    assert task.writes == deque()


@pytest.mark.anyio
async def test_astream_with_retry_timeout_discards_pre_timeout_writes():
    class SlowStreamWriterProc:
        async def astream(self, input, config):
            config[CONF][CONFIG_KEY_SEND]([("value", "stale-before-idle-timeout")])
            await asyncio.sleep(0.2)
            if False:
                yield None

    task = _make_task(
        SlowStreamWriterProc(), timeout=_idle_timeout(0.05), name="astream-writer"
    )
    with pytest.raises(NodeTimeoutError):
        await arun_with_retry(task, retry_policy=None, stream=True)
    assert task.writes == deque()


@pytest.mark.anyio
async def test_arun_with_retry_timeout_cannot_be_swallowed():
    class StubbornProc:
        async def ainvoke(self, input, config):
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                config[CONF][CONFIG_KEY_SEND]([("value", "stale")])
                await asyncio.sleep(0)
                return "late"
            return "ok"

    task = _make_task(StubbornProc(), timeout=_idle_timeout(0.05), name="stubborn")
    with pytest.raises(NodeTimeoutError) as excinfo:
        await arun_with_retry(task, retry_policy=None)
    assert excinfo.value.node == "stubborn"
    await asyncio.sleep(0.05)
    assert task.writes == deque()


@pytest.mark.anyio
async def test_astream_with_retry_timeout_cannot_be_swallowed():
    class StubbornStreamProc:
        async def astream(self, input, config):
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                config[CONF][CONFIG_KEY_SEND]([("value", "stale")])
                await asyncio.sleep(0)
                if False:
                    yield None
                return
            yield "ok"

    task = _make_task(
        StubbornStreamProc(), timeout=_idle_timeout(0.05), name="stubborn-stream"
    )
    with pytest.raises(NodeTimeoutError) as excinfo:
        await arun_with_retry(task, retry_policy=None, stream=True)
    assert excinfo.value.node == "stubborn-stream"
    await asyncio.sleep(0.05)
    assert task.writes == deque()


class _TimeoutState(TypedDict):
    x: int


def test_timeout_validation_is_eager_across_apis():
    with pytest.raises(ValueError, match="greater than 0"):
        task(timeout=0)

    with pytest.raises(ValueError, match="greater than 0"):
        entrypoint(timeout=0)

    with pytest.raises(ValueError, match="greater than 0"):
        NodeBuilder().set_timeout(0)

    with pytest.raises(ValueError, match="greater than 0"):
        PregelNode(channels="x", triggers=["x"], timeout=0)

    with pytest.raises(ValueError, match="greater than 0"):
        Send("slow", {}, timeout=0)

    builder = StateGraph(_TimeoutState)
    with pytest.raises(ValueError, match="greater than 0"):
        builder.add_node("slow", lambda state: state, timeout=0)


def test_timeout_rejects_sync_functional_apis_at_declaration_time():
    with pytest.raises(ValueError, match="only supported for async nodes"):

        @task(timeout=TimeoutPolicy(idle_timeout=0.05))
        def sync_task(value: int) -> int:
            return value

    with pytest.raises(ValueError, match="only supported for async nodes"):

        @entrypoint(timeout=TimeoutPolicy(idle_timeout=0.05))
        def sync_entrypoint(value: int) -> int:
            return value


def test_state_graph_compile_rejects_sync_node_timeout():
    def slow(state: _TimeoutState) -> _TimeoutState:
        return {"x": state["x"] + 1}

    builder = StateGraph(_TimeoutState)
    builder.add_node("slow", slow, timeout=TimeoutPolicy(idle_timeout=0.05))
    builder.add_edge(START, "slow")
    builder.add_edge("slow", END)

    with pytest.raises(ValueError, match="only supported for async nodes"):
        builder.compile()


def test_pregel_validate_rejects_sync_writer_timeout():
    async def bound(value: int) -> int:
        return value + 1

    def sync_writer(value: int) -> int:
        return value

    with pytest.raises(ValueError, match="only supported for async nodes"):
        Pregel(
            nodes={
                "slow": PregelNode(
                    channels="input",
                    triggers=["input"],
                    bound=RunnableLambda(bound),
                    writers=[RunnableLambda(sync_writer)],
                    timeout=TimeoutPolicy(run_timeout=1),
                )
            },
            channels={
                "input": EphemeralValue(int),
                "output": LastValue(int),
            },
            input_channels="input",
            output_channels="output",
        )


def test_pregel_validate_rejects_wrapped_sync_runnable_lambda_timeout():
    def slow(value: int) -> int:
        return value + 1

    with pytest.raises(ValueError, match="only supported for async nodes"):
        Pregel(
            nodes={
                "slow": (
                    NodeBuilder()
                    .subscribe_only("input")
                    .do(RunnableLambda(slow).with_config(tags=["wrapped"]))
                    .set_timeout(TimeoutPolicy(idle_timeout=0.05))
                    .write_to("output")
                )
            },
            channels={
                "input": EphemeralValue(int),
                "output": LastValue(int),
            },
            input_channels="input",
            output_channels="output",
        )


def test_pregel_validate_accepts_wrapped_async_runnable_lambda_timeout():
    async def slow(value: int) -> int:
        return value + 1

    Pregel(
        nodes={
            "slow": (
                NodeBuilder()
                .subscribe_only("input")
                .do(RunnableLambda(slow).with_config(tags=["wrapped"]))
                .set_timeout(TimeoutPolicy(idle_timeout=0.05))
                .write_to("output")
            )
        },
        channels={
            "input": EphemeralValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )


def test_pregel_validate_rejects_parallel_sync_branch_timeout():
    def sync_branch(value: int) -> int:
        return value + 1

    async def async_branch(value: int) -> int:
        return value + 1

    with pytest.raises(ValueError, match="only supported for async nodes"):
        Pregel(
            nodes={
                "parallel": (
                    NodeBuilder()
                    .subscribe_only("input")
                    .do(
                        RunnableParallel(
                            sync=RunnableLambda(sync_branch),
                            async_=RunnableLambda(async_branch),
                        )
                    )
                    .set_timeout(TimeoutPolicy(idle_timeout=0.05))
                    .write_to("output")
                )
            },
            channels={
                "input": EphemeralValue(int),
                "output": LastValue(dict),
            },
            input_channels="input",
            output_channels="output",
        )


def test_pregel_validate_rejects_sync_node_timeout():
    def slow(value: int) -> int:
        return value + 1

    with pytest.raises(ValueError, match="only supported for async nodes"):
        Pregel(
            nodes={
                "slow": (
                    NodeBuilder()
                    .subscribe_only("input")
                    .do(slow)
                    .set_timeout(TimeoutPolicy(idle_timeout=0.05))
                    .write_to("output")
                )
            },
            channels={
                "input": EphemeralValue(int),
                "output": LastValue(int),
            },
            input_channels="input",
            output_channels="output",
        )


@pytest.mark.anyio
async def test_pregel_validate_accepts_async_runnable_lambda_timeout():
    async def slow(value: int) -> int:
        await asyncio.sleep(0.2)
        return value + 1

    graph = Pregel(
        nodes={
            "slow": (
                NodeBuilder()
                .subscribe_only("input")
                .do(RunnableLambda(slow))
                .set_timeout(TimeoutPolicy(idle_timeout=0.05))
                .write_to("output")
            )
        },
        channels={
            "input": EphemeralValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )
    with pytest.raises(NodeTimeoutError):
        await graph.ainvoke(1)


@pytest.mark.anyio
async def test_pregel_validate_accepts_runnable_callable_with_sync_and_async_timeout():
    def sync(value: int) -> int:
        return value + 1

    async def async_(value: int) -> int:
        await asyncio.sleep(0.2)
        return value + 1

    graph = Pregel(
        nodes={
            "slow": (
                NodeBuilder()
                .subscribe_only("input")
                .do(RunnableCallable(sync, async_))
                .set_timeout(TimeoutPolicy(idle_timeout=0.05))
                .write_to("output")
            )
        },
        channels={
            "input": EphemeralValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )
    with pytest.raises(NodeTimeoutError):
        await graph.ainvoke(1)


@pytest.mark.anyio
async def test_state_graph_add_node_timeout_e2e():
    async def slow(state: _TimeoutState) -> _TimeoutState:
        await asyncio.sleep(1.0)
        return {"x": state["x"] + 1}

    builder = StateGraph(_TimeoutState)
    builder.add_node("slow", slow, timeout=TimeoutPolicy(idle_timeout=0.05))
    builder.add_edge(START, "slow")
    builder.add_edge("slow", END)
    graph = builder.compile()
    with pytest.raises(NodeTimeoutError):
        await graph.ainvoke({"x": 1})


@pytest.mark.anyio
async def test_send_timeout_overrides_target_node_timeout():
    async def slow(state: _TimeoutState) -> _TimeoutState:
        await asyncio.sleep(0.2)
        return {"x": state["x"] + 1}

    def route(state: _TimeoutState) -> list[Send]:
        return [Send("slow", state, timeout=TimeoutPolicy(idle_timeout=0.05))]

    builder = StateGraph(_TimeoutState)
    builder.add_node("slow", slow, timeout=TimeoutPolicy(idle_timeout=1.0))
    builder.add_conditional_edges(START, route)
    builder.add_edge("slow", END)
    graph = builder.compile()

    with pytest.raises(NodeTimeoutError) as excinfo:
        await graph.ainvoke({"x": 1})
    assert excinfo.value.node == "slow"
    assert excinfo.value.kind == "idle"
    assert excinfo.value.idle_timeout == 0.05


@pytest.mark.anyio
async def test_state_graph_add_node_timeout_composes_with_retry():
    """add_node(..., timeout=TimeoutPolicy(...)) retries then succeeds."""

    attempts: list[int] = []

    async def flaky(state: _TimeoutState) -> _TimeoutState:
        attempts.append(len(attempts))
        if len(attempts) < 2:
            await asyncio.sleep(0.5)
        return {"x": state["x"] + 1}

    builder = StateGraph(_TimeoutState)
    builder.add_node(
        "flaky",
        flaky,
        timeout=TimeoutPolicy(idle_timeout=0.1),
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.0,
            jitter=False,
            retry_on=NodeTimeoutError,
        ),
    )
    builder.add_edge(START, "flaky")
    builder.add_edge("flaky", END)
    graph = builder.compile()
    result = await graph.ainvoke({"x": 0})
    assert result == {"x": 1}
    assert len(attempts) == 2


@NEEDS_CONTEXTVARS
@pytest.mark.anyio
async def test_task_decorator_timeout_e2e():
    @task(timeout=TimeoutPolicy(idle_timeout=0.05))
    async def slow_task(x: int) -> int:
        await asyncio.sleep(0.2)
        return x + 1

    @entrypoint()
    async def workflow(x: int) -> int:
        return await slow_task(x)

    with pytest.raises(NodeTimeoutError):
        await workflow.ainvoke(1)


@NEEDS_CONTEXTVARS
@pytest.mark.anyio
async def test_task_decorator_preserves_user_idle_timeout_kwarg():
    @task(timeout=TimeoutPolicy(idle_timeout=1.0))
    async def echo_idle_timeout(*, idle_timeout: int) -> int:
        await asyncio.sleep(0)
        return idle_timeout

    @entrypoint()
    async def workflow(x: int) -> int:
        return await echo_idle_timeout(idle_timeout=x)

    assert await workflow.ainvoke(5) == 5


@NEEDS_CONTEXTVARS
@pytest.mark.anyio
async def test_task_decorator_preserves_user_timeout_kwarg():
    @task(timeout=1.0)
    async def echo_timeout(*, timeout: int) -> int:
        await asyncio.sleep(0)
        return timeout

    @entrypoint()
    async def workflow(x: int) -> int:
        return await echo_timeout(timeout=x)

    assert await workflow.ainvoke(5) == 5


@pytest.mark.anyio
async def test_entrypoint_timeout_e2e():
    @entrypoint(timeout=TimeoutPolicy(idle_timeout=0.05))
    async def slow_workflow(x: int) -> int:
        await asyncio.sleep(0.2)
        return x

    with pytest.raises(NodeTimeoutError):
        await slow_workflow.ainvoke(1)


class _MessageStreamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class _SlowStreamingChatModel(GenericFakeChatModel):
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for i in range(3):
            await asyncio.sleep(0.08)
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=str(i),
                    chunk_position="last" if i == 2 else None,
                )
            )
            if run_manager:
                await run_manager.on_llm_new_token(str(i), chunk=chunk)
            yield chunk

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])


@pytest.mark.anyio
async def test_idle_timeout_resets_on_message_stream_callbacks():
    model = _SlowStreamingChatModel(messages=iter([]))

    async def call_model(state: _MessageStreamState) -> _MessageStreamState:
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(_MessageStreamState)
    builder.add_node(
        "call_model",
        call_model,
        timeout=TimeoutPolicy(idle_timeout=0.15),
    )
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    graph = builder.compile()

    chunks: list[str] = []
    async for chunk, _metadata in graph.astream(
        {"messages": [HumanMessage(content="hi")]},
        stream_mode="messages",
    ):
        chunks.append(chunk.content)
    assert chunks == ["0", "1", "2"]


@pytest.mark.anyio
async def test_node_builder_timeout_e2e():
    async def slow(value: int) -> int:
        await asyncio.sleep(0.2)
        return value + 1

    graph = Pregel(
        nodes={
            "slow": (
                NodeBuilder()
                .subscribe_only("input")
                .do(slow)
                .set_timeout(TimeoutPolicy(idle_timeout=0.05))
                .write_to("output")
            )
        },
        channels={
            "input": EphemeralValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )
    with pytest.raises(NodeTimeoutError):
        await graph.ainvoke(1)


@pytest.mark.anyio
async def test_arun_with_retry_timeout_observer_tracks_attempts():
    events: list = []

    class FlakyProc:
        async def ainvoke(self, input, config):
            runtime = config[CONF][CONFIG_KEY_RUNTIME]
            if runtime.execution_info.node_attempt == 1:
                await asyncio.sleep(0.2)
            return "ok"

    policy = RetryPolicy(
        max_attempts=2,
        initial_interval=0.0,
        jitter=False,
        retry_on=NodeTimeoutError,
    )
    task = _make_task(
        FlakyProc(),
        timeout=_idle_timeout(0.05),
        retry_policy=(policy,),
        name="flaky",
    )
    task.config[CONF][CONFIG_KEY_TIMED_ATTEMPT_OBSERVER] = events.append
    assert await arun_with_retry(task, retry_policy=None) == "ok"

    starts = [event for event in events if event.event == "start"]
    finishes = [event for event in events if event.event == "finish"]
    assert [event.context.attempt for event in starts] == [1, 2]
    assert [event.context.attempt for event in finishes] == [1, 2]
    assert [event.status for event in finishes] == ["error", "success"]
    assert starts[0].context.idle_timeout_secs == 0.05
    assert starts[0].context.task_name == "flaky"
    assert isinstance(starts[0].context.started_at, datetime)
    assert isinstance(finishes[0].finished_at, datetime)


@pytest.mark.anyio
async def test_arun_with_retry_timeout_observer_emits_progress_on_heartbeat():
    events: list = []

    class HeartbeatProc:
        async def ainvoke(self, input, config):
            runtime = config[CONF][CONFIG_KEY_RUNTIME]
            for _ in range(8):
                await asyncio.sleep(0.05)
                runtime.heartbeat()
            return "ok"

    task = _make_task(HeartbeatProc(), timeout=_idle_timeout(0.2), name="heartbeat")
    task.config[CONF][CONFIG_KEY_TIMED_ATTEMPT_OBSERVER] = events.append
    assert await arun_with_retry(task, retry_policy=None) == "ok"

    by_event = [ev.event for ev in events]
    assert by_event[0] == "start"
    assert by_event[-1] == "finish"
    progress = [ev for ev in events if ev.event == "progress"]
    assert progress, "expected at least one progress event from heartbeat"
    # Rate limit is `idle_timeout / 4` = 0.05s; with 8 heartbeats spaced ~0.05s
    # we should see at most ~one progress event per heartbeat (well below 8).
    assert len(progress) <= len(by_event)
    for ev in progress:
        assert ev.context.task_name == "heartbeat"
        assert ev.context.attempt == 1
        assert ev.context.idle_timeout_secs == 0.2
        assert isinstance(ev.progress_at, datetime)


@pytest.mark.anyio
async def test_arun_with_retry_timeout_observer_treats_parent_command_as_non_error():
    events: list = []

    class ParentProc:
        async def ainvoke(self, input, config):
            raise ParentCommand(Command(graph=Command.PARENT))

    task = _make_task(ParentProc(), timeout=_idle_timeout(0.05), name="parent")
    task.config[CONF][CONFIG_KEY_TIMED_ATTEMPT_OBSERVER] = events.append
    with pytest.raises(ParentCommand):
        await arun_with_retry(task, retry_policy=None)

    finish = next(event for event in events if event.event == "finish")
    assert finish.status == "success"
    assert finish.error_type is None
    assert finish.error_message is None


@pytest.mark.anyio
async def test_arun_with_retry_timeout_observer_finishes_when_parent_writer_errors():
    events: list = []

    class ParentProc:
        async def ainvoke(self, input, config):
            raise ParentCommand(Command(graph="parent", update={"value": "updated"}))

    class FailingWriter:
        def invoke(self, input, config):
            raise ValueError("writer failed")

    task = _make_task(
        ParentProc(),
        timeout=_idle_timeout(0.05),
        name="parent",
        writers=(FailingWriter(),),
    )
    task.config[CONF][CONFIG_KEY_TIMED_ATTEMPT_OBSERVER] = events.append
    with pytest.raises(ValueError, match="writer failed"):
        await arun_with_retry(task, retry_policy=None)

    finish = next(event for event in events if event.event == "finish")
    assert finish.status == "error"
    assert finish.error_type == "ValueError"
    assert finish.error_message == "writer failed"


@pytest.mark.anyio
async def test_arun_with_retry_timeout_observer_treats_bubble_up_as_non_error():
    events: list = []

    class BubbleProc:
        async def ainvoke(self, input, config):
            raise GraphInterrupt(())

    task = _make_task(BubbleProc(), timeout=_idle_timeout(0.05), name="bubble")
    task.config[CONF][CONFIG_KEY_TIMED_ATTEMPT_OBSERVER] = events.append
    with pytest.raises(GraphInterrupt):
        await arun_with_retry(task, retry_policy=None)

    finish = next(event for event in events if event.event == "finish")
    assert finish.status == "success"
    assert finish.error_type is None
    assert finish.error_message is None


def test_graph_error_handler_runs_after_retry_exhaustion():
    class State(TypedDict):
        foo: str

    attempts = 0
    captured: dict[str, object] = {}

    def always_failing_node(state: State) -> State:
        nonlocal attempts
        attempts += 1
        raise ValueError("Always fails")

    def err_handler_node(state: State, runtime: Runtime) -> State:
        captured["from_node_names"] = runtime.execution_info.from_node_names
        captured["from_node_errors"] = runtime.execution_info.from_node_errors
        return {"foo": "handled"}

    def after_handler(state: State) -> State:
        return {"foo": f"{state['foo']}_after"}

    retry_policy = RetryPolicy(
        max_attempts=2,
        initial_interval=0.01,
        jitter=False,
        retry_on=ValueError,
    )

    graph = (
        StateGraph(State)
        .add_node("always_failing", always_failing_node, retry_policy=retry_policy)
        .set_graph_error_handler("err_handler", err_handler_node)
        .add_node("after_handler", after_handler)
        .add_edge(START, "always_failing")
        .add_edge("err_handler", "after_handler")
        .compile()
    )

    with patch("time.sleep"):
        result = graph.invoke({"foo": ""})

    assert attempts == 2
    assert result["foo"] == "handled_after"
    assert captured["from_node_names"] == ("always_failing",)
    assert isinstance(captured["from_node_errors"], tuple)
    assert len(captured["from_node_errors"]) == 1
    assert isinstance(captured["from_node_errors"][0], BaseException)


def test_graph_error_handler_can_route_with_command():
    class State(TypedDict):
        foo: str

    attempts = 0

    def always_failing_node(state: State) -> State:
        nonlocal attempts
        attempts += 1
        raise ValueError("Always fails")

    def err_handler_node(state: State) -> Command:
        return Command(update={"foo": "handled"}, goto="next_node")

    def next_node(state: State) -> State:
        return {"foo": f"{state['foo']}_next"}

    retry_policy = RetryPolicy(
        max_attempts=1,
        initial_interval=0.01,
        jitter=False,
        retry_on=ValueError,
    )

    graph = (
        StateGraph(State)
        .add_node("always_failing", always_failing_node, retry_policy=retry_policy)
        .set_graph_error_handler(
            "err_handler",
            err_handler_node,
            destinations=("next_node",),
        )
        .add_node("next_node", next_node)
        .add_edge(START, "always_failing")
        .compile()
    )

    result = graph.invoke({"foo": ""})
    assert attempts == 1
    assert result["foo"] == "handled_next"


def test_graph_error_handler_failure_fails_run():
    class State(TypedDict):
        foo: str

    def always_failing_node(state: State) -> State:
        raise ValueError("Always fails")

    def err_handler_node(state: State) -> State:
        raise RuntimeError("handler failed")

    graph = (
        StateGraph(State)
        .add_node("always_failing", always_failing_node)
        .set_graph_error_handler("err_handler", err_handler_node)
        .add_edge(START, "always_failing")
        .compile()
    )

    with pytest.raises(RuntimeError, match="handler failed"):
        graph.invoke({"foo": ""})


def test_graph_error_handler_handles_subgraph_internal_failure():
    class SubState(TypedDict):
        foo: str

    class ParentState(TypedDict):
        foo: str

    parent_handler_called = False
    captured: dict[str, object] = {}

    def sub_fail_node(state: SubState) -> SubState:
        raise ValueError("subgraph boom")

    def parent_handler(state: ParentState, runtime: Runtime) -> ParentState:
        nonlocal parent_handler_called
        parent_handler_called = True
        captured["from_node_names"] = runtime.execution_info.from_node_names
        captured["from_node_errors"] = runtime.execution_info.from_node_errors
        return {"foo": "handled_by_parent"}

    subgraph = (
        StateGraph(SubState)
        .add_node("sub_fail_node", sub_fail_node)
        .add_edge(START, "sub_fail_node")
        .compile()
    )

    parent_graph = (
        StateGraph(ParentState)
        .add_node("subgraph_node", subgraph)
        .set_graph_error_handler("parent_handler", parent_handler)
        .add_edge(START, "subgraph_node")
        .compile()
    )

    result = parent_graph.invoke({"foo": ""})
    assert result["foo"] == "handled_by_parent"
    assert parent_handler_called is True
    assert captured["from_node_names"] == ("subgraph_node",)
    assert isinstance(captured["from_node_errors"], tuple)
    assert len(captured["from_node_errors"]) == 1
    assert isinstance(captured["from_node_errors"][0], BaseException)


def test_graph_error_handler_error_context_survives_checkpoint_resume():
    class State(TypedDict):
        foo: str

    captured: dict[str, object] = {}

    def always_failing_node(state: State) -> State:
        raise RuntimeError("failed before handler")

    def err_handler_node(state: State, runtime: Runtime) -> State:
        captured["from_node_names"] = runtime.execution_info.from_node_names
        captured["from_node_errors"] = runtime.execution_info.from_node_errors
        return {"foo": "handled_after_resume"}

    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "graph-error-resume"}}
    graph = (
        StateGraph(State)
        .add_node("always_failing", always_failing_node)
        .set_graph_error_handler("err_handler", err_handler_node)
        .add_edge(START, "always_failing")
        .compile(checkpointer=checkpointer, interrupt_before=["err_handler"])
    )

    # First run pauses before handler, after failure context is checkpointed.
    graph.invoke({"foo": ""}, config)
    # Resume should execute handler and recover serialized error context.
    result = graph.invoke(None, config)

    assert result["foo"] == "handled_after_resume"
    assert captured["from_node_names"] == ("always_failing",)
    assert isinstance(captured["from_node_errors"], tuple)
    assert len(captured["from_node_errors"]) == 1
    assert isinstance(captured["from_node_errors"][0], BaseException)


def test_graph_error_info_supports_multiple_failures_from_pending_writes():
    pending_writes = [
        ("task_a", GRAPH_ERROR_INFO, "fail_a"),
        ("task_b", GRAPH_ERROR_INFO, "fail_b"),
        ("task_a", ERROR, ValueError("a failed")),
        ("task_b", ERROR, KeyError("b failed")),
    ]

    failed_node_names = _read_failed_node_names_from_pending_writes(pending_writes)
    assert failed_node_names == [("task_a", "fail_a"), ("task_b", "fail_b")]
    errors = _read_errors_for_task_ids_from_pending_writes(
        pending_writes, tuple(task_id for task_id, _ in failed_node_names)
    )
    assert len(errors) == 2
    assert isinstance(errors[0], ValueError)
    assert isinstance(errors[1], KeyError)
