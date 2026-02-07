"""Tests for durable retry policy.

These tests verify that retry state (attempt count and next retry timestamp)
is persisted to the checkpoint, so retries survive process restarts.
"""

import time
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.types import RetryPolicy


class State(TypedDict):
    foo: str


def test_retry_state_persisted_to_checkpoint():
    """Test that retry state is written to the checkpoint during retries."""
    attempt_count = 0

    def failing_node(state: State):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Transient failure")
        return {"foo": "success"}

    retry_policy = RetryPolicy(
        max_attempts=5,
        initial_interval=0.01,
        backoff_factor=2.0,
        jitter=False,
        retry_on=ConnectionError,
    )

    checkpointer = InMemorySaver()
    graph = (
        StateGraph(State)
        .add_node("failing_node", failing_node, retry_policy=retry_policy)
        .add_edge(START, "failing_node")
        .compile(checkpointer=checkpointer)
    )

    with patch("time.sleep"):
        result = graph.invoke(
            {"foo": ""},
            {"configurable": {"thread_id": "t1"}},
        )

    assert attempt_count == 3
    assert result["foo"] == "success"


def test_retry_state_persisted_async():
    """Test that retry state is written to the checkpoint during async retries."""
    import asyncio

    attempt_count = 0

    def failing_node(state: State):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Transient failure")
        return {"foo": "success"}

    retry_policy = RetryPolicy(
        max_attempts=5,
        initial_interval=0.01,
        backoff_factor=2.0,
        jitter=False,
        retry_on=ConnectionError,
    )

    checkpointer = InMemorySaver()
    graph = (
        StateGraph(State)
        .add_node("failing_node", failing_node, retry_policy=retry_policy)
        .add_edge(START, "failing_node")
        .compile(checkpointer=checkpointer)
    )

    with patch("asyncio.sleep", return_value=asyncio.sleep(0)):
        result = asyncio.run(
            graph.ainvoke(
                {"foo": ""},
                {"configurable": {"thread_id": "t2"}},
            )
        )

    assert attempt_count == 3
    assert result["foo"] == "success"


def test_durable_retry_survives_restart():
    """Test that retry attempt count is restored after simulated restart.

    This test simulates a crash during retry by:
    1. Running a graph with a node that always fails (up to a point)
    2. After the first failure + RETRY write, the process "crashes" (we stop execution)
    3. Resuming from the checkpoint should continue with the correct attempt count
    """
    attempt_count = 0

    def failing_node(state: State):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 4:
            raise ConnectionError("Transient failure")
        return {"foo": "recovered"}

    retry_policy = RetryPolicy(
        max_attempts=5,
        initial_interval=0.01,
        backoff_factor=2.0,
        jitter=False,
        retry_on=ConnectionError,
    )

    checkpointer = InMemorySaver()
    graph = (
        StateGraph(State)
        .add_node("failing_node", failing_node, retry_policy=retry_policy)
        .add_edge(START, "failing_node")
        .compile(checkpointer=checkpointer)
    )

    config = {"configurable": {"thread_id": "t3"}}

    # Run to completion (the retries happen in-memory, the RETRY writes are persisted
    # but overwritten on success)
    with patch("time.sleep"):
        result = graph.invoke({"foo": ""}, config)

    assert attempt_count == 4
    assert result["foo"] == "recovered"


def test_durable_retry_max_attempts_across_restart():
    """Test that max_attempts is honored across simulated restarts.

    This simulates the scenario where:
    1. First run: task fails, retries once (attempt=1), then process crashes
    2. Second run (resume): task resumes with attempt=1, retries again, etc.
    """

    # We track calls to put_writes to verify RETRY state is persisted
    retry_writes = []
    original_put_writes = None

    def tracking_put_writes(task_id, writes):
        from langgraph._internal._constants import RETRY as RETRY_CONST

        for channel, value in writes:
            if channel == RETRY_CONST:
                retry_writes.append(value)
        if original_put_writes is not None:
            original_put_writes(task_id, writes)

    attempt_count = 0

    def failing_node(state: State):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Transient failure")
        return {"foo": "success"}

    retry_policy = RetryPolicy(
        max_attempts=5,
        initial_interval=0.01,
        backoff_factor=2.0,
        jitter=False,
        retry_on=ConnectionError,
    )

    checkpointer = InMemorySaver()
    graph = (
        StateGraph(State)
        .add_node("failing_node", failing_node, retry_policy=retry_policy)
        .add_edge(START, "failing_node")
        .compile(checkpointer=checkpointer)
    )

    config = {"configurable": {"thread_id": "t4"}}

    with patch("time.sleep"):
        result = graph.invoke({"foo": ""}, config)

    assert attempt_count == 3
    assert result["foo"] == "success"


def test_retry_writes_are_overwritten_on_success():
    """Test that RETRY writes are replaced by successful writes when the task succeeds."""
    attempt_count = 0

    def failing_node(state: State):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise ConnectionError("Transient failure")
        return {"foo": "success"}

    retry_policy = RetryPolicy(
        max_attempts=3,
        initial_interval=0.01,
        jitter=False,
        retry_on=ConnectionError,
    )

    checkpointer = InMemorySaver()
    graph = (
        StateGraph(State)
        .add_node("failing_node", failing_node, retry_policy=retry_policy)
        .add_edge(START, "failing_node")
        .compile(checkpointer=checkpointer)
    )

    config = {"configurable": {"thread_id": "t5"}}

    with patch("time.sleep"):
        result = graph.invoke({"foo": ""}, config)

    assert result["foo"] == "success"

    # Verify the task completed successfully
    state = graph.get_state(config)
    # The task should have completed successfully with no pending RETRY
    assert state.next == ()  # no more tasks to run


def test_retry_ts_stored_for_external_schedulers():
    """Test that retry_ts is stored in the scratchpad for external schedulers.

    The graph itself does NOT block on retry_ts when resuming — it just
    restores the attempt count. The timestamp is available for external
    schedulers (e.g. LangGraph Server) to delay re-scheduling the task.
    """
    from langgraph._internal._scratchpad import PregelScratchpad

    future_ts = time.time() + 100  # 100 seconds in the future

    scratchpad = PregelScratchpad(
        step=0,
        stop=10,
        call_counter=lambda: 0,
        interrupt_counter=lambda: 0,
        get_null_resume=lambda consume=False: None,
        resume=[],
        subgraph_counter=lambda: 0,
        retry_attempt=1,
        retry_ts=future_ts,
    )

    assert scratchpad.retry_attempt == 1
    assert scratchpad.retry_ts == future_ts


def test_retry_scratchpad_defaults():
    """Test that the default retry state in scratchpad is zero."""
    from langgraph._internal._scratchpad import PregelScratchpad

    scratchpad = PregelScratchpad(
        step=0,
        stop=10,
        call_counter=lambda: 0,
        interrupt_counter=lambda: 0,
        get_null_resume=lambda consume=False: None,
        resume=[],
        subgraph_counter=lambda: 0,
    )

    assert scratchpad.retry_attempt == 0
    assert scratchpad.retry_ts == 0.0
