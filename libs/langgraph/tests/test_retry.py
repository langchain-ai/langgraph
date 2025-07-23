from unittest.mock import Mock, patch

import pytest
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.pregel._retry import _should_retry_on
from langgraph.types import RetryPolicy


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

    def failing_node(state: State):
        nonlocal attempt_count
        attempt_count += 1
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
    assert result["foo"] == "other_node"  # Final result should be from other_node

    # Verify the sleep intervals
    call_args_list = [args[0][0] for args in mock_sleep.call_args_list]
    assert call_args_list == [0.01, 0.02]


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
