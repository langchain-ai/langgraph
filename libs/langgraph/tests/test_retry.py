import pytest

from langgraph.pregel.retry import _should_retry_on
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
    policy = RetryPolicy(retry_on=[ValueError, KeyError])

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
    policy = RetryPolicy(retry_on=[])

    # Should not retry when sequence is empty
    assert _should_retry_on(policy, ValueError("test error")) is False


def test_should_retry_default_retry_on():
    """Test the default retry_on function."""
    from unittest.mock import Mock

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
