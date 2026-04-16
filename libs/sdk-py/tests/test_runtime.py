"""Tests for langgraph_sdk.runtime — verifies field presence required by runtime_to_proto."""

from langgraph_sdk.runtime import _ExecutionRuntime, _ReadRuntime


class _MockStore:
    """Minimal stand-in for BaseStore."""


def test_execution_runtime_has_previous():
    """_ExecutionRuntime must expose a `previous` attribute (used by runtime_to_proto)."""
    store = _MockStore()
    er = _ExecutionRuntime(access_context="threads.create_run", store=store)
    assert hasattr(er, "previous")
    assert er.previous is None


def test_execution_runtime_has_context():
    """_ExecutionRuntime must expose a `context` attribute."""
    store = _MockStore()
    er = _ExecutionRuntime(access_context="threads.create_run", store=store)
    assert hasattr(er, "context")
    assert er.context is None


def test_read_runtime_has_previous():
    """_ReadRuntime must expose a `previous` attribute (used by runtime_to_proto)."""
    store = _MockStore()
    rr = _ReadRuntime(access_context="threads.read", store=store)
    assert hasattr(rr, "previous")
    assert rr.previous is None


def test_read_runtime_has_context():
    """_ReadRuntime must expose a `context` attribute (used by runtime_to_proto)."""
    store = _MockStore()
    rr = _ReadRuntime(access_context="threads.read", store=store)
    assert hasattr(rr, "context")
    assert rr.context is None
