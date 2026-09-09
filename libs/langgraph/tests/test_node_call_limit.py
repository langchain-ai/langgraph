import pytest

from langgraph.errors import GraphRecursionError
from langgraph.runtime import NodeCallLimitError, Runtime


def test_counts_are_recorded_per_node():
    rt = Runtime()
    assert rt.record_node_call("a") == 1
    assert rt.record_node_call("a") == 2
    assert rt.record_node_call("b") == 1
    assert rt.node_call_counts == {"a": 2, "b": 1}


def test_limit_is_enforced():
    rt = Runtime(node_call_limits={"a": 2})
    rt.record_node_call("a")
    rt.record_node_call("a")
    with pytest.raises(NodeCallLimitError):
        rt.record_node_call("a")


def test_unlimited_node_never_raises():
    rt = Runtime()
    for _ in range(25):
        rt.record_node_call("a")
    assert rt.node_call_counts["a"] == 25


def test_limits_only_apply_to_configured_nodes():
    rt = Runtime(node_call_limits={"a": 1})
    rt.record_node_call("a")
    # "b" has no limit configured
    rt.record_node_call("b")
    rt.record_node_call("b")
    assert rt.node_call_counts["b"] == 2


def test_error_is_catchable_as_recursion_error():
    rt = Runtime(node_call_limits={"a": 0})
    with pytest.raises(GraphRecursionError):
        rt.record_node_call("a")
