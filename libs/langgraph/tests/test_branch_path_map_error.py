"""Regression tests for a clearer error when a conditional router returns a
value that isn't a `path_map` key.

Before this change, `_branch.py` did a bare `self.ends[route_key]`, so a routing
function that returned anything other than a `path_map` key surfaced as a bare
``KeyError: 'whatever'`` -- with no indication of which node's router misbehaved
or which values were actually valid.

Run against the checked-out repo:

    pytest tests/test_branch_path_map_error.py -v
"""

from typing import Annotated, Any, TypedDict

import operator
import pytest

from langgraph.graph import START, StateGraph
from langgraph.types import Send


class State(TypedDict):
    items: Annotated[list, operator.add]


def _graph(path_ret, path_map, nodes=("b", "c")):
    """Build a one-hop graph whose router returns `path_ret`.

    The default-arg binding on the node lambdas avoids Python's late binding of
    loop variables, which would otherwise make every node report the same name.
    """
    builder = StateGraph(State)
    builder.add_node("a", lambda s: {"items": ["a_out"]})
    for name in nodes:
        builder.add_node(name, lambda s, _n=name: {"items": [_n]})
    if path_map is None:
        builder.add_conditional_edges("a", path_ret)
    else:
        builder.add_conditional_edges("a", path_ret, path_map)
    builder.add_edge(START, "a")
    return builder.compile()


def test_bad_router_value_reports_valid_keys() -> None:
    """The error names the offending value and the legal ones."""
    graph = _graph(lambda s: "count_retry", {"b": "b", "c": "c"})

    with pytest.raises(ValueError) as exc:
        graph.invoke({"items": []})

    message = str(exc.value)
    assert "count_retry" in message
    assert "'b'" in message and "'c'" in message
    # Should tell the user how to fix it, not just what went wrong.
    assert "path_map" in message


def test_bad_router_value_is_not_a_bare_keyerror() -> None:
    """Guard the regression: the old behaviour was an unhelpful bare KeyError."""
    graph = _graph(lambda s: "count_retry", {"b": "b", "c": "c"})

    with pytest.raises(ValueError):
        graph.invoke({"items": []})

    try:
        graph.invoke({"items": []})
    except ValueError as exc:
        assert not isinstance(exc, KeyError)


def test_single_destination_still_routes() -> None:
    assert _graph(lambda s: "b", {"b": "b", "c": "c"}).invoke({"items": []})[
        "items"
    ] == ["a_out", "b"]


def test_multiple_destinations_still_route() -> None:
    result = _graph(lambda s: ["b", "c"], {"b": "b", "c": "c"}).invoke({"items": []})
    assert sorted(result["items"]) == ["a_out", "b", "c"]


def test_list_path_map_still_routes() -> None:
    graph = _graph(lambda s: "b", ["b"], nodes=("b",))
    assert graph.invoke({"items": []})["items"] == ["a_out", "b"]


def test_missing_path_map_still_routes() -> None:
    graph = _graph(lambda s: "b", None, nodes=("b",))
    assert graph.invoke({"items": []})["items"] == ["a_out", "b"]


def test_send_objects_pass_through() -> None:
    graph = _graph(lambda s: [Send("b", s)], {"b": "a"}, nodes=("b",))
    assert "b" in graph.invoke({"items": []})["items"]


def test_non_string_path_map_keys_still_route() -> None:
    graph = _graph(lambda s: 1, {1: "b"}, nodes=("b",))
    assert graph.invoke({"items": []})["items"] == ["a_out", "b"]
