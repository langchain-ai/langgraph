"""Tests for subgraph auto-detection (`pregel/_utils.py`).

Detection failing is silent — the graph still runs, only introspection goes
quiet — so every shape a node can hold a graph in is pinned here. The expected
values are what the source-parsing implementation this replaced produced for
the same shapes, except for `sourceless`, whose source it could not read,
`empty_closure_cell`, on which it raised, and `unreachable_attribute_chain`,
where it reported a graph that dropped code could never invoke.
"""

import functools
import operator
from typing import Annotated, Any

import pytest
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.pregel._utils import get_function_nonlocals


class State(TypedDict):
    log: Annotated[list, operator.add]


def _leaf(tag: str) -> Any:
    """Return a compiled graph that reports itself as `tag`."""
    builder = StateGraph(State)
    builder.add_node(tag, lambda s: {"log": [tag]})
    builder.add_edge(START, tag)
    builder.add_edge(tag, END)
    compiled = builder.compile()
    compiled.name = tag
    return compiled


def _detect(node: Any) -> str | None:
    """Return the name of the subgraph detected for `node`, or None."""
    builder = StateGraph(State)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    subgraphs = builder.compile().nodes["n"].subgraphs
    return getattr(subgraphs[0], "name", "?") if subgraphs else None


class _Box:
    def __init__(self, payload: Any) -> None:
        self.payload = payload


class _ListSubclass(list):
    pass


class _MethodHolder:
    def __init__(self) -> None:
        self.graph = _leaf("via_self")

    def as_node(self, state: State) -> Any:
        return self.graph.invoke(state)


MODULE_GRAPH = _leaf("module_global")
CHAIN = _Box(_Box(_leaf("attr_chain")))
GRAPH_IN_PLAIN_LIST = [_leaf("in_list")]
METHOD_HOLDER = _MethodHolder()


def closure_capture() -> Any:
    sub = _leaf("closure")

    def node(state: State) -> Any:
        return sub.invoke(state)

    return node


def module_global() -> Any:
    def node(state: State) -> Any:
        return MODULE_GRAPH.invoke(state)

    return node


def attribute_chain() -> Any:
    def node(state: State) -> Any:
        return CHAIN.payload.payload.invoke(state)

    return node


def nested_def_captured_attribute() -> Any:
    """A chain on a captured holder, named only inside a nested code object.

    The captured value is the holder, not the graph, so the chain itself has to
    be recovered from the nested scope.
    """
    holder = _Box(_leaf("nested_captured"))

    def node(state: State) -> Any:
        def inner() -> Any:
            return holder.payload.invoke(state)

        return inner()

    return node


def unreachable_branch() -> Any:
    """A captured graph referenced only from code the compiler removes."""
    sub = _leaf("unreachable")

    def node(state: State) -> Any:
        if False:
            sub.invoke(state)
        return {"log": []}

    return node


def unreachable_attribute_chain() -> Any:
    """A graph named only along an attribute path the compiler dropped.

    The closure keeps `holder`, but the `.payload` load is gone. The source
    parser reported this one; dropped code cannot invoke anything, so that was
    a phantom rather than a detection.
    """
    holder = _Box(_leaf("unreachable_attr"))

    def node(state: State) -> Any:
        if False:
            holder.payload.invoke(state)
        return {"log": []}

    return node


def wrapper_referencing_nothing() -> Any:
    """A wrapper whose own scope holds nothing, so only `__wrapped__` leads on.

    `functools.wraps` would leave the wrapper closing over the inner function;
    setting the attribute by hand does not.
    """
    sub = _leaf("via_wrapped")

    def inner(state: State) -> Any:
        return sub.invoke(state)

    def wrapper(state: State) -> Any:
        return {"log": []}

    wrapper.__wrapped__ = inner
    return wrapper


def captured_list_subclass() -> Any:
    """A `list` subclass is not a leaf: it can carry a graph as an attribute."""
    holder = _ListSubclass()
    holder.payload = _leaf("list_subclass")

    def node(state: State) -> Any:
        return holder.payload.invoke(state)

    return node


def empty_closure_cell() -> Any:
    """An unassigned closure variable leaves a cell that cannot be read."""
    sub = _leaf("beside_empty_cell")

    def node(state: State) -> Any:
        return unassigned, sub.invoke(state)

    return node
    unassigned = 1  # never runs, so the cell it creates is never filled


def sourceless() -> Any:
    """A node compiled without a source file, which `getsource` could not read."""
    namespace: dict[str, Any] = {"SOURCELESS": _leaf("sourceless")}
    exec(
        compile(
            "def node(state):\n    return SOURCELESS.invoke(state)", "<test>", "exec"
        ),
        namespace,
    )
    return namespace["node"]


async def _async_node(state: State) -> Any:
    return await MODULE_GRAPH.ainvoke(state)


def async_node() -> Any:
    return _async_node


def no_subgraph() -> Any:
    """Nothing but leaf values in reach, so the bytecode walk is skipped."""

    def node(state: State) -> Any:
        return {"log": [len("abc") + 1]}

    return node


def recombined_names() -> Any:
    """Loads `CHAIN.payload` and `local.payload`, never `CHAIN.payload.payload`."""

    def node(state: State) -> Any:
        local = _Box("not a graph")
        return {"log": [CHAIN.payload, local.payload]}

    return node


def broken_attribute_chain() -> Any:
    holder = _Box("a string, so `.payload.missing` cannot resolve")

    def node(state: State) -> Any:
        return holder.payload.missing.invoke(state)

    return node


def nested_def_global() -> Any:
    """A global named only in a nested code object: out of reach, as before."""

    def node(state: State) -> Any:
        def inner() -> Any:
            return MODULE_GRAPH.invoke(state)

        return inner()

    return node


def graph_in_plain_list() -> Any:
    def node(state: State) -> Any:
        return GRAPH_IN_PLAIN_LIST[0].invoke(state)

    return node


def bound_method_self() -> Any:
    return METHOD_HOLDER.as_node


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (closure_capture, "closure"),
        (module_global, "module_global"),
        (attribute_chain, "attr_chain"),
        (nested_def_captured_attribute, "nested_captured"),
        (unreachable_branch, "unreachable"),
        (wrapper_referencing_nothing, "via_wrapped"),
        (captured_list_subclass, "list_subclass"),
        (empty_closure_cell, "beside_empty_cell"),
        (sourceless, "sourceless"),
        (async_node, "module_global"),
        # Shapes no reference chain reaches: a subscript, an instance attribute
        # of `self`, a global named only in a nested scope, and an attribute
        # path the compiler dropped.
        (no_subgraph, None),
        (recombined_names, None),
        (broken_attribute_chain, None),
        (nested_def_global, None),
        (graph_in_plain_list, None),
        (bound_method_self, None),
        (unreachable_attribute_chain, None),
    ],
    ids=lambda value: value.__name__ if callable(value) else str(value),
)
def test_subgraph_detection(factory: Any, expected: str | None) -> None:
    assert _detect(factory()) == expected


@pytest.mark.parametrize(
    "candidate",
    [functools.partial(lambda state, extra: {"log": [extra]}, extra="x"), len],
    ids=["partial", "builtin"],
)
def test_callables_without_a_code_object_are_handled(candidate: Any) -> None:
    assert get_function_nonlocals(candidate) == []
