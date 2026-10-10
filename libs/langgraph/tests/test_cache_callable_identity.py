"""Regression matrix for cache callable-identity collisions.

These tests cover langchain-ai/langgraph#8753 without selecting a production
policy. Known collision rows are strict xfails so the suite stays green before
a source fix lands; once a fix makes a row pass, XPASS is intentionally fatal
until the marker is removed and the row becomes an ordinary regression.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import pytest
from langgraph.cache.memory import InMemoryCache
from typing_extensions import TypedDict

from langgraph._internal._runnable import RunnableCallable
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy


class State(TypedDict):
    out: str


NodePairFactory = Callable[[], tuple[Any, Any]]

KNOWN_CACHE_IDENTITY_COLLISION = pytest.mark.xfail(
    strict=True,
    reason="Known cache callable-identity collision tracked by #8753",
)


def _named_a(_: State) -> State:
    return {"out": "A"}


def _named_b(_: State) -> State:
    return {"out": "B"}


async def _async_named_a(_: State) -> State:
    return {"out": "A"}


async def _async_named_b(_: State) -> State:
    return {"out": "B"}


def _tagged(_: State, *, tag: str) -> State:
    return {"out": tag}


async def _atagged(_: State, *, tag: str) -> State:
    return {"out": tag}


class _CallableNode:
    def __init__(self, value: str) -> None:
        self.value = value

    def __call__(self, _: State) -> State:
        return {"out": self.value}


class _BoundNode:
    def __init__(self, value: str) -> None:
        self.value = value

    def run(self, _: State) -> State:
        return {"out": self.value}


def _make_closure(value: str) -> Callable[[State], State]:
    def node(_: State) -> State:
        return {"out": value}

    return node


def _callable_instance_pair() -> tuple[Any, Any]:
    return _CallableNode("A"), _CallableNode("B")


def _lambda_pair() -> tuple[Any, Any]:
    return (
        lambda _: {"out": "A"},
        lambda _: {"out": "B"},
    )


def _partial_pair() -> tuple[Any, Any]:
    return (
        functools.partial(_tagged, tag="A"),
        functools.partial(_tagged, tag="B"),
    )


def _closure_pair() -> tuple[Any, Any]:
    return _make_closure("A"), _make_closure("B")


def _bound_method_pair() -> tuple[Any, Any]:
    return _BoundNode("A").run, _BoundNode("B").run


def _wrapper_kwargs_pair() -> tuple[Any, Any]:
    return (
        RunnableCallable(_tagged, _atagged, tag="A"),
        RunnableCallable(_tagged, _atagged, tag="B"),
    )


CALLABLE_IDENTITY_CASES: tuple[tuple[str, NodePairFactory], ...] = (
    ("callable-instance", _callable_instance_pair),
    ("lambda", _lambda_pair),
    ("partial", _partial_pair),
    ("closure", _closure_pair),
    ("bound-method", _bound_method_pair),
    ("wrapper-kwargs", _wrapper_kwargs_pair),
)


def _compile(node: Any, cache: InMemoryCache) -> Any:
    builder = StateGraph(State)
    builder.add_node("process", node, cache_policy=CachePolicy())
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    return builder.compile(cache=cache)


def _assert_sync_isolated(node_a: Any, node_b: Any) -> None:
    cache = InMemoryCache()
    graph_a = _compile(node_a, cache)
    graph_b = _compile(node_b, cache)

    assert graph_a.invoke({"out": "input"}) == {"out": "A"}
    assert graph_b.invoke({"out": "input"}) == {"out": "B"}


async def _assert_async_isolated(node_a: Any, node_b: Any) -> None:
    cache = InMemoryCache()
    graph_a = _compile(node_a, cache)
    graph_b = _compile(node_b, cache)

    assert await graph_a.ainvoke({"out": "input"}) == {"out": "A"}
    assert await graph_b.ainvoke({"out": "input"}) == {"out": "B"}


def test_cache_keeps_distinct_top_level_named_functions_separate() -> None:
    """Control: sharing a cache alone must not imply a collision."""
    _assert_sync_isolated(_named_a, _named_b)


@pytest.mark.anyio
async def test_async_cache_keeps_distinct_top_level_named_functions_separate() -> None:
    """Async-path control for two distinguishable sync functions."""
    await _assert_async_isolated(_named_a, _named_b)


@pytest.mark.parametrize(
    "make_pair",
    [pytest.param(factory, id=name) for name, factory in CALLABLE_IDENTITY_CASES],
)
@KNOWN_CACHE_IDENTITY_COLLISION
def test_cache_does_not_cross_hit_stateful_callable_bindings(
    make_pair: NodePairFactory,
) -> None:
    """Different callable bindings must not share one cached computation."""
    _assert_sync_isolated(*make_pair())


@pytest.mark.anyio
@pytest.mark.parametrize(
    "make_pair",
    [pytest.param(factory, id=name) for name, factory in CALLABLE_IDENTITY_CASES],
)
@KNOWN_CACHE_IDENTITY_COLLISION
async def test_async_cache_does_not_cross_hit_stateful_callable_bindings(
    make_pair: NodePairFactory,
) -> None:
    """The same binding boundary must hold through `ainvoke`."""
    await _assert_async_isolated(*make_pair())


@pytest.mark.anyio
@KNOWN_CACHE_IDENTITY_COLLISION
async def test_cache_keeps_distinct_named_async_functions_separate() -> None:
    """Named async nodes must not collapse because only `.func` is unwrapped."""
    await _assert_async_isolated(_async_named_a, _async_named_b)
