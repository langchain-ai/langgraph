import functools
import sys
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    List,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)
from unittest.mock import patch

import langsmith
import pytest

from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.utils import is_async_callable, is_async_generator, is_optional_type


def test_is_async() -> None:
    async def func() -> None:
        pass

    assert is_async_callable(func)
    wrapped_func = functools.wraps(func)(func)
    assert is_async_callable(wrapped_func)

    def sync_func() -> None:
        pass

    assert not is_async_callable(sync_func)
    wrapped_sync_func = functools.wraps(sync_func)(sync_func)
    assert not is_async_callable(wrapped_sync_func)

    class AsyncFuncCallable:
        async def __call__(self) -> None:
            pass

    runnable = AsyncFuncCallable()
    assert is_async_callable(runnable)
    wrapped_runnable = functools.wraps(runnable)(runnable)
    assert is_async_callable(wrapped_runnable)

    class SyncFuncCallable:
        def __call__(self) -> None:
            pass

    sync_runnable = SyncFuncCallable()
    assert not is_async_callable(sync_runnable)
    wrapped_sync_runnable = functools.wraps(sync_runnable)(sync_runnable)
    assert not is_async_callable(wrapped_sync_runnable)


def test_is_generator() -> None:
    async def gen():
        yield

    assert is_async_generator(gen)

    wrapped_gen = functools.wraps(gen)(gen)
    assert is_async_generator(wrapped_gen)

    def sync_gen():
        yield

    assert not is_async_generator(sync_gen)
    wrapped_sync_gen = functools.wraps(sync_gen)(sync_gen)
    assert not is_async_generator(wrapped_sync_gen)

    class AsyncGenCallable:
        async def __call__(self):
            yield

    runnable = AsyncGenCallable()
    assert is_async_generator(runnable)
    wrapped_runnable = functools.wraps(runnable)(runnable)
    assert is_async_generator(wrapped_runnable)

    class SyncGenCallable:
        def __call__(self):
            yield

    sync_runnable = SyncGenCallable()
    assert not is_async_generator(sync_runnable)
    wrapped_sync_runnable = functools.wraps(sync_runnable)(sync_runnable)
    assert not is_async_generator(wrapped_sync_runnable)


@pytest.fixture
def rt_graph() -> CompiledGraph:
    class State(TypedDict):
        foo: int
        node_run_id: int

    def node(_: State):
        from langsmith import get_current_run_tree  # type: ignore

        return {"node_run_id": get_current_run_tree().id}  # type: ignore

    graph = StateGraph(State)
    graph.add_node(node)
    graph.set_entry_point("node")
    graph.add_edge("node", END)
    return graph.compile()


def test_runnable_callable_tracing_nested(rt_graph: CompiledGraph) -> None:
    with patch("langsmith.client.Client", spec=langsmith.Client) as mock_client:
        with patch("langchain_core.tracers.langchain.get_client") as mock_get_client:
            mock_get_client.return_value = mock_client
            with langsmith.tracing_context(enabled=True):
                res = rt_graph.invoke({"foo": 1})
    assert isinstance(res["node_run_id"], uuid.UUID)


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
async def test_runnable_callable_tracing_nested_async(rt_graph: CompiledGraph) -> None:
    with patch("langsmith.client.Client", spec=langsmith.Client) as mock_client:
        with patch("langchain_core.tracers.langchain.get_client") as mock_get_client:
            mock_get_client.return_value = mock_client
            with langsmith.tracing_context(enabled=True):
                res = await rt_graph.ainvoke({"foo": 1})
    assert isinstance(res["node_run_id"], uuid.UUID)


def test_is_optional_type():
    assert is_optional_type(None)
    assert not is_optional_type(type(None))
    assert is_optional_type(Optional[list])
    assert not is_optional_type(int)
    assert is_optional_type(Optional[Literal[1, 2, 3]])
    assert not is_optional_type(Literal[1, 2, 3])
    assert is_optional_type(Optional[List[int]])
    assert is_optional_type(Optional[Dict[str, int]])
    assert not is_optional_type(List[Optional[int]])
    assert is_optional_type(Union[Optional[str], Optional[int]])
    assert is_optional_type(
        Union[
            Union[Optional[str], Optional[int]], Union[Optional[float], Optional[dict]]
        ]
    )
    assert not is_optional_type(Union[Union[str, int], Union[float, dict]])

    assert is_optional_type(Union[int, None])
    assert is_optional_type(Union[str, None, int])
    assert is_optional_type(Union[None, str, int])
    assert not is_optional_type(Union[int, str])

    assert not is_optional_type(Any)  # Do we actually want this?
    assert is_optional_type(Optional[Any])

    class MyClass:
        pass

    assert is_optional_type(Optional[MyClass])
    assert not is_optional_type(MyClass)
    assert is_optional_type(Optional[ForwardRef("MyClass")])
    assert not is_optional_type(ForwardRef("MyClass"))

    assert is_optional_type(Optional[Union[List[int], Dict[str, Optional[int]]]])
    assert not is_optional_type(Union[List[int], Dict[str, Optional[int]]])

    assert is_optional_type(Optional[Callable[[int], str]])
    assert not is_optional_type(Callable[[int], Optional[str]])

    T = TypeVar("T")
    assert is_optional_type(Optional[T])
    assert not is_optional_type(T)

    U = TypeVar("U", bound=Optional[T])
    assert is_optional_type(U)
