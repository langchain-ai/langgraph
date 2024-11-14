import asyncio
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
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import Annotated, NotRequired, Required

from langgraph.constants import CONFIG_KEY_STREAM_WRITER
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.store.base import BaseStore
from langgraph.utils.fields import _is_optional_type, get_field_default
from langgraph.utils.runnable import (
    RunnableCallable,
    RunnableSeq,
    coerce_to_runnable,
    is_async_callable,
    is_async_generator,
)

pytestmark = pytest.mark.anyio


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
    assert _is_optional_type(None)
    assert not _is_optional_type(type(None))
    assert _is_optional_type(Optional[list])
    assert not _is_optional_type(int)
    assert _is_optional_type(Optional[Literal[1, 2, 3]])
    assert not _is_optional_type(Literal[1, 2, 3])
    assert _is_optional_type(Optional[List[int]])
    assert _is_optional_type(Optional[Dict[str, int]])
    assert not _is_optional_type(List[Optional[int]])
    assert _is_optional_type(Union[Optional[str], Optional[int]])
    assert _is_optional_type(
        Union[
            Union[Optional[str], Optional[int]], Union[Optional[float], Optional[dict]]
        ]
    )
    assert not _is_optional_type(Union[Union[str, int], Union[float, dict]])

    assert _is_optional_type(Union[int, None])
    assert _is_optional_type(Union[str, None, int])
    assert _is_optional_type(Union[None, str, int])
    assert not _is_optional_type(Union[int, str])

    assert not _is_optional_type(Any)  # Do we actually want this?
    assert _is_optional_type(Optional[Any])

    class MyClass:
        pass

    assert _is_optional_type(Optional[MyClass])
    assert not _is_optional_type(MyClass)
    assert _is_optional_type(Optional[ForwardRef("MyClass")])
    assert not _is_optional_type(ForwardRef("MyClass"))

    assert _is_optional_type(Optional[Union[List[int], Dict[str, Optional[int]]]])
    assert not _is_optional_type(Union[List[int], Dict[str, Optional[int]]])

    assert _is_optional_type(Optional[Callable[[int], str]])
    assert not _is_optional_type(Callable[[int], Optional[str]])

    T = TypeVar("T")
    assert _is_optional_type(Optional[T])
    assert not _is_optional_type(T)

    U = TypeVar("U", bound=Optional[T])  # type: ignore
    assert _is_optional_type(U)


def test_is_required():
    class MyBaseTypedDict(TypedDict):
        val_1: Required[Optional[str]]
        val_2: Required[str]
        val_3: NotRequired[str]
        val_4: NotRequired[Optional[str]]
        val_5: Annotated[NotRequired[int], "foo"]
        val_6: NotRequired[Annotated[int, "foo"]]
        val_7: Annotated[Required[int], "foo"]
        val_8: Required[Annotated[int, "foo"]]
        val_9: Optional[str]
        val_10: str

    annos = MyBaseTypedDict.__annotations__
    assert get_field_default("val_1", annos["val_1"], MyBaseTypedDict) == ...
    assert get_field_default("val_2", annos["val_2"], MyBaseTypedDict) == ...
    assert get_field_default("val_3", annos["val_3"], MyBaseTypedDict) is None
    assert get_field_default("val_4", annos["val_4"], MyBaseTypedDict) is None
    # See https://peps.python.org/pep-0655/#interaction-with-annotated
    assert get_field_default("val_5", annos["val_5"], MyBaseTypedDict) is None
    assert get_field_default("val_6", annos["val_6"], MyBaseTypedDict) is None
    assert get_field_default("val_7", annos["val_7"], MyBaseTypedDict) == ...
    assert get_field_default("val_8", annos["val_8"], MyBaseTypedDict) == ...
    assert get_field_default("val_9", annos["val_9"], MyBaseTypedDict) is None
    assert get_field_default("val_10", annos["val_10"], MyBaseTypedDict) == ...

    class MyChildDict(MyBaseTypedDict):
        val_11: int
        val_11b: Optional[int]
        val_11c: Union[int, None, str]

    class MyGrandChildDict(MyChildDict, total=False):
        val_12: int
        val_13: Required[str]

    cannos = MyChildDict.__annotations__
    gcannos = MyGrandChildDict.__annotations__
    assert get_field_default("val_11", cannos["val_11"], MyChildDict) == ...
    assert get_field_default("val_11b", cannos["val_11b"], MyChildDict) is None
    assert get_field_default("val_11c", cannos["val_11c"], MyChildDict) is None
    assert get_field_default("val_12", gcannos["val_12"], MyGrandChildDict) is None
    assert get_field_default("val_9", gcannos["val_9"], MyGrandChildDict) is None
    assert get_field_default("val_13", gcannos["val_13"], MyGrandChildDict) == ...


async def test_runnable_callable_ainvoke_context() -> None:
    async def afunc(x):
        await asyncio.sleep(0.1)
        return x * 2

    runnable = RunnableCallable(func=None, afunc=afunc)
    config = {"configurable": {}, "conf": {}}

    result = await runnable.ainvoke(5, config)
    assert result == 10

    # Test error handling
    async def error_func(x):
        raise ValueError("Test error")

    error_runnable = RunnableCallable(func=None, afunc=error_func)
    with pytest.raises(ValueError, match="Test error"):
        await error_runnable.ainvoke(5, config)


def test_runnable_callable_missing_config_key() -> None:
    from langgraph.store.base import BaseStore
    from langgraph.utils.runnable import RunnableCallable

    def func(x, store: BaseStore):
        return x

    runnable = RunnableCallable(func=func)
    config = {"configurable": {}, "conf": {}}

    with pytest.raises(ValueError, match="Missing required config key"):
        runnable.invoke("test", config)


def test_runnable_seq_init_validation() -> None:
    from langgraph.utils.runnable import RunnableSeq

    def step1(x):
        return x + 1

    def step2(x):
        return x * 2

    # Test valid initialization
    seq = RunnableSeq(step1, step2, name="test_seq")
    assert len(seq.steps) == 2
    assert seq.name == "test_seq"

    # Test error with single step
    with pytest.raises(ValueError, match="must have at least 2 steps"):
        RunnableSeq(step1)


def test_runnable_seq_edge_cases() -> None:
    from langgraph.utils.runnable import RunnableSeq

    def step1(x):
        return x + 1

    def step2(x):
        return x * 2

    seq = RunnableSeq(step1, step2)

    # Test __or__ with RunnableSeq
    other_seq = RunnableSeq(step1, step2)
    combined = seq | other_seq
    assert len(combined.steps) == 4

    # Test __ror__ with RunnableSeq
    combined = other_seq | seq
    assert len(combined.steps) == 4

    # Test __or__ with regular function
    combined = seq | step1
    assert len(combined.steps) == 3

    # Test name propagation
    named_seq = RunnableSeq(step1, step2, name="test_seq")
    combined = named_seq | step1
    assert combined.name == "test_seq"


def test_runnable_seq_stream() -> None:
    def step1(x):
        return x + 1

    def step2(x):
        return x * 2

    seq = RunnableSeq(step1, step2)

    # Test normal streaming
    config = {"configurable": {}, "conf": {}, "recursion_limit": 10}
    result = list(seq.stream(5, config))
    assert result == [12]

    # Test with custom stream writer
    def stream_writer(x):
        return x

    config["conf"][CONFIG_KEY_STREAM_WRITER] = stream_writer
    result = list(seq.stream(5, config))
    assert result == [12]

    # Test error handling
    def error_step(x):
        raise ValueError("Test error")

    error_seq = RunnableSeq(step1, error_step)
    with pytest.raises(ValueError, match="Test error"):
        list(error_seq.stream(5, config))


def test_runnable_callable_config_errors() -> None:
    def func(x, store: BaseStore, writer=None):
        return x

    runnable = RunnableCallable(func=func)
    config = {"configurable": {}, "conf": {}}

    # Test missing required config key
    with pytest.raises(ValueError, match="Missing required config key"):
        runnable.invoke("test", config)

    # Test with config but missing store
    config["conf"]["writer"] = None
    with pytest.raises(ValueError, match="Missing required config key"):
        runnable.invoke("test", config)


def test_runnable_callable_repr() -> None:
    def test_func(x):
        return x

    runnable = RunnableCallable(func=test_func, tags=["test"])
    repr_str = repr(runnable)
    assert "test_func" in repr_str
    assert "tags=['test']" in repr_str


def test_runnable_callable_missing_functions() -> None:
    with pytest.raises(
        ValueError, match="At least one of func or afunc must be provided"
    ):
        RunnableCallable(func=None, afunc=None)


async def test_runnable_seq_async_error_handling() -> None:
    def step1(x):
        return x + 1

    async def step2(x):
        await asyncio.sleep(0.1)
        raise ValueError("Test async error")

    seq = RunnableSeq(step1, step2)
    config = {"configurable": {}, "conf": {}}

    # Test error handling in ainvoke
    with pytest.raises(ValueError, match="Test async error"):
        await seq.ainvoke(5, config)

    # Test error handling in astream
    with pytest.raises(ValueError, match="Test async error"):
        async for _ in seq.astream(5, config):
            pass


async def test_runnable_callable_config_recursion() -> None:
    # Test recursive runnable that returns another runnable
    class InnerRunnable(Runnable):
        def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
            return input * 2

    def outer_func(x):
        return InnerRunnable()

    runnable = RunnableCallable(func=outer_func, recurse=True)
    result = runnable.invoke(5, {"configurable": {}, "conf": {}})
    assert result == 10

    # Test async version
    async def async_outer_func(x):
        return InnerRunnable()

    async_runnable = RunnableCallable(func=None, afunc=async_outer_func, recurse=True)
    result = await async_runnable.ainvoke(5, {"configurable": {}, "conf": {}})
    assert result == 10

    # Test with recursion disabled
    no_recurse = RunnableCallable(func=outer_func, recurse=False)
    result = no_recurse.invoke(5, {"configurable": {}, "conf": {}})
    assert isinstance(result, InnerRunnable)


async def test_runnable_callable_missing_sync_func() -> None:
    async def afunc(x):
        return x * 2

    runnable = RunnableCallable(func=None, afunc=afunc)

    # Test error when trying to use sync invoke with async-only runnable
    with pytest.raises(TypeError, match="No synchronous function provided"):
        runnable.invoke(5, None)

    # Test successful async invoke
    result = await runnable.ainvoke(5, None)
    assert result == 10


async def test_runnable_seq_ainvoke_error_handling() -> None:
    def step1(x):
        return x + 1

    async def error_step(x):
        raise ValueError("Test error")

    seq = RunnableSeq(step1, error_step)

    config = {"configurable": {}, "conf": {}}

    with pytest.raises(ValueError, match="Test error"):
        await seq.ainvoke(5, config)


async def test_runnable_seq_astream() -> None:
    def step1(x):
        return x + 1

    async def astep2(x):
        return x * 2

    seq = RunnableSeq(step1, astep2)

    config = {"configurable": {}, "conf": {}}
    result = []

    async for chunk in seq.astream(5, config):
        result.append(chunk)

    assert result == [12]


async def test_runnable_callable_no_sync_func() -> None:
    async def afunc(x):
        return x

    runnable = RunnableCallable(func=None, afunc=afunc)

    with pytest.raises(TypeError, match="No synchronous function provided"):
        runnable.invoke(5, None)


def test_coerce_unsupported_type() -> None:
    # Test with unsupported type
    with pytest.raises(TypeError, match="Expected a Runnable, callable or dict"):
        coerce_to_runnable([1, 2, 3], name="test", trace=True)

    with pytest.raises(TypeError, match="Expected a Runnable, callable or dict"):
        coerce_to_runnable(123, name="test", trace=True)


async def test_runnable_callable_ainvoke_missing_config() -> None:
    async def afunc(x, store: BaseStore):
        return x

    runnable = RunnableCallable(func=None, afunc=afunc)
    config = {"configurable": {}, "conf": {}}

    with pytest.raises(ValueError, match="Missing required config key"):
        await runnable.ainvoke("test", config)
