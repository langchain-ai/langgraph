import functools
import sys
import uuid
from typing import (
    Annotated,
    Any,
    Callable,
    ForwardRef,
    Literal,
    Optional,
    TypeVar,
    Union,
)
from unittest.mock import patch

import langsmith
import pytest
from typing_extensions import NotRequired, Required, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.utils.config import _is_not_empty
from langgraph.utils.fields import (
    _is_optional_type,
    get_enhanced_type_hints,
    get_field_default,
)
from langgraph.utils.runnable import (
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
def rt_graph() -> CompiledStateGraph:
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


def test_runnable_callable_tracing_nested(rt_graph: CompiledStateGraph) -> None:
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
async def test_runnable_callable_tracing_nested_async(
    rt_graph: CompiledStateGraph,
) -> None:
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
    assert _is_optional_type(Optional[list[int]])
    assert _is_optional_type(Optional[dict[str, int]])
    assert not _is_optional_type(list[Optional[int]])
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

    assert _is_optional_type(Optional[Union[list[int], dict[str, Optional[int]]]])
    assert not _is_optional_type(Union[list[int], dict[str, Optional[int]]])

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


def test_enhanced_type_hints() -> None:
    from dataclasses import dataclass
    from typing import Annotated

    from pydantic import BaseModel, Field

    class MyTypedDict(TypedDict):
        val_1: str
        val_2: int = 42
        val_3: str = "default"

    hints = list(get_enhanced_type_hints(MyTypedDict))
    assert len(hints) == 3
    assert hints[0] == ("val_1", str, None, None)
    assert hints[1] == ("val_2", int, 42, None)
    assert hints[2] == ("val_3", str, "default", None)

    @dataclass
    class MyDataclass:
        val_1: str
        val_2: int = 42
        val_3: str = "default"

    hints = list(get_enhanced_type_hints(MyDataclass))
    assert len(hints) == 3
    assert hints[0] == ("val_1", str, None, None)
    assert hints[1] == ("val_2", int, 42, None)
    assert hints[2] == ("val_3", str, "default", None)

    class MyPydanticModel(BaseModel):
        val_1: str
        val_2: int = 42
        val_3: str = Field(default="default", description="A description")

    hints = list(get_enhanced_type_hints(MyPydanticModel))
    assert len(hints) == 3
    assert hints[0] == ("val_1", str, None, None)
    assert hints[1] == ("val_2", int, 42, None)
    assert hints[2] == ("val_3", str, "default", "A description")

    class MyPydanticModelWithAnnotated(BaseModel):
        val_1: Annotated[str, Field(description="A description")]
        val_2: Annotated[int, Field(default=42)]
        val_3: Annotated[
            str, Field(default="default", description="Another description")
        ]

    hints = list(get_enhanced_type_hints(MyPydanticModelWithAnnotated))
    assert len(hints) == 3
    assert hints[0] == ("val_1", str, None, "A description")
    assert hints[1] == ("val_2", int, 42, None)
    assert hints[2] == ("val_3", str, "default", "Another description")


def test_is_not_empty() -> None:
    assert _is_not_empty("foo")
    assert _is_not_empty("")
    assert _is_not_empty(1)
    assert _is_not_empty(0)
    assert not _is_not_empty(None)
    assert not _is_not_empty([])
    assert not _is_not_empty(())
    assert not _is_not_empty({})
