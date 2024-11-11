import asyncio
import concurrent
import concurrent.futures
import types
from functools import partial, update_wrapper
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
    overload,
)

from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START, TAG_HIDDEN
from langgraph.pregel import Pregel
from langgraph.pregel.call import get_runnable_for_func
from langgraph.pregel.read import PregelNode
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy, acall, call

P = ParamSpec("P")
T = TypeVar("T")


@overload
def task(
    *, retry: Optional[RetryPolicy] = None
) -> Callable[
    [Callable[P, Coroutine[None, None, T]]], Callable[P, asyncio.Future[T]]
]: ...


@overload
def task(
    *, retry: Optional[RetryPolicy] = None
) -> Callable[[Callable[P, T]], Callable[P, concurrent.futures.Future[T]]]: ...


def task(
    *, retry: Optional[RetryPolicy] = None
) -> Callable[
    [Callable[P, Union[T, Awaitable[T]]]],
    Callable[P, Union[concurrent.futures.Future[T], asyncio.Future[T]]],
]:
    def _task(func: Callable[P, T]) -> Callable[P, concurrent.futures.Future[T]]:
        if asyncio.iscoroutinefunction(func):
            return update_wrapper(partial(acall, func), func)
        else:
            return update_wrapper(partial(call, func), func)

    return _task


def imp(
    *,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
) -> Callable[[types.FunctionType], Pregel]:
    def _imp(func: types.FunctionType):
        return Pregel(
            nodes={
                func.__name__: PregelNode(
                    bound=get_runnable_for_func(func),
                    triggers=[START],
                    channels=[START],
                    writers=[ChannelWrite([ChannelWriteEntry(END)], tags=[TAG_HIDDEN])],
                )
            },
            channels={START: EphemeralValue(Any, START), END: LastValue(Any, END)},
            input_channels=START,
            output_channels=END,
            stream_mode="updates",
            checkpointer=checkpointer,
            store=store,
        )

    return _imp
