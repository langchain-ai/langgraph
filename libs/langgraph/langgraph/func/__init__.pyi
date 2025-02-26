from _typeshed import Incomplete
from dataclasses import dataclass
from langgraph.checkpoint.base import BaseCheckpointSaver as BaseCheckpointSaver
from langgraph.pregel import Pregel
from langgraph.pregel.call import P as P, SyncAsyncFuture as SyncAsyncFuture, T as T
from langgraph.store.base import BaseStore as BaseStore
from langgraph.types import RetryPolicy as RetryPolicy, StreamMode as StreamMode, _DC_KWARGS
from typing import Any, Callable, Generic, TypeVar, overload

@overload
def task(*, name: str | None = None, retry: RetryPolicy | None = None) -> Callable[[Callable[P, T]], Callable[P, SyncAsyncFuture[T]]]: ...
@overload
def task(__func_or_none__: Callable[P, T]) -> Callable[P, SyncAsyncFuture[T]]: ...
R = TypeVar('R')
S = TypeVar('S')

class entrypoint:
    checkpointer: Incomplete
    store: Incomplete
    config_schema: Incomplete
    def __init__(self, checkpointer: BaseCheckpointSaver | None = None, store: BaseStore | None = None, config_schema: type[Any] | None = None) -> None: ...
    @dataclass(**_DC_KWARGS)
    class final(Generic[R, S]):
        value: R
        save: S
        def __init__(self, value, save) -> None: ...
    def __call__(self, func: Callable[..., Any]) -> Pregel: ...
