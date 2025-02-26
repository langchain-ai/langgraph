from _typeshed import Incomplete
from langgraph.managed.base import ConfiguredManagedValue, ManagedValue, V
from langgraph.types import LoopProtocol as LoopProtocol
from typing import Any, AsyncContextManager, AsyncIterator, Callable, ContextManager, Generic, Iterator
from typing_extensions import Self

class Context(ManagedValue[V], Generic[V]):
    runtime: bool
    value: V
    @staticmethod
    def of(ctx: None | Callable[..., ContextManager[V]] | type[ContextManager[V]] | Callable[..., AsyncContextManager[V]] | type[AsyncContextManager[V]] = None, actx: Callable[..., AsyncContextManager[V]] | type[AsyncContextManager[V]] | None = None) -> ConfiguredManagedValue: ...
    @classmethod
    def enter(cls, loop: LoopProtocol, **kwargs: Any) -> Iterator[Self]: ...
    @classmethod
    async def aenter(cls, loop: LoopProtocol, **kwargs: Any) -> AsyncIterator[Self]: ...
    ctx: Incomplete
    actx: Incomplete
    def __init__(self, loop: LoopProtocol, *, ctx: None | type[ContextManager[V]] | type[AsyncContextManager[V]] = None, actx: type[AsyncContextManager[V]] | None = None) -> None: ...
    def __call__(self) -> V: ...
