from _typeshed import Incomplete
from langgraph.managed.base import ConfiguredManagedValue, WritableManagedValue
from langgraph.types import LoopProtocol as LoopProtocol
from typing import Any, AsyncIterator, Iterator, Sequence
from typing_extensions import Self

V = dict[str, Any]
Value = dict[str, V]
Update = dict[str, V | None]

class SharedValue(WritableManagedValue[Value, Update]):
    @staticmethod
    def on(scope: str) -> ConfiguredManagedValue: ...
    @classmethod
    def enter(cls, loop: LoopProtocol, **kwargs: Any) -> Iterator[Self]: ...
    @classmethod
    async def aenter(cls, loop: LoopProtocol, **kwargs: Any) -> AsyncIterator[Self]: ...
    scope: Incomplete
    value: Incomplete
    ns: Incomplete
    def __init__(self, loop: LoopProtocol, *, typ: type[Any], scope: str, key: str) -> None: ...
    def __call__(self) -> Value: ...
    def update(self, values: Sequence[Update]) -> None: ...
    async def aupdate(self, writes: Sequence[Update]) -> None: ...
