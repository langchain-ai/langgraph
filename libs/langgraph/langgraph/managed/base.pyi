import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from langgraph.types import LoopProtocol as LoopProtocol
from typing import Any, AsyncIterator, Generic, Iterator, NamedTuple, Sequence, TypeVar
from typing_extensions import Self, TypeGuard

V = TypeVar('V')
U = TypeVar('U')

class ManagedValue(ABC, Generic[V], metaclass=abc.ABCMeta):
    loop: Incomplete
    def __init__(self, loop: LoopProtocol) -> None: ...
    @classmethod
    def enter(cls, loop: LoopProtocol, **kwargs: Any) -> Iterator[Self]: ...
    @classmethod
    async def aenter(cls, loop: LoopProtocol, **kwargs: Any) -> AsyncIterator[Self]: ...
    @abstractmethod
    def __call__(self) -> V: ...

class WritableManagedValue(ManagedValue[V], ABC, Generic[V, U], metaclass=abc.ABCMeta):
    @abstractmethod
    def update(self, writes: Sequence[U]) -> None: ...
    @abstractmethod
    async def aupdate(self, writes: Sequence[U]) -> None: ...

class ConfiguredManagedValue(NamedTuple):
    cls: type[ManagedValue]
    kwargs: dict[str, Any]
ManagedValueSpec = type[ManagedValue] | ConfiguredManagedValue

def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]: ...
def is_readonly_managed_value(value: Any) -> TypeGuard[type[ManagedValue]]: ...
def is_writable_managed_value(value: Any) -> TypeGuard[type[WritableManagedValue]]: ...

ChannelKeyPlaceholder: Incomplete
ChannelTypePlaceholder: Incomplete
ManagedValueMapping = dict[str, ManagedValue]
