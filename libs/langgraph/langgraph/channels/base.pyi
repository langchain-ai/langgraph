import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from langgraph.errors import EmptyChannelError as EmptyChannelError, InvalidUpdateError as InvalidUpdateError
from typing import Any, Generic, Sequence, TypeVar
from typing_extensions import Self

__all__ = ['BaseChannel', 'EmptyChannelError', 'InvalidUpdateError']

Value = TypeVar('Value')
Update = TypeVar('Update')
C = TypeVar('C')

class BaseChannel(ABC, Generic[Value, Update, C], metaclass=abc.ABCMeta):
    typ: Incomplete
    key: Incomplete
    def __init__(self, typ: Any, key: str = '') -> None: ...
    @property
    @abstractmethod
    def ValueType(self) -> Any: ...
    @property
    @abstractmethod
    def UpdateType(self) -> Any: ...
    def checkpoint(self) -> C | None: ...
    @abstractmethod
    def from_checkpoint(self, checkpoint: C | None) -> Self: ...
    @abstractmethod
    def update(self, values: Sequence[Update]) -> bool: ...
    @abstractmethod
    def get(self) -> Value: ...
    def consume(self) -> bool: ...
