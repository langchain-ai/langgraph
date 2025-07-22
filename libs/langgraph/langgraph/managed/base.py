from abc import ABC, abstractmethod
from inspect import isclass
from typing import (
    Any,
    Generic,
    TypeVar,
)

from typing_extensions import TypeGuard

from langgraph.pregel._scratchpad import PregelScratchpad

V = TypeVar("V")
U = TypeVar("U")

__all__ = ("ManagedValueSpec", "ManagedValueMapping")


class ManagedValue(ABC, Generic[V]):
    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V: ...


ManagedValueSpec = type[ManagedValue]


def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]:
    return isclass(value) and issubclass(value, ManagedValue)


ManagedValueMapping = dict[str, ManagedValueSpec]
