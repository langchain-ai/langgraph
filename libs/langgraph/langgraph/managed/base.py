from abc import ABC, abstractmethod
from inspect import isclass
from typing import (
    Any,
    Generic,
    TypeVar,
)

from typing_extensions import TypeGuard

from langgraph.types import PregelScratchpad

V = TypeVar("V")
U = TypeVar("U")


class ManagedValue(ABC, Generic[V]):
    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V: ...


ManagedValueSpec = type[ManagedValue]


def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]:
    return isclass(value) and issubclass(value, ManagedValue)


ManagedValueMapping = dict[str, ManagedValueSpec]
