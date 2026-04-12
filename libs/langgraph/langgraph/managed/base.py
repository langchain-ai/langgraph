from abc import ABC, abstractmethod
from inspect import isclass
from typing import (
    Any,
    Generic,
    TypeGuard,
    TypeVar,
)

from langgraph._internal._scratchpad import PregelScratchpad

V = TypeVar("V")
U = TypeVar("U")

__all__ = ("ManagedValueSpec", "ManagedValueMapping")


class ManagedValue(ABC, Generic[V]):
    """Abstract base class for managed graph values.

    Managed values are read-only values injected into graph nodes at runtime.
    They are not part of the graph state and are not checkpointed. Instead,
    they are computed fresh on each step from the current `PregelScratchpad`.

    To create a custom managed value, subclass `ManagedValue` and implement
    the `get` static method:

    ```python
    from typing import Annotated
    from langgraph.managed.base import ManagedValue
    from langgraph._internal._scratchpad import PregelScratchpad

    class CurrentStepManager(ManagedValue[int]):
        @staticmethod
        def get(scratchpad: PregelScratchpad) -> int:
            return scratchpad.step

    CurrentStep = Annotated[int, CurrentStepManager]
    ```
    """

    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V:
        """Compute and return the managed value for the current step.

        Args:
            scratchpad: The current Pregel execution scratchpad, containing
                step counters and other runtime state.

        Returns:
            The computed value for this managed value type.
        """
        ...


ManagedValueSpec = type[ManagedValue]
"""Type alias for a class (not instance) that is a subclass of `ManagedValue`.

Used in type annotations to indicate that a value is a managed value
specification, i.e. a class object that implements `ManagedValue.get`.
"""


def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]:
    """Return `True` if `value` is a `ManagedValue` subclass, `False` otherwise.

    Args:
        value: The object to check.

    Returns:
        `True` if `value` is a class that subclasses `ManagedValue`.
    """
    return isclass(value) and issubclass(value, ManagedValue)


ManagedValueMapping = dict[str, ManagedValueSpec]
"""Type alias for a mapping from state key names to `ManagedValue` subclasses.

Used internally by the graph runtime to look up and inject managed values
into node invocations.
"""
