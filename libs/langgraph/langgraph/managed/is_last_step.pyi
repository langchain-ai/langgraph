from langgraph.managed.base import ManagedValue
from typing import Annotated

class IsLastStepManager(ManagedValue[bool]):
    def __call__(self) -> bool: ...
IsLastStep = Annotated[bool, IsLastStepManager]

class RemainingStepsManager(ManagedValue[int]):
    def __call__(self) -> int: ...
RemainingSteps = Annotated[int, RemainingStepsManager]
