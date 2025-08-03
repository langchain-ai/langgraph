from typing import Annotated

from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.managed.base import ManagedValue

__all__ = ("IsLastStep", "RemainingStepsManager")


class IsLastStepManager(ManagedValue[bool]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        return scratchpad.step == scratchpad.stop - 1


IsLastStep = Annotated[bool, IsLastStepManager]


class RemainingStepsManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.stop - scratchpad.step


RemainingSteps = Annotated[int, RemainingStepsManager]
