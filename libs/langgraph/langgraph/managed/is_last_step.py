from typing import Annotated

from langgraph.managed.base import ManagedValue


class IsLastStepManager(ManagedValue[bool]):
    def __call__(self) -> bool:
        return self.loop.step == self.loop.stop - 1


IsLastStep = Annotated[bool, IsLastStepManager]


class RemainingStepsManager(ManagedValue[int]):
    def __call__(self) -> int:
        return self.loop.stop - self.loop.step


RemainingSteps = Annotated[int, RemainingStepsManager]
