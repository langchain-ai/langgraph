from typing import Annotated

from langgraph.managed.base import ManagedValue


class IsLastStepManager(ManagedValue[bool]):
    def __call__(self, step: int, stop: int) -> bool:
        return step == (stop - 1)


IsLastStep = Annotated[bool, IsLastStepManager]
