from typing import Annotated

from langgraph.managed.base import ManagedValue


class IsLastStepManager(ManagedValue[bool]):
    def __call__(self, step: int) -> bool:
        return step == self.config.get("recursion_limit", 0) - 1


IsLastStep = Annotated[bool, IsLastStepManager]
