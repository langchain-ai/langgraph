from typing import Annotated

from langgraph.managed.base import ManagedValue
from langgraph.pregel.types import PregelExecutableTask


class IsLastStepManager(ManagedValue[bool]):
    def __call__(self, step: int, task: PregelExecutableTask) -> bool:
        return step == self.config["recursion_limit"] - 1


IsLastStep = Annotated[bool, IsLastStepManager]
