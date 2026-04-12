from typing import Annotated

from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.managed.base import ManagedValue

__all__ = ("IsLastStep", "RemainingStepsManager")


class IsLastStepManager(ManagedValue[bool]):
    """Managed value that indicates whether the current step is the last step.

    Inject `IsLastStep` into a node's type annotations to receive a boolean
    flag that is `True` only when the graph is on its final allowed step
    (i.e. one step before the recursion limit).

    This is useful for gracefully handling the recursion limit — for example,
    by returning a final response instead of invoking more tools.

    Example:
        ```python
        from typing import Annotated
        from langgraph.managed import IsLastStep

        def my_node(state: State, is_last_step: IsLastStep) -> dict:
            if is_last_step:
                return {"messages": [{"role": "assistant", "content": "Stopping here."}]}
            # ... normal processing
        ```
    """

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        """Return `True` if the current step is the last permitted step.

        Args:
            scratchpad: The current Pregel execution scratchpad.

        Returns:
            `True` if `scratchpad.step == scratchpad.stop - 1`, else `False`.
        """
        return scratchpad.step == scratchpad.stop - 1


IsLastStep = Annotated[bool, IsLastStepManager]
"""Managed value annotation that resolves to `True` on the graph's final step.

Use this as a type annotation for a node parameter to receive a boolean
indicating whether the current step is the last step before the recursion
limit is reached.

Example:
    ```python
    from langgraph.managed import IsLastStep

    def call_model(state: State, is_last_step: IsLastStep) -> dict:
        if is_last_step:
            # Force a final response without further tool calls
            ...
    ```
"""


class RemainingStepsManager(ManagedValue[int]):
    """Managed value that reports the number of steps remaining before the recursion limit.

    Inject `RemainingSteps` into a node's type annotations to receive an
    integer count of how many more steps the graph may execute.

    A value of `1` means the current step is the last permitted step.
    A value of `0` would indicate the limit has been reached (not normally
    observable in practice, since execution stops beforehand).

    Example:
        ```python
        from langgraph.managed import RemainingSteps

        def my_node(state: State, remaining_steps: RemainingSteps) -> dict:
            if remaining_steps < 3:
                # Wrap up early
                ...
        ```
    """

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        """Return the number of steps remaining before the recursion limit.

        Args:
            scratchpad: The current Pregel execution scratchpad.

        Returns:
            The number of remaining steps, computed as
            `scratchpad.stop - scratchpad.step`.
        """
        return scratchpad.stop - scratchpad.step


RemainingSteps = Annotated[int, RemainingStepsManager]
"""Managed value annotation that resolves to the number of steps remaining.

Use this as a type annotation for a node parameter to receive the count of
steps the graph may still execute before hitting the recursion limit.

Example:
    ```python
    from langgraph.managed import RemainingSteps

    def call_model(state: State, remaining_steps: RemainingSteps) -> dict:
        if remaining_steps < 5:
            # Not enough steps left for more tool calls — respond directly
            ...
    ```
"""
