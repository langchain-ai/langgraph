"""Custom run priority for LangGraph.

.. warning::
    This API is in beta and may change in future versions.

This module lets developers decide the priority settings of each thread with
an async function that the server calls at the first run on a thread with no
settings. The function is registered on an object that `langgraph.json` names,
and runs server-side.
"""

from __future__ import annotations

import functools
import inspect
import typing
import warnings
from collections.abc import Awaitable, Callable

from langgraph_sdk.encryption import DuplicateHandlerError, LangGraphBetaWarning
from langgraph_sdk.priority import types

_AssignHandler = Callable[
    [types.PriorityContext], Awaitable[types.PrioritySettings | None]
]
_AssignHandlerT = typing.TypeVar("_AssignHandlerT", bound=_AssignHandler)


@functools.lru_cache(maxsize=1)
def _warn_priority_beta() -> None:
    warnings.warn(
        "The Priority API is in beta and may change in future versions.",
        LangGraphBetaWarning,
        stacklevel=4,
    )


def _validate_handler(fn: typing.Callable) -> None:
    """Validate that the assign handler is async and takes one parameter.

    Raises:
        TypeError: If the handler is not an async function or has wrong parameter count
    """
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(
            f"Priority assign handler must be an async function, got {type(fn)}"
        )

    sig = inspect.signature(fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(params) != 1:
        raise TypeError(
            "Priority assign handler must accept exactly 1 parameter (ctx), "
            f"got {len(params)}"
        )


class Priority:
    """Decide the priority settings of your LangGraph application's threads.

    .. warning::
        This API is in beta and may change in future versions.

    To use, create a separate Python file and add the path to the file to your
    LangGraph API configuration file (`langgraph.json`). Within that file, create
    an instance of the Priority class and register an assign function with
    `@priority.assign`.

    Example `langgraph.json` file:

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "agent": "./my_agent/agent.py:graph"
      },
      "env": ".env",
      "priority": {
        "path": "./priority.py:priority"
      }
    }
    ```

    The server calls the assign function at the first run on a thread with no
    settings, which can come long after the thread was created, and stores the
    settings it returns on the thread. Later runs on the thread use the stored
    settings. The function can be called more than once for one thread, for
    example after it raises, and only one result is stored, so keep it free of
    side effects.

    ???+ example "Basic Usage"

        ```python
        from langgraph_sdk import Priority, PriorityContext, PrioritySettings

        priority = Priority()

        @priority.assign
        async def assign(ctx: PriorityContext) -> PrioritySettings | None:
            return {
                "priority": -3 if "premium" in ctx.permissions else 0,
                # Stored unencrypted, so an opaque ID rather than personal data.
                "fairness_group": ctx.user["org_id"],
            }
        ```
    """

    __slots__ = ("_assign_handler",)

    types = types
    """Reference to priority type definitions.

    Provides access to the type definitions used by the priority system:
    PriorityContext and PrioritySettings.
    """

    def __init__(self) -> None:
        """Initialize the Priority instance."""
        _warn_priority_beta()
        self._assign_handler: _AssignHandler | None = None

    def assign(self, fn: _AssignHandlerT) -> _AssignHandlerT:
        """Register the function that assigns a thread's priority settings.

        The function receives a `PriorityContext` and returns `PrioritySettings`,
        or `None` for the defaults.

        Example:
            ```python
            @priority.assign
            async def assign(ctx: PriorityContext) -> PrioritySettings | None:
                if ctx.cron_id is not None:
                    return {"priority": 2}
                return None
            ```

        Args:
            fn: The assign handler function

        Returns:
            The registered handler function

        Raises:
            DuplicateHandlerError: If an assign handler is already registered
            TypeError: If the handler has an invalid signature
        """
        if self._assign_handler is not None:
            raise DuplicateHandlerError("Priority assign handler already registered")
        _validate_handler(fn)
        self._assign_handler = fn
        return fn
