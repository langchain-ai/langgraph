from functools import wraps
from typing import Callable, Literal, Optional, Union

from langchain_core.tools import StructuredTool

from langgraph.graph import GraphCommand


class HandoffTool(StructuredTool):
    # should this be a list?
    goto: str
    response_format: Literal["content_and_artifact"] = "content_and_artifact"
    return_direct: Literal[True] = True


def handoff(
    goto: str,
    *,
    tool_message: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Union[HandoffTool, Callable[[Callable], HandoffTool]]:
    """Decorator that creates a tool that can hand off control to another node while updating state.
    Args:
        goto: The name of the node to hand off control to.
        tool_message: Optional message to return when the tool is called.
            If provided, creates a simple handoff tool that returns this message.
            If not provided, `handoff` will be used as a decorator and
                expects a function that returns (message, state_update).
        name: Optional name for the tool. Defaults to the function name if decorating a function.
        description: Optional description for the tool. Defaults to the function's docstring if decorating a function.

    Returns:
        If tool_message is provided, returns a HandoffTool instance.
        If tool_message is not provided, returns a decorator that creates a HandoffTool from the decorated function.
        The decorated function should return a tuple of (message, state_update).

    Examples:
        Handoff as a function (requires tool_message):
            >>> handoff_tool = handoff(
            ...     goto="other_agent",
            ...     tool_message="Handing off to other agent",
            ...     name="transfer_to_other_agent",
            ...     description="Transfer to other agent"
            ... )

        Handoff as a decorator (allows passing state updates from the function):
            >>> @handoff(goto="other_agent")
            ... def handoff_with_state(user_input: str):
            ...     '''Transfer to other agent.'''
            ...     # Process input and create state updates
            ...     state_update = {"key": "value"}
            ...     return f"Handing off to other agent", state_update

    """
    # Handle case when used as a plain decorator without arguments
    if callable(goto):
        raise ValueError(
            "You must provide 'goto' parameter when using @handoff as a decorator."
        )

    if tool_message is not None:
        command = GraphCommand(goto=goto)

        def func():
            return tool_message, command

        return HandoffTool.from_function(
            func,
            goto=goto,
            name=name,
            description=description or func.__doc__,
            response_format="content_and_artifact",
            args_schema=None,
            infer_schema=True,
            return_direct=True,
        )
    else:

        def decorator(func):
            @wraps(func)
            def wrapped_func(*args, **kwargs):
                # Call the original function which returns (message, state_update)
                message, state_update = func(*args, **kwargs)
                # Create command with both goto and state update
                command = GraphCommand(goto=goto, update=state_update)
                return message, command

            # Create and return the tool
            return HandoffTool.from_function(
                wrapped_func,
                goto=goto,
                name=name or func.__name__,
                description=description or func.__doc__,
                response_format="content_and_artifact",
                args_schema=None,
                infer_schema=True,
                return_direct=True,
            )

        return decorator
