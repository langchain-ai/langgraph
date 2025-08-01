"""Tool execution node for LangGraph workflows.

This module provides prebuilt functionality for executing tools in LangGraph.

Tools are functions that models can call to interact with external systems,
APIs, databases, or perform computations.

The module implements several key design patterns:
- Parallel execution of multiple tool calls for efficiency
- Robust error handling with customizable error messages
- State injection for tools that need access to graph state
- Store injection for tools that need persistent storage
- Command-based state updates for advanced control flow

Key Components:
    ToolNode: Main class for executing tools in LangGraph workflows
    InjectedState: Annotation for injecting graph state into tools
    InjectedStore: Annotation for injecting persistent store into tools
    tools_condition: Utility function for conditional routing based on tool calls

Typical Usage:
    ```python
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    @tool
    def my_tool(x: int) -> str:
        return f"Result: {x}"

    tool_node = ToolNode([my_tool])
    ```
"""

import asyncio
import inspect
import json
from copy import copy, deepcopy
from dataclasses import replace
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    get_type_hints,
)

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    RemoveMessage,
    ToolCall,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import (
    get_config_list,
    get_executor_for_config,
)
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.tools import tool as create_tool
from langchain_core.tools.base import (
    TOOL_MESSAGE_BLOCK_TYPES,
    get_all_basemodel_annotations,
)
from pydantic import BaseModel
from typing_extensions import Annotated, get_args, get_origin

from langgraph._internal._runnable import RunnableCallable
from langgraph.errors import GraphBubbleUp
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt._internal import ToolCallWithContext
from langgraph.store.base import BaseStore
from langgraph.types import Command, Send

INVALID_TOOL_NAME_ERROR_TEMPLATE = (
    "Error: {requested_tool} is not a valid tool, try one of [{available_tools}]."
)
TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."


def msg_content_output(output: Any) -> Union[str, list[dict]]:
    """Convert tool output to valid message content format.

    LangChain ToolMessages accept either string content or a list of content blocks.
    This function ensures tool outputs are properly formatted for message consumption
    by attempting to preserve structured data when possible, falling back to JSON
    serialization or string conversion.

    Args:
        output: The raw output from a tool execution. Can be any type.

    Returns:
        Either a string representation of the output or a list of content blocks
        if the output is already in the correct format for structured content.

    Note:
        This function prioritizes backward compatibility by defaulting to JSON
        serialization rather than supporting all possible message content formats.
    """
    if isinstance(output, str):
        return output
    elif isinstance(output, list) and all(
        [
            isinstance(x, dict) and x.get("type") in TOOL_MESSAGE_BLOCK_TYPES
            for x in output
        ]
    ):
        return output
    # Technically a list of strings is also valid message content, but it's
    # not currently well tested that all chat models support this.
    # And for backwards compatibility we want to make sure we don't break
    # any existing ToolNode usage.
    else:
        try:
            return json.dumps(output, ensure_ascii=False)
        except Exception:
            return str(output)


def _handle_tool_error(
    e: Exception,
    *,
    flag: Union[
        bool,
        str,
        Callable[..., str],
        tuple[type[Exception], ...],
    ],
) -> str:
    """Generate error message content based on exception handling configuration.

    This function centralizes error message generation logic, supporting different
    error handling strategies configured via the ToolNode's handle_tool_errors
    parameter.

    Args:
        e: The exception that occurred during tool execution.
        flag: Configuration for how to handle the error. Can be:
            - bool: If True, use default error template
            - str: Use this string as the error message
            - Callable: Call this function with the exception to get error message
            - tuple: Not used in this context (handled by caller)

    Returns:
        A string containing the error message to include in the ToolMessage.

    Raises:
        ValueError: If flag is not one of the supported types.

    Note:
        The tuple case is handled by the caller through exception type checking,
        not by this function directly.
    """
    if isinstance(flag, (bool, tuple)):
        content = TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)
    else:
        raise ValueError(
            f"Got unexpected type of `handle_tool_error`. Expected bool, str "
            f"or callable. Received: {flag}"
        )
    return content


def _infer_handled_types(handler: Callable[..., str]) -> tuple[type[Exception], ...]:
    """Infer exception types handled by a custom error handler function.

    This function analyzes the type annotations of a custom error handler to determine
    which exception types it's designed to handle. This enables type-safe error handling
    where only specific exceptions are caught and processed by the handler.

    Args:
        handler: A callable that takes an exception and returns an error message string.
                The first parameter (after self/cls if present) should be type-annotated
                with the exception type(s) to handle.

    Returns:
        A tuple of exception types that the handler can process. Returns (Exception,)
        if no specific type information is available for backward compatibility.

    Raises:
        ValueError: If the handler's annotation contains non-Exception types or
                   if Union types contain non-Exception types.

    Note:
        This function supports both single exception types and Union types for
        handlers that need to handle multiple exception types differently.
    """
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())
    if params:
        # If it's a method, the first argument is typically 'self' or 'cls'
        if params[0].name in ["self", "cls"] and len(params) == 2:
            first_param = params[1]
        else:
            first_param = params[0]

        type_hints = get_type_hints(handler)
        if first_param.name in type_hints:
            origin = get_origin(first_param.annotation)
            if origin is Union:
                args = get_args(first_param.annotation)
                if all(issubclass(arg, Exception) for arg in args):
                    return tuple(args)
                else:
                    raise ValueError(
                        "All types in the error handler error annotation must be "
                        "Exception types. For example, "
                        "`def custom_handler(e: Union[ValueError, TypeError])`. "
                        f"Got '{first_param.annotation}' instead."
                    )

            exception_type = type_hints[first_param.name]
            if Exception in exception_type.__mro__:
                return (exception_type,)
            else:
                raise ValueError(
                    f"Arbitrary types are not supported in the error handler "
                    f"signature. Please annotate the error with either a "
                    f"specific Exception type or a union of Exception types. "
                    "For example, `def custom_handler(e: ValueError)` or "
                    "`def custom_handler(e: Union[ValueError, TypeError])`. "
                    f"Got '{exception_type}' instead."
                )

    # If no type information is available, return (Exception,)
    # for backwards compatibility.
    return (Exception,)


class ToolNode(RunnableCallable):
    """A node that runs the tools called in the last AIMessage.

    It can be used either in StateGraph with a "messages" state key (or a custom key passed via ToolNode's 'messages_key').
    If multiple tool calls are requested, they will be run in parallel. The output will be
    a list of ToolMessages, one for each tool call.

    Tool calls can also be passed directly as a list of `ToolCall` dicts.

    Args:
        tools: A sequence of tools that can be invoked by this node. Tools can be
            BaseTool instances or plain functions that will be converted to tools.
        name: The name identifier for this node in the graph. Used for debugging
            and visualization. Defaults to "tools".
        tags: Optional metadata tags to associate with the node for filtering
            and organization. Defaults to None.
        handle_tool_errors: Configuration for error handling during tool execution.
            Defaults to True. Supports multiple strategies:

            - True: Catch all errors and return a ToolMessage with the default
              error template containing the exception details.
            - str: Catch all errors and return a ToolMessage with this custom
              error message string.
            - tuple[type[Exception], ...]: Only catch exceptions of the specified
              types and return default error messages for them.
            - Callable[..., str]: Catch exceptions matching the callable's signature
              and return the string result of calling it with the exception.
            - False: Disable error handling entirely, allowing exceptions to propagate.

        messages_key: The key in the state dictionary that contains the message list.
            This same key will be used for the output ToolMessages. Defaults to "messages".

    Example:
        Basic usage with simple tools:

        ```python
        from langgraph.prebuilt import ToolNode
        from langchain_core.tools import tool

        @tool
        def calculator(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b

        tool_node = ToolNode([calculator])
        ```

        Custom error handling:

        ```python
        def handle_math_errors(e: ZeroDivisionError) -> str:
            return "Cannot divide by zero!"

        tool_node = ToolNode([calculator], handle_tool_errors=handle_math_errors)
        ```

        Direct tool call execution:

        ```python
        tool_calls = [{"name": "calculator", "args": {"a": 5, "b": 3}, "id": "1", "type": "tool_call"}]
        result = tool_node.invoke(tool_calls)
        ```

    Note:
        The ToolNode expects input in one of three formats:
        1. A dictionary with a messages key containing a list of messages
        2. A list of messages directly
        3. A list of tool call dictionaries

        When using message formats, the last message must be an AIMessage with
        tool_calls populated. The node automatically extracts and processes these
        tool calls concurrently.

        For advanced use cases involving state injection or store access, tools
        can be annotated with InjectedState or InjectedStore to receive graph
        context automatically.
    """

    name: str = "ToolNode"

    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        name: str = "tools",
        tags: Optional[list[str]] = None,
        handle_tool_errors: Union[
            bool, str, Callable[..., str], tuple[type[Exception], ...]
        ] = True,
        messages_key: str = "messages",
    ) -> None:
        """Initialize the ToolNode with the provided tools and configuration.

        Args:
            tools: Sequence of tools to make available for execution.
            name: Node name for graph identification.
            tags: Optional metadata tags.
            handle_tool_errors: Error handling configuration.
            messages_key: State key containing messages.
        """
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.tools_by_name: dict[str, BaseTool] = {}
        self.tool_to_state_args: dict[str, dict[str, Optional[str]]] = {}
        self.tool_to_store_arg: dict[str, Optional[str]] = {}
        self.handle_tool_errors = handle_tool_errors
        self.messages_key = messages_key
        for tool_ in tools:
            if not isinstance(tool_, BaseTool):
                tool_ = create_tool(tool_)
            self.tools_by_name[tool_.name] = tool_
            self.tool_to_state_args[tool_.name] = _get_state_args(tool_)
            self.tool_to_store_arg[tool_.name] = _get_store_arg(tool_)

    def _func(
        self,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
        config: RunnableConfig,
        *,
        store: Optional[BaseStore],
    ) -> Any:
        tool_calls, input_type = self._parse_input(input)
        tool_calls = [self.inject_tool_args(call, input, store) for call in tool_calls]
        config_list = get_config_list(config, len(tool_calls))
        input_types = [input_type] * len(tool_calls)
        with get_executor_for_config(config) as executor:
            outputs = [
                *executor.map(self._run_one, tool_calls, input_types, config_list)
            ]

        return self._combine_tool_outputs(outputs, input_type)

    async def _afunc(
        self,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
        config: RunnableConfig,
        *,
        store: Optional[BaseStore],
    ) -> Any:
        tool_calls, input_type = self._parse_input(input)
        tool_calls = [self.inject_tool_args(call, input, store) for call in tool_calls]
        outputs = await asyncio.gather(
            *(self._arun_one(call, input_type, config) for call in tool_calls)
        )

        return self._combine_tool_outputs(outputs, input_type)

    def _combine_tool_outputs(
        self,
        outputs: list[ToolMessage],
        input_type: Literal["list", "dict", "tool_calls"],
    ) -> list[Union[Command, list[ToolMessage], dict[str, list[ToolMessage]]]]:
        # preserve existing behavior for non-command tool outputs for backwards
        # compatibility
        if not any(isinstance(output, Command) for output in outputs):
            # TypedDict, pydantic, dataclass, etc. should all be able to load from dict
            return outputs if input_type == "list" else {self.messages_key: outputs}

        # LangGraph will automatically handle list of Command and non-command node
        # updates
        combined_outputs: list[
            Command | list[ToolMessage] | dict[str, list[ToolMessage]]
        ] = []

        # combine all parent commands with goto into a single parent command
        parent_command: Optional[Command] = None
        for output in outputs:
            if isinstance(output, Command):
                if (
                    output.graph is Command.PARENT
                    and isinstance(output.goto, list)
                    and all(isinstance(send, Send) for send in output.goto)
                ):
                    if parent_command:
                        parent_command = replace(
                            parent_command,
                            goto=cast(list[Send], parent_command.goto) + output.goto,
                        )
                    else:
                        parent_command = Command(graph=Command.PARENT, goto=output.goto)
                else:
                    combined_outputs.append(output)
            else:
                combined_outputs.append(
                    [output] if input_type == "list" else {self.messages_key: [output]}
                )

        if parent_command:
            combined_outputs.append(parent_command)
        return combined_outputs

    def _run_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        config: RunnableConfig,
    ) -> ToolMessage:
        """Run a single tool call synchronously."""
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message
        try:
            call_args = {**call, **{"type": "tool_call"}}
            response = self.tools_by_name[call["name"]].invoke(call_args, config)

        # GraphInterrupt is a special exception that will always be raised.
        # It can be triggered in the following scenarios,
        # Where GraphInterrupt(GraphBubbleUp) is raised from an `interrupt` invocation most commonly:
        # (1) a GraphInterrupt is raised inside a tool
        # (2) a GraphInterrupt is raised inside a graph node for a graph called as a tool
        # (3) a GraphInterrupt is raised when a subgraph is interrupted inside a graph called as a tool
        # (2 and 3 can happen in a "supervisor w/ tools" multi-agent architecture)
        except GraphBubbleUp as e:
            raise e
        except Exception as e:
            if isinstance(self.handle_tool_errors, tuple):
                handled_types: tuple = self.handle_tool_errors
            elif callable(self.handle_tool_errors):
                handled_types = _infer_handled_types(self.handle_tool_errors)
            else:
                # default behavior is catching all exceptions
                handled_types = (Exception,)

            # Unhandled
            if not self.handle_tool_errors or not isinstance(e, handled_types):
                raise e
            # Handled
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_errors)
            return ToolMessage(
                content=content,
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )

        if isinstance(response, Command):
            return self._validate_tool_command(response, call, input_type)
        elif isinstance(response, ToolMessage):
            response.content = cast(
                Union[str, list], msg_content_output(response.content)
            )
            return response
        else:
            raise TypeError(
                f"Tool {call['name']} returned unexpected type: {type(response)}"
            )

    async def _arun_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        config: RunnableConfig,
    ) -> ToolMessage:
        """Run a single tool call asynchronously."""
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message

        try:
            input = {**call, **{"type": "tool_call"}}
            response = await self.tools_by_name[call["name"]].ainvoke(input, config)

        # GraphInterrupt is a special exception that will always be raised.
        # It can be triggered in the following scenarios:
        # (1) a NodeInterrupt is raised inside a tool
        # (2) a NodeInterrupt is raised inside a graph node for a graph called as a tool
        # (3) a GraphInterrupt is raised when a subgraph is interrupted inside a graph called as a tool
        # (2 and 3 can happen in a "supervisor w/ tools" multi-agent architecture)
        except GraphBubbleUp as e:
            raise e
        except Exception as e:
            if isinstance(self.handle_tool_errors, tuple):
                handled_types: tuple = self.handle_tool_errors
            elif callable(self.handle_tool_errors):
                handled_types = _infer_handled_types(self.handle_tool_errors)
            else:
                # default behavior is catching all exceptions
                handled_types = (Exception,)

            # Unhandled
            if not self.handle_tool_errors or not isinstance(e, handled_types):
                raise e
            # Handled
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_errors)

            return ToolMessage(
                content=content,
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )

        if isinstance(response, Command):
            return self._validate_tool_command(response, call, input_type)
        elif isinstance(response, ToolMessage):
            response.content = cast(
                Union[str, list], msg_content_output(response.content)
            )
            return response
        else:
            raise TypeError(
                f"Tool {call['name']} returned unexpected type: {type(response)}"
            )

    def _parse_input(
        self,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
    ) -> Tuple[list[ToolCall], Literal["list", "dict", "tool_calls"]]:
        input_type: Literal["list", "dict", "tool_calls"]
        if isinstance(input, list):
            if isinstance(input[-1], dict) and input[-1].get("type") == "tool_call":
                input_type = "tool_calls"
                tool_calls = cast(list[ToolCall], input)
                return tool_calls, input_type
            else:
                input_type = "list"
                messages = input
        elif (
            isinstance(input, dict) and input.get("__type") == "tool_call_with_context"
        ):
            # mypy will not be able to type narrow correctly since the signature
            # for input contains dict[str, Any]. We'd need to type dict[str, Any]
            # before we can apply correct typing.
            input = cast(ToolCallWithContext, input)  # type: ignore[assignment]
            input_type = "tool_calls"
            return [input["tool_call"]], input_type
        elif isinstance(input, dict) and (messages := input.get(self.messages_key, [])):
            input_type = "dict"
        elif messages := getattr(input, self.messages_key, []):
            # Assume dataclass-like state that can coerce from dict
            input_type = "dict"
        else:
            raise ValueError("No message found in input")

        try:
            latest_ai_message = next(
                m for m in reversed(messages) if isinstance(m, AIMessage)
            )
        except StopIteration:
            raise ValueError("No AIMessage found in input")

        tool_calls = [call for call in latest_ai_message.tool_calls]
        return tool_calls, input_type

    def _validate_tool_call(self, call: ToolCall) -> Optional[ToolMessage]:
        if (requested_tool := call["name"]) not in self.tools_by_name:
            content = INVALID_TOOL_NAME_ERROR_TEMPLATE.format(
                requested_tool=requested_tool,
                available_tools=", ".join(self.tools_by_name.keys()),
            )
            return ToolMessage(
                content, name=requested_tool, tool_call_id=call["id"], status="error"
            )
        else:
            return None

    def _inject_state(
        self,
        tool_call: ToolCall,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
    ) -> ToolCall:
        state_args = self.tool_to_state_args[tool_call["name"]]
        if state_args and isinstance(input, list):
            required_fields = list(state_args.values())
            if (
                len(required_fields) == 1
                and required_fields[0] == self.messages_key
                or required_fields[0] is None
            ):
                input = {self.messages_key: input}
            else:
                err_msg = (
                    f"Invalid input to ToolNode. Tool {tool_call['name']} requires "
                    f"graph state dict as input."
                )
                if any(state_field for state_field in state_args.values()):
                    required_fields_str = ", ".join(f for f in required_fields if f)
                    err_msg += f" State should contain fields {required_fields_str}."
                raise ValueError(err_msg)

        if isinstance(input, dict) and input.get("__type") == "tool_call_with_context":
            state = input["state"]
        else:
            state = input

        if isinstance(state, dict):
            tool_state_args = {
                tool_arg: state[state_field] if state_field else state
                for tool_arg, state_field in state_args.items()
            }
        else:
            tool_state_args = {
                tool_arg: getattr(state, state_field) if state_field else state
                for tool_arg, state_field in state_args.items()
            }

        tool_call["args"] = {
            **tool_call["args"],
            **tool_state_args,
        }
        return tool_call

    def _inject_store(
        self, tool_call: ToolCall, store: Optional[BaseStore]
    ) -> ToolCall:
        store_arg = self.tool_to_store_arg[tool_call["name"]]
        if not store_arg:
            return tool_call

        if store is None:
            raise ValueError(
                "Cannot inject store into tools with InjectedStore annotations - "
                "please compile your graph with a store."
            )

        tool_call["args"] = {
            **tool_call["args"],
            store_arg: store,
        }
        return tool_call

    def inject_tool_args(
        self,
        tool_call: ToolCall,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
        store: Optional[BaseStore],
    ) -> ToolCall:
        """Inject graph state and store into tool call arguments.

        This method enables tools to access graph context that should not be controlled
        by the model. Tools can declare dependencies on graph state or persistent storage
        using InjectedState and InjectedStore annotations. This method automatically
        identifies these dependencies and injects the appropriate values.

        The injection process preserves the original tool call structure while adding
        the necessary context arguments. This allows tools to be both model-callable
        and context-aware without exposing internal state management to the model.

        Args:
            tool_call: The tool call dictionary to augment with injected arguments.
                Must contain 'name', 'args', 'id', and 'type' fields.
            input: The current graph state to inject into tools requiring state access.
                Can be a message list, state dictionary, or BaseModel instance.
            store: The persistent store instance to inject into tools requiring storage.
                Will be None if no store is configured for the graph.

        Returns:
            A new ToolCall dictionary with the same structure as the input but with
            additional arguments injected based on the tool's annotation requirements.

        Raises:
            ValueError: If a tool requires store injection but no store is provided,
                       or if state injection requirements cannot be satisfied.

        Note:
            This method is automatically called during tool execution but can also
            be used manually when working with the Send API or custom routing logic.
            The injection is performed on a copy of the tool call to avoid mutating
            the original.
        """
        if tool_call["name"] not in self.tools_by_name:
            return tool_call

        tool_call_copy: ToolCall = copy(tool_call)
        tool_call_with_state = self._inject_state(tool_call_copy, input)
        tool_call_with_store = self._inject_store(tool_call_with_state, store)
        return tool_call_with_store

    def _validate_tool_command(
        self,
        command: Command,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
    ) -> Command:
        if isinstance(command.update, dict):
            # input type is dict when ToolNode is invoked with a dict input (e.g. {"messages": [AIMessage(..., tool_calls=[...])]})
            if input_type not in ("dict", "tool_calls"):
                raise ValueError(
                    f"Tools can provide a dict in Command.update only when using dict with '{self.messages_key}' key as ToolNode input, "
                    f"got: {command.update} for tool '{call['name']}'"
                )

            updated_command = deepcopy(command)
            state_update = cast(dict[str, Any], updated_command.update) or {}
            messages_update = state_update.get(self.messages_key, [])
        elif isinstance(command.update, list):
            # input type is list when ToolNode is invoked with a list input (e.g. [AIMessage(..., tool_calls=[...])])
            if input_type != "list":
                raise ValueError(
                    f"Tools can provide a list of messages in Command.update only when using list of messages as ToolNode input, "
                    f"got: {command.update} for tool '{call['name']}'"
                )

            updated_command = deepcopy(command)
            messages_update = updated_command.update
        else:
            return command

        # convert to message objects if updates are in a dict format
        messages_update = convert_to_messages(messages_update)

        # no validation needed if all messages are being removed
        if messages_update == [RemoveMessage(id=REMOVE_ALL_MESSAGES)]:
            return updated_command

        has_matching_tool_message = False
        for message in messages_update:
            if not isinstance(message, ToolMessage):
                continue

            if message.tool_call_id == call["id"]:
                message.name = call["name"]
                has_matching_tool_message = True

        # validate that we always have a ToolMessage matching the tool call in
        # Command.update if command is sent to the CURRENT graph
        if updated_command.graph is None and not has_matching_tool_message:
            example_update = (
                '`Command(update={"messages": [ToolMessage("Success", tool_call_id=tool_call_id), ...]}, ...)`'
                if input_type == "dict"
                else '`Command(update=[ToolMessage("Success", tool_call_id=tool_call_id), ...], ...)`'
            )
            raise ValueError(
                f"Expected to have a matching ToolMessage in Command.update for tool '{call['name']}', got: {messages_update}. "
                "Every tool call (LLM requesting to call a tool) in the message history MUST have a corresponding ToolMessage. "
                f"You can fix it by modifying the tool to return {example_update}."
            )
        return updated_command


def tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:
    """Conditional routing function for tool-calling workflows.

    This utility function implements the standard conditional logic for ReAct-style
    agents: if the last AI message contains tool calls, route to the tool execution
    node; otherwise, end the workflow. This pattern is fundamental to most tool-calling
    agent architectures.

    The function handles multiple state formats commonly used in LangGraph applications,
    making it flexible for different graph designs while maintaining consistent behavior.

    Args:
        state: The current graph state to examine for tool calls. Supported formats:
            - List of messages (for MessageGraph)
            - Dictionary containing a messages key (for StateGraph)
            - BaseModel instance with a messages attribute
        messages_key: The key or attribute name containing the message list in the state.
            This allows customization for graphs using different state schemas.
            Defaults to "messages".

    Returns:
        Either "tools" if tool calls are present in the last AI message, or "__end__"
        to terminate the workflow. These are the standard routing destinations for
        tool-calling conditional edges.

    Raises:
        ValueError: If no messages can be found in the provided state format.

    Example:
        Basic usage in a ReAct agent:

        ```python
        from langgraph.graph import StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition
        from typing_extensions import TypedDict

        class State(TypedDict):
            messages: list

        graph = StateGraph(State)
        graph.add_node("llm", call_model)
        graph.add_node("tools", ToolNode([my_tool]))
        graph.add_conditional_edges(
            "llm",
            tools_condition,  # Routes to "tools" or "__end__"
            {"tools": "tools", "__end__": "__end__"}
        )
        ```

        Custom messages key:

        ```python
        def custom_condition(state):
            return tools_condition(state, messages_key="chat_history")
        ```

    Note:
        This function is designed to work seamlessly with ToolNode and standard
        LangGraph patterns. It expects the last message to be an AIMessage when
        tool calls are present, which is the standard output format for tool-calling
        language models.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


class InjectedState(InjectedToolArg):
    """Annotation for injecting graph state into tool arguments.

    This annotation enables tools to access graph state without exposing state
    management details to the language model. Tools annotated with InjectedState
    receive state data automatically during execution while remaining invisible
    to the model's tool-calling interface.

    Args:
        field: Optional key to extract from the state dictionary. If None, the entire
            state is injected. If specified, only that field's value is injected.
            This allows tools to request specific state components rather than
            processing the full state structure.

    Example:
        ```python
        from typing import List
        from typing_extensions import Annotated, TypedDict

        from langchain_core.messages import BaseMessage, AIMessage
        from langchain_core.tools import tool

        from langgraph.prebuilt import InjectedState, ToolNode


        class AgentState(TypedDict):
            messages: List[BaseMessage]
            foo: str

        @tool
        def state_tool(x: int, state: Annotated[dict, InjectedState]) -> str:
            '''Do something with state.'''
            if len(state["messages"]) > 2:
                return state["foo"] + str(x)
            else:
                return "not enough messages"

        @tool
        def foo_tool(x: int, foo: Annotated[str, InjectedState("foo")]) -> str:
            '''Do something else with state.'''
            return foo + str(x + 1)

        node = ToolNode([state_tool, foo_tool])

        tool_call1 = {"name": "state_tool", "args": {"x": 1}, "id": "1", "type": "tool_call"}
        tool_call2 = {"name": "foo_tool", "args": {"x": 1}, "id": "2", "type": "tool_call"}
        state = {
            "messages": [AIMessage("", tool_calls=[tool_call1, tool_call2])],
            "foo": "bar",
        }
        node.invoke(state)
        ```

        ```pycon
        [
            ToolMessage(content='not enough messages', name='state_tool', tool_call_id='1'),
            ToolMessage(content='bar2', name='foo_tool', tool_call_id='2')
        ]
        ```

    Note:
        - InjectedState arguments are automatically excluded from tool schemas
          presented to language models
        - ToolNode handles the injection process during execution
        - Tools can mix regular arguments (controlled by the model) with injected
          arguments (controlled by the system)
        - State injection occurs after the model generates tool calls but before
          tool execution
    """  # noqa: E501

    def __init__(self, field: Optional[str] = None) -> None:
        self.field = field


class InjectedStore(InjectedToolArg):
    """Annotation for injecting persistent store into tool arguments.

    This annotation enables tools to access LangGraph's persistent storage system
    without exposing storage details to the language model. Tools annotated with
    InjectedStore receive the store instance automatically during execution while
    remaining invisible to the model's tool-calling interface.

    The store provides persistent, cross-session data storage that tools can use
    for maintaining context, user preferences, or any other data that needs to
    persist beyond individual workflow executions.

    !!! Warning
        `InjectedStore` annotation requires `langchain-core >= 0.3.8`

    Example:
        ```python
        from typing_extensions import Annotated
        from langchain_core.tools import tool
        from langgraph.store.memory import InMemoryStore
        from langgraph.prebuilt import InjectedStore, ToolNode

        @tool
        def save_preference(
            key: str,
            value: str,
            store: Annotated[Any, InjectedStore()]
        ) -> str:
            \"\"\"Save user preference to persistent storage.\"\"\"
            store.put(("preferences",), key, value)
            return f"Saved {key} = {value}"

        @tool
        def get_preference(
            key: str,
            store: Annotated[Any, InjectedStore()]
        ) -> str:
            \"\"\"Retrieve user preference from persistent storage.\"\"\"
            result = store.get(("preferences",), key)
            return result.value if result else "Not found"
        ```

        Usage with ToolNode and graph compilation:

        ```python
        from langgraph.graph import StateGraph
        from langgraph.store.memory import InMemoryStore

        store = InMemoryStore()
        tool_node = ToolNode([save_preference, get_preference])

        graph = StateGraph(State)
        graph.add_node("tools", tool_node)
        compiled_graph = graph.compile(store=store)  # Store is injected automatically
        ```

        Cross-session persistence:

        ```python
        # First session
        result1 = graph.invoke({"messages": [HumanMessage("Save my favorite color as blue")]})

        # Later session - data persists
        result2 = graph.invoke({"messages": [HumanMessage("What's my favorite color?")]})
        ```

    Note:
        - InjectedStore arguments are automatically excluded from tool schemas
          presented to language models
        - The store instance is automatically injected by ToolNode during execution
        - Tools can access namespaced storage using the store's get/put methods
        - Store injection requires the graph to be compiled with a store instance
        - Multiple tools can share the same store instance for data consistency
    """  # noqa: E501


def _is_injection(
    type_arg: Any, injection_type: Union[Type[InjectedState], Type[InjectedStore]]
) -> bool:
    """Check if a type argument represents an injection annotation.

    This utility function determines whether a type annotation indicates that
    an argument should be injected with state or store data. It handles both
    direct annotations and nested annotations within Union or Annotated types.

    Args:
        type_arg: The type argument to check for injection annotations.
        injection_type: The injection type to look for (InjectedState or InjectedStore).

    Returns:
        True if the type argument contains the specified injection annotation.
    """
    if isinstance(type_arg, injection_type) or (
        isinstance(type_arg, type) and issubclass(type_arg, injection_type)
    ):
        return True
    origin_ = get_origin(type_arg)
    if origin_ is Union or origin_ is Annotated:
        return any(_is_injection(ta, injection_type) for ta in get_args(type_arg))
    return False


def _get_state_args(tool: BaseTool) -> dict[str, Optional[str]]:
    """Extract state injection mappings from tool annotations.

    This function analyzes a tool's input schema to identify arguments that should
    be injected with graph state. It processes InjectedState annotations to build
    a mapping of tool argument names to state field names.

    Args:
        tool: The tool to analyze for state injection requirements.

    Returns:
        A dictionary mapping tool argument names to state field names. If a field
        name is None, the entire state should be injected for that argument.
    """
    full_schema = tool.get_input_schema()
    tool_args_to_state_fields: dict = {}

    for name, type_ in get_all_basemodel_annotations(full_schema).items():
        injections = [
            type_arg
            for type_arg in get_args(type_)
            if _is_injection(type_arg, InjectedState)
        ]
        if len(injections) > 1:
            raise ValueError(
                "A tool argument should not be annotated with InjectedState more than "
                f"once. Received arg {name} with annotations {injections}."
            )
        elif len(injections) == 1:
            injection = injections[0]
            if isinstance(injection, InjectedState) and injection.field:
                tool_args_to_state_fields[name] = injection.field
            else:
                tool_args_to_state_fields[name] = None
        else:
            pass
    return tool_args_to_state_fields


def _get_store_arg(tool: BaseTool) -> Optional[str]:
    """Extract store injection argument from tool annotations.

    This function analyzes a tool's input schema to identify the argument that
    should be injected with the graph store. Only one store argument is supported
    per tool.

    Args:
        tool: The tool to analyze for store injection requirements.

    Returns:
        The name of the argument that should receive the store injection, or None
        if no store injection is required.

    Raises:
        ValueError: If a tool argument has multiple InjectedStore annotations.
    """
    full_schema = tool.get_input_schema()
    for name, type_ in get_all_basemodel_annotations(full_schema).items():
        injections = [
            type_arg
            for type_arg in get_args(type_)
            if _is_injection(type_arg, InjectedStore)
        ]
        if len(injections) > 1:
            raise ValueError(
                "A tool argument should not be annotated with InjectedStore more than "
                f"once. Received arg {name} with annotations {injections}."
            )
        elif len(injections) == 1:
            return name
        else:
            pass

    return None
