import asyncio
import inspect
import json
from copy import copy, deepcopy
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
from langchain_core.tools.base import get_all_basemodel_annotations
from pydantic import BaseModel
from typing_extensions import Annotated, get_args, get_origin

from langgraph.errors import GraphBubbleUp
from langgraph.store.base import BaseStore
from langgraph.types import Command
from langgraph.utils.runnable import RunnableCallable

INVALID_TOOL_NAME_ERROR_TEMPLATE = (
    "Error: {requested_tool} is not a valid tool, try one of [{available_tools}]."
)
TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."


def msg_content_output(output: Any) -> Union[str, list[dict]]:
    recognized_content_block_types = ("image", "image_url", "text", "json")
    if isinstance(output, str):
        return output
    elif isinstance(output, list) and all(
        [
            isinstance(x, dict) and x.get("type") in recognized_content_block_types
            for x in output
        ]
    ):
        return output
    # Technically a list of strings is also valid message content but it's not currently
    # well tested that all chat models support this. And for backwards compatibility
    # we want to make sure we don't break any existing ToolNode usage.
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
                        "All types in the error handler error annotation must be Exception types. "
                        "For example, `def custom_handler(e: Union[ValueError, TypeError])`. "
                        f"Got '{first_param.annotation}' instead."
                    )

            exception_type = type_hints[first_param.name]
            if Exception in exception_type.__mro__:
                return (exception_type,)
            else:
                raise ValueError(
                    f"Arbitrary types are not supported in the error handler signature. "
                    "Please annotate the error with either a specific Exception type or a union of Exception types. "
                    "For example, `def custom_handler(e: ValueError)` or `def custom_handler(e: Union[ValueError, TypeError])`. "
                    f"Got '{exception_type}' instead."
                )

    # If no type information is available, return (Exception,) for backwards compatibility.
    return (Exception,)


class ToolNode(RunnableCallable):
    """A node that runs the tools called in the last AIMessage.

    It can be used either in StateGraph with a "messages" state key (or a custom key passed via ToolNode's 'messages_key').
    If multiple tool calls are requested, they will be run in parallel. The output will be
    a list of ToolMessages, one for each tool call.

    Tool calls can also be passed directly as a list of `ToolCall` dicts.

    Args:
        tools: A sequence of tools that can be invoked by the ToolNode.
        name: The name of the ToolNode in the graph. Defaults to "tools".
        tags: Optional tags to associate with the node. Defaults to None.
        handle_tool_errors: How to handle tool errors raised by tools inside the node. Defaults to True.
            Must be one of the following:

            - True: all errors will be caught and
                a ToolMessage with a default error message (TOOL_CALL_ERROR_TEMPLATE) will be returned.
            - str: all errors will be caught and
                a ToolMessage with the string value of 'handle_tool_errors' will be returned.
            - tuple[type[Exception], ...]: exceptions in the tuple will be caught and
                a ToolMessage with a default error message (TOOL_CALL_ERROR_TEMPLATE) will be returned.
            - Callable[..., str]: exceptions from the signature of the callable will be caught and
                a ToolMessage with the string value of the result of the 'handle_tool_errors' callable will be returned.
            - False: none of the errors raised by the tools will be caught
        messages_key: The state key in the input that contains the list of messages.
            The same key will be used for the output from the ToolNode.
            Defaults to "messages".

    The `ToolNode` is roughly analogous to:

    ```python
    tools_by_name = {tool.name: tool for tool in tools}
    def tool_node(state: dict):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}
    ```

    Tool calls can also be passed directly to a ToolNode. This can be useful when using
    the Send API, e.g., in a conditional edge:

    ```python
    def example_conditional_edge(state: dict) -> List[Send]:
        tool_calls = state["messages"][-1].tool_calls
        # If tools rely on state or store variables (whose values are not generated
        # directly by a model), you can inject them into the tool calls.
        tool_calls = [
            tool_node.inject_tool_args(call, state, store)
            for call in last_message.tool_calls
        ]
        return [Send("tools", [tool_call]) for tool_call in tool_calls]
    ```

    Important:
        - The input state can be one of the following:
            - A dict with a messages key containing a list of messages.
            - A list of messages.
            - A list of tool calls.
        - If operating on a message list, the last message must be an `AIMessage` with
            `tool_calls` populated.
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
        tool_calls, input_type = self._parse_input(input, store)
        config_list = get_config_list(config, len(tool_calls))
        input_types = [input_type] * len(tool_calls)
        with get_executor_for_config(config) as executor:
            outputs = [
                *executor.map(self._run_one, tool_calls, input_types, config_list)
            ]

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
        for output in outputs:
            if isinstance(output, Command):
                combined_outputs.append(output)
            else:
                combined_outputs.append(
                    [output] if input_type == "list" else {self.messages_key: [output]}
                )
        return combined_outputs

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
        tool_calls, input_type = self._parse_input(input, store)
        outputs = await asyncio.gather(
            *(self._arun_one(call, input_type, config) for call in tool_calls)
        )

        # preserve existing behavior for non-command tool outputs for backwards compatibility
        if not any(isinstance(output, Command) for output in outputs):
            # TypedDict, pydantic, dataclass, etc. should all be able to load from dict
            return outputs if input_type == "list" else {self.messages_key: outputs}

        # LangGraph will automatically handle list of Command and non-command node updates
        combined_outputs: list[
            Command | list[ToolMessage] | dict[str, list[ToolMessage]]
        ] = []
        for output in outputs:
            if isinstance(output, Command):
                combined_outputs.append(output)
            else:
                combined_outputs.append(
                    [output] if input_type == "list" else {self.messages_key: [output]}
                )
        return combined_outputs

    def _run_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        config: RunnableConfig,
    ) -> ToolMessage:
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message

        try:
            input = {**call, **{"type": "tool_call"}}
            response = self.tools_by_name[call["name"]].invoke(input, config)

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

    async def _arun_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        config: RunnableConfig,
    ) -> ToolMessage:
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
        store: Optional[BaseStore],
    ) -> Tuple[list[ToolCall], Literal["list", "dict", "tool_calls"]]:
        if isinstance(input, list):
            if isinstance(input[-1], dict) and input[-1].get("type") == "tool_call":
                input_type = "tool_calls"
                tool_calls = input
                return tool_calls, input_type
            else:
                input_type = "list"
                message: AnyMessage = input[-1]
        elif isinstance(input, dict) and (messages := input.get(self.messages_key, [])):
            input_type = "dict"
            message = messages[-1]
        elif messages := getattr(input, self.messages_key, None):
            # Assume dataclass-like state that can coerce from dict
            input_type = "dict"
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        tool_calls = [
            self.inject_tool_args(call, input, store) for call in message.tool_calls
        ]
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
        if isinstance(input, dict):
            tool_state_args = {
                tool_arg: input[state_field] if state_field else input
                for tool_arg, state_field in state_args.items()
            }

        else:
            tool_state_args = {
                tool_arg: getattr(input, state_field) if state_field else input
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
        """Injects the state and store into the tool call.

        Tool arguments with types annotated as `InjectedState` and `InjectedStore` are
        ignored in tool schemas for generation purposes. This method injects them into
        tool calls for tool invocation.

        Args:
            tool_call (ToolCall): The tool call to inject state and store into.
            input (Union[list[AnyMessage], dict[str, Any], BaseModel]): The input state
                to inject.
            store (Optional[BaseStore]): The store to inject.

        Returns:
            ToolCall: The tool call with injected state and store.
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
    """Use in the conditional_edge to route to the ToolNode if the last message

    has tool calls. Otherwise, route to the end.

    Args:
        state (Union[list[AnyMessage], dict[str, Any], BaseModel]): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.


    Examples:
        Create a custom ReAct-style agent with tools.

        ```pycon
        >>> from langchain_anthropic import ChatAnthropic
        >>> from langchain_core.tools import tool
        ...
        >>> from langgraph.graph import StateGraph
        >>> from langgraph.prebuilt import ToolNode, tools_condition
        >>> from langgraph.graph.message import add_messages
        ...
        >>> from typing import Annotated
        >>> from typing_extensions import TypedDict
        ...
        >>> @tool
        >>> def divide(a: float, b: float) -> int:
        ...     \"\"\"Return a / b.\"\"\"
        ...     return a / b
        ...
        >>> llm = ChatAnthropic(model="claude-3-haiku-20240307")
        >>> tools = [divide]
        ...
        >>> class State(TypedDict):
        ...     messages: Annotated[list, add_messages]
        >>>
        >>> graph_builder = StateGraph(State)
        >>> graph_builder.add_node("tools", ToolNode(tools))
        >>> graph_builder.add_node("chatbot", lambda state: {"messages":llm.bind_tools(tools).invoke(state['messages'])})
        >>> graph_builder.add_edge("tools", "chatbot")
        >>> graph_builder.add_conditional_edges(
        ...     "chatbot", tools_condition
        ... )
        >>> graph_builder.set_entry_point("chatbot")
        >>> graph = graph_builder.compile()
        >>> graph.invoke({"messages": {"role": "user", "content": "What's 329993 divided by 13662?"}})
        ```
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
    """Annotation for a Tool arg that is meant to be populated with the graph state.

    Any Tool argument annotated with InjectedState will be hidden from a tool-calling
    model, so that the model doesn't attempt to generate the argument. If using
    ToolNode, the appropriate graph state field will be automatically injected into
    the model-generated tool args.

    Args:
        field: The key from state to insert. If None, the entire state is expected to
            be passed in.

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
    """  # noqa: E501

    def __init__(self, field: Optional[str] = None) -> None:
        self.field = field


class InjectedStore(InjectedToolArg):
    """Annotation for a Tool arg that is meant to be populated with LangGraph store.

    Any Tool argument annotated with InjectedStore will be hidden from a tool-calling
    model, so that the model doesn't attempt to generate the argument. If using
    ToolNode, the appropriate store field will be automatically injected into
    the model-generated tool args. Note: if a graph is compiled with a store object,
    the store will be automatically propagated to the tools with InjectedStore args
    when using ToolNode.

    !!! Warning
        `InjectedStore` annotation requires `langchain-core >= 0.3.8`

    Example:
        ```python
        from typing import Any
        from typing_extensions import Annotated

        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool

        from langgraph.store.memory import InMemoryStore
        from langgraph.prebuilt import InjectedStore, ToolNode

        store = InMemoryStore()
        store.put(("values",), "foo", {"bar": 2})

        @tool
        def store_tool(x: int, my_store: Annotated[Any, InjectedStore()]) -> str:
            '''Do something with store.'''
            stored_value = my_store.get(("values",), "foo").value["bar"]
            return stored_value + x

        node = ToolNode([store_tool])

        tool_call = {"name": "store_tool", "args": {"x": 1}, "id": "1", "type": "tool_call"}
        state = {
            "messages": [AIMessage("", tool_calls=[tool_call])],
        }

        node.invoke(state, store=store)
        ```

        ```pycon
        {
            "messages": [
                ToolMessage(content='3', name='store_tool', tool_call_id='1'),
            ]
        }
        ```
    """  # noqa: E501


def _is_injection(
    type_arg: Any, injection_type: Union[Type[InjectedState], Type[InjectedStore]]
) -> bool:
    if isinstance(type_arg, injection_type) or (
        isinstance(type_arg, type) and issubclass(type_arg, injection_type)
    ):
        return True
    origin_ = get_origin(type_arg)
    if origin_ is Union or origin_ is Annotated:
        return any(_is_injection(ta, injection_type) for ta in get_args(type_arg))
    return False


def _get_state_args(tool: BaseTool) -> dict[str, Optional[str]]:
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
    full_schema = tool.get_input_schema()
    for name, type_ in get_all_basemodel_annotations(full_schema).items():
        injections = [
            type_arg
            for type_arg in get_args(type_)
            if _is_injection(type_arg, InjectedStore)
        ]
        if len(injections) > 1:
            ValueError(
                "A tool argument should not be annotated with InjectedStore more than "
                f"once. Received arg {name} with annotations {injections}."
            )
        elif len(injections) == 1:
            return name
        else:
            pass

    return None
