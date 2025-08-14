from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)
from warnings import warn

from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
    LanguageModelLike,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnableSequence,
)
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from pydantic import BaseModel
from typing_extensions import Annotated, NotRequired, TypedDict

from langgraph._internal._runnable import RunnableCallable, RunnableLike
from langgraph._internal._typing import MISSING
from langgraph.errors import ErrorCode, create_error_message
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt._internal._typing import (
    ContextT,
    PreConfiguredChatModel,
    SyncOrAsync,
)
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Command, Send
from langgraph.warnings import LangGraphDeprecatedSinceV10

BASE_MODEL_DOC = BaseModel.__doc__

StructuredResponse = Union[dict, BaseModel]
StructuredResponseSchema = Union[dict, type[BaseModel]]

F = TypeVar("F", bound=Callable[..., Any])


# We create the AgentState that we will pass around
# This simply involves a list of messages
# We want steps to return messages to append to the list
# So we annotate the messages attribute with `add_messages` reducer
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    remaining_steps: NotRequired[RemainingSteps]


class AgentStatePydantic(BaseModel):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    remaining_steps: RemainingSteps = 25


class AgentStateWithStructuredResponse(AgentState):
    """The state of the agent with a structured response."""

    structured_response: StructuredResponse


class AgentStateWithStructuredResponsePydantic(AgentStatePydantic):
    """The state of the agent with a structured response."""

    structured_response: StructuredResponse


StateSchema = TypeVar("StateSchema", bound=Union[AgentState, AgentStatePydantic])
StateSchemaType = Type[StateSchema]


PROMPT_RUNNABLE_NAME = "Prompt"

Prompt = Union[
    SystemMessage,
    str,
    Callable[[StateSchema], LanguageModelInput],
    Runnable[StateSchema, LanguageModelInput],
]


def _get_state_value(state: StateSchema, key: str, default: Any = None) -> Any:
    return (
        state.get(key, default)
        if isinstance(state, dict)
        else getattr(state, key, default)
    )


def _get_prompt_runnable(prompt: Optional[Prompt]) -> Runnable:
    prompt_runnable: Runnable
    if prompt is None:
        prompt_runnable = RunnableCallable(
            lambda state: _get_state_value(state, "messages"), name=PROMPT_RUNNABLE_NAME
        )
    elif isinstance(prompt, str):
        _system_message: BaseMessage = SystemMessage(content=prompt)
        prompt_runnable = RunnableCallable(
            lambda state: [_system_message] + _get_state_value(state, "messages"),
            name=PROMPT_RUNNABLE_NAME,
        )
    elif isinstance(prompt, SystemMessage):
        prompt_runnable = RunnableCallable(
            lambda state: [prompt] + _get_state_value(state, "messages"),
            name=PROMPT_RUNNABLE_NAME,
        )
    elif inspect.iscoroutinefunction(prompt):
        prompt_runnable = RunnableCallable(
            None,
            prompt,
            name=PROMPT_RUNNABLE_NAME,
        )
    elif callable(prompt):
        prompt_runnable = RunnableCallable(
            prompt,
            name=PROMPT_RUNNABLE_NAME,
        )
    elif isinstance(prompt, Runnable):
        prompt_runnable = prompt
    else:
        raise ValueError(f"Got unexpected type for `prompt`: {type(prompt)}")

    return prompt_runnable


def _get_model(model: LanguageModelLike) -> BaseChatModel:
    """Get the underlying model from a RunnableBinding or return the model itself."""
    if isinstance(model, RunnableSequence):
        model = next(
            (
                step
                for step in model.steps
                if isinstance(step, (RunnableBinding, BaseChatModel))
            ),
            model,
        )

    if isinstance(model, RunnableBinding):
        model = model.bound

    if not isinstance(model, BaseChatModel):
        raise TypeError(
            f"Expected `model` to be a ChatModel or RunnableBinding (e.g. model.bind_tools(...)), got {type(model)}"
        )

    return model


def _validate_chat_history(
    messages: Sequence[BaseMessage],
) -> None:
    """Validate that all tool calls in AIMessages have a corresponding ToolMessage."""
    all_tool_calls = [
        tool_call
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    tool_call_ids_with_results = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    tool_calls_without_results = [
        tool_call
        for tool_call in all_tool_calls
        if tool_call["id"] not in tool_call_ids_with_results
    ]
    if not tool_calls_without_results:
        return

    error_message = create_error_message(
        message="Found AIMessages with tool_calls that do not have a corresponding ToolMessage. "
        f"Here are the first few of those tool calls: {tool_calls_without_results[:3]}.\n\n"
        "Every tool call (LLM requesting to call a tool) in the message history MUST have a corresponding ToolMessage "
        "(result of a tool invocation to return to the LLM) - this is required by most LLM providers.",
        error_code=ErrorCode.INVALID_CHAT_HISTORY,
    )
    raise ValueError(error_message)


class _StructuredToolInfo(TypedDict):
    """Internal type for tracking structured output tool metadata.

    This contains all necessary information to handle structured responses
    generated via tool calls, including the original schema, its type classification,
    and the corresponding tool implementation used by the tools strategy.
    """

    schema: StructuredResponseSchema
    """The original schema provided for structured output (Pydantic model or dict schema)."""
    kind: Literal["pydantic", "dict"]
    """Classification of the schema type for proper response construction."""
    tool: BaseTool
    """LangChain tool instance created from the schema for model binding."""


TModel = TypeVar("TModel", bound=BaseModel)
# Keep dict as an option too
SchemaType = Union[dict, Type[TModel]]


@dataclass(frozen=True)
class ToolOutput(Generic[TModel]):
    """Structured output format for model responses."""

    schemas: Sequence[SchemaType]
    """The schema of the structured output the model may return."""
    tool_choice: bool = True
    """Whether to use the tools strategy for structured output."""


class _AgentBuilder:
    """Internal builder class for constructing and agent."""

    def __init__(
        self,
        model: Union[
            str,
            BaseChatModel,
            PreConfiguredChatModel,
            SyncOrAsync[[StateSchema, Runtime[ContextT]], BaseModel],
            SyncOrAsync[
                [StateSchema, Runtime[ContextT]],
                Awaitable[PreConfiguredChatModel],
            ],
        ],
        tools: Union[Sequence[Union[BaseTool, Callable, dict[str, Any]]], ToolNode],
        *,
        prompt: Optional[Prompt] = None,
        response_format: Optional[ToolOutput] = None,
        pre_model_hook: Optional[RunnableLike] = None,
        post_model_hook: Optional[RunnableLike] = None,
        state_schema: Optional[StateSchemaType] = None,
        context_schema: Optional[Type[Any]] = None,
        version: Literal["v1", "v2"] = "v2",
        name: Optional[str] = None,
        store: Optional[BaseStore] = None,
        use_individual_tool_nodes: bool = False,
    ) -> None:
        """Initialize the agent builder."""
        if version == "v1" and use_individual_tool_nodes:
            # This edge case is ill-defined. "v1" refers specifically to a single
            # tools node that handles all tool calls.
            raise AssertionError(
                "The 'use_individual_tool_nodes' option is only supported "
                "in version 'v2' agents."
            )

        if isinstance(model, Runnable) and not isinstance(model, BaseChatModel):
            # Then we allow for a preconfigured model at least for now.
            if not hasattr(model, "bound") or not isinstance(
                model.bound, BaseChatModel
            ):
                raise TypeError(
                    "Expected `model` to be a BaseChatModel or a chat model that "
                    f"was pre-configured using `.bind()`. Instead got {type(model)}"
                )

            # Then it's a runnable binding. We don't want any pre-bound tools.
            if (kwargs := getattr(model, "kwargs", {})) and "tools" in kwargs:
                raise ValueError(
                    "The `model` parameter should not have pre-bound tools. "
                    "You are getting this error because the chat model you are using"
                    "was pre-bound with tools somewhere. The code that binds tools "
                    "looks like this: `model.bind_tools(...)`. "
                    "Remove the `bind_tools` call and pass the unbound model "
                    "and the `tools` parameter separately."
                )

        self.model = model
        self.tools = tools
        self.prompt = prompt
        self.response_format = response_format
        self.pre_model_hook = pre_model_hook
        self.post_model_hook = post_model_hook
        self.state_schema = state_schema
        self.context_schema = context_schema
        self.version = version
        self.name = name
        self.store = store
        self._use_individual_tool_nodes = use_individual_tool_nodes

        self._setup_tools()
        self._setup_state_schema()
        self._setup_structured_output_tools()
        self._setup_model()

    def _setup_tools(self) -> None:
        """Setup tool-related attributes."""
        if isinstance(self.tools, ToolNode):
            self._tool_classes = list(self.tools.tools_by_name.values())
            self._tool_node = self.tools
            self._llm_builtin_tools = []
        else:
            self._llm_builtin_tools = [t for t in self.tools if isinstance(t, dict)]
            self._tool_node = ToolNode(
                [t for t in self.tools if not isinstance(t, dict)]
            )
            self._tool_classes = list(self._tool_node.tools_by_name.values())

        self._should_return_direct = {
            t.name for t in self._tool_classes if t.return_direct
        }
        self._tool_calling_enabled = len(self._tool_classes) > 0

    def _setup_structured_output_tools(self) -> None:
        """Set up structured output tools tracking for the tools strategy.

        This method implements the "tools" strategy for structured output by:
        1. Converting response format schemas to LangChain tools
        2. Creating metadata for proper response reconstruction
        3. Handling both Pydantic models and dict schemas

        Future strategies (json_mode, guided) will have separate setup methods.
        """
        self.structured_output_tools: dict[str, _StructuredToolInfo] = {}
        if self.response_format is not None:
            response_format = self.response_format

            # Handle ToolOutput wrapper
            if isinstance(response_format, ToolOutput):
                # Use tools strategy - process each schema in the ToolOutput
                for schema in response_format.schemas:
                    kwargs = {}
                    if isinstance(schema, type) and issubclass(schema, BaseModel):
                        # Patch for behavior in langchain-core for vanilla BaseModel
                        description = (
                            "" if schema.__doc__ == BASE_MODEL_DOC else schema.__doc__
                        )
                        kwargs = {"description": description}
                        kind: Literal["pydantic", "dict"] = "pydantic"
                    else:
                        kind = "dict"

                    tool = create_tool(schema, **kwargs)
                    self.structured_output_tools[tool.name] = _StructuredToolInfo(
                        schema=schema,
                        kind=kind,
                        tool=tool,
                    )
            else:
                # Handle legacy format (direct schema or tuple)
                if isinstance(response_format, tuple):
                    _, schema = response_format
                else:
                    schema = response_format
                kwargs = {}
                if isinstance(schema, type) and issubclass(schema, BaseModel):
                    # Patch for behavior in langchain-core for vanilla BaseModel
                    description = (
                        "" if schema.__doc__ == BASE_MODEL_DOC else schema.__doc__
                    )
                    kwargs = {"description": description}
                    kind: Literal["pydantic", "dict"] = "pydantic"
                else:
                    kind = "dict"

                tool = create_tool(schema, **kwargs)
                self.structured_output_tools[tool.name] = _StructuredToolInfo(
                    schema=schema,
                    kind=kind,
                    tool=tool,
                )

    def _setup_state_schema(self) -> None:
        """Setup state schema with validation."""
        if self.state_schema is not None:
            required_keys = {"messages", "remaining_steps"}
            if self.response_format is not None:
                required_keys.add("structured_response")

            schema_keys = set(get_type_hints(self.state_schema))
            if missing_keys := required_keys - schema_keys:
                raise ValueError(
                    f"Missing required key(s) {missing_keys} in state_schema"
                )

            self._final_state_schema = self.state_schema
        else:
            self._final_state_schema = (
                AgentStateWithStructuredResponse
                if self.response_format is not None
                else AgentState
            )

    def _handle_structured_response_tool_calls(
        self, response: AIMessage
    ) -> Optional[Command]:
        """Handle tool calls that match structured output tools using the tools strategy.

        Args:
            response: The AI message containing potential tool calls

        Returns:
            Command with structured response update if found, None otherwise

        Raises:
            AssertionError: If multiple structured responses are returned
        """
        if not response.tool_calls:
            return None

        structured_tool_calls = [
            tool_call
            for tool_call in response.tool_calls
            if tool_call["name"] in self.structured_output_tools
        ]

        if len(structured_tool_calls) > 1:
            raise AssertionError(
                "Model incorrectly returned multiple structured responses. "
                "Behavior has not yet been defined in this case."
            )

        if len(structured_tool_calls) == 1:
            tool_call = structured_tool_calls[0]
            messages = [
                response,
                ToolMessage(
                    content="ok!",
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                ),
            ]
            structured_tool_info = self.structured_output_tools[tool_call["name"]]
            args = tool_call["args"]

            if structured_tool_info["kind"] == "pydantic":
                schema = structured_tool_info["schema"]
                structured_response = schema(**args)
            elif structured_tool_info["kind"] == "dict":
                structured_response = tool_call["args"]
            else:
                msg = (
                    f"Internal error: Unsupported structured "
                    f"response kind: {structured_tool_info['kind']}"
                )
                raise AssertionError(msg)

            return Command(
                update={
                    "messages": messages,
                    "structured_response": structured_response,
                }
            )

        return None

    def _setup_model(self) -> None:
        """Setup model-related attributes."""
        self._is_dynamic_model = not isinstance(
            self.model, (str, Runnable)
        ) and callable(self.model)
        self._is_async_dynamic_model = (
            self._is_dynamic_model and inspect.iscoroutinefunction(self.model)
        )

        if not self._is_dynamic_model:
            model = self.model
            if isinstance(model, str):
                try:
                    from langchain.chat_models import (  # type: ignore[import-not-found]
                        init_chat_model,
                    )
                except ImportError:
                    raise ImportError(
                        "Please install langchain (`pip install langchain`) to use '<provider>:<model>' string syntax for `model` parameter."
                    )
                model = init_chat_model(model)

            if len(self._tool_classes + self._llm_builtin_tools) > 0:
                model = cast(BaseChatModel, model).bind_tools(
                    self._tool_classes + self._llm_builtin_tools  # type: ignore[operator]
                )
            # Extract just the model part for direct invocation
            self._static_model: Optional[Runnable] = model  # type: ignore[assignment]
        else:
            self._static_model = None

    def _resolve_model(
        self, state: StateSchema, runtime: Runtime[ContextT]
    ) -> LanguageModelLike:
        """Resolve the model to use, handling both static and dynamic models."""
        if self._is_dynamic_model:
            return self.model(state, runtime)  # type: ignore[operator, arg-type]
        else:
            return self._static_model

    async def _aresolve_model(
        self, state: StateSchema, runtime: Runtime[ContextT]
    ) -> LanguageModelLike:
        """Async resolve the model to use, handling both static and dynamic models."""
        if self._is_async_dynamic_model:
            dynamic_model = cast(
                Callable[[StateSchema, Runtime[ContextT]], Awaitable[BaseChatModel]],
                self.model,
            )
            resolved_model = await dynamic_model(state, runtime)
            return resolved_model
        elif self._is_dynamic_model:
            return self.model(state, runtime)  # type: ignore[arg-type, operator]
        else:
            return self._static_model

    def create_model_node(self) -> RunnableCallable:
        """Create the 'agent' node that calls the LLM."""

        def _get_model_input_state(state: StateSchema) -> StateSchema:
            if self.pre_model_hook is not None:
                messages = _get_state_value(
                    state, "llm_input_messages"
                ) or _get_state_value(state, "messages")
                error_msg = (
                    f"Expected input to call_model to have 'llm_input_messages' "
                    f"or 'messages' key, but got {state}"
                )
            else:
                messages = _get_state_value(state, "messages")
                error_msg = (
                    f"Expected input to call_model to "
                    f"have 'messages' key, but got {state}"
                )

            if messages is None:
                raise ValueError(error_msg)

            _validate_chat_history(messages)

            if isinstance(self._final_state_schema, type) and issubclass(
                self._final_state_schema, BaseModel
            ):
                # we're passing messages under `messages` key, as this
                # is expected by the prompt
                state.messages = messages  # type: ignore
            else:
                state["messages"] = messages  # type: ignore
            return state

        def _are_more_steps_needed(state: StateSchema, response: BaseMessage) -> bool:
            has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
            all_tools_return_direct = (
                all(
                    call["name"] in self._should_return_direct
                    for call in response.tool_calls
                )
                if isinstance(response, AIMessage)
                else False
            )
            remaining_steps = _get_state_value(state, "remaining_steps", None)
            if remaining_steps is not None:
                if remaining_steps < 1 and all_tools_return_direct:
                    return True
                elif remaining_steps < 2 and has_tool_calls:
                    return True
            return False

        def call_model(
            state: StateSchema, runtime: Runtime[ContextT], config: RunnableConfig
        ) -> dict[str, Any] | Command:
            """Call the model with the current state and return the response."""
            if self._is_async_dynamic_model:
                raise RuntimeError(
                    "Async model callable provided but agent invoked synchronously. "
                    "Use agent.ainvoke() or agent.astream(), or provide a sync model callable."
                )

            model_input = _get_model_input_state(state)
            model = self._resolve_model(state, runtime)

            # Get prompt runnable and invoke it first to prepare messages
            prompt_runnable = _get_prompt_runnable(self.prompt)
            prepared_messages = prompt_runnable.invoke(model_input, config)

            # Then invoke the model with the prepared messages
            response = cast(AIMessage, model.invoke(prepared_messages, config))
            response.name = self.name

            if _are_more_steps_needed(state, response):
                return {
                    "messages": [
                        AIMessage(
                            id=response.id,
                            content="Sorry, need more steps to process this request.",
                        )
                    ]
                }

            # Check if any tool calls match structured output tools
            structured_command = self._handle_structured_response_tool_calls(response)
            if structured_command:
                return structured_command

            return {"messages": [response]}

        async def acall_model(
            state: StateSchema, runtime: Runtime[ContextT], config: RunnableConfig
        ) -> dict[str, Any] | Command:
            """Call the model with the current state and return the response."""
            model_input = _get_model_input_state(state)

            model = await self._aresolve_model(state, runtime)

            # Get prompt runnable and invoke it first to prepare messages
            prompt_runnable = _get_prompt_runnable(self.prompt)
            prepared_messages = await prompt_runnable.ainvoke(model_input, config)

            # Then invoke the model with the prepared messages
            response = cast(
                AIMessage,
                await model.ainvoke(prepared_messages, config),
            )
            response.name = self.name
            if _are_more_steps_needed(state, response):
                return {
                    "messages": [
                        AIMessage(
                            id=response.id,
                            content="Sorry, need more steps to process this request.",
                        )
                    ]
                }

            # Check if any tool calls match structured output tools
            structured_command = self._handle_structured_response_tool_calls(response)
            if structured_command:
                return structured_command

            return {"messages": [response]}

        return RunnableCallable(call_model, acall_model)

    def _get_input_schema(self) -> StateSchemaType:
        """Get input schema for model node."""
        if self.pre_model_hook is not None:
            if isinstance(self._final_state_schema, type) and issubclass(
                self._final_state_schema, BaseModel
            ):
                from pydantic import create_model

                return create_model(
                    "CallModelInputSchema",
                    llm_input_messages=(list[AnyMessage], ...),
                    __base__=self._final_state_schema,
                )
            else:

                class CallModelInputSchema(self._final_state_schema):  # type: ignore
                    llm_input_messages: list[AnyMessage]

                return CallModelInputSchema
        else:
            return self._final_state_schema

    def create_structured_response_node(self) -> Optional[RunnableCallable]:
        """Create the 'generate_structured_response' node if configured."""
        if self.response_format is None:
            return None

        def generate_structured_response(
            state: StateSchema, runtime: Runtime[ContextT], config: RunnableConfig
        ) -> StateSchema:
            if self._is_async_dynamic_model:
                raise RuntimeError(
                    "Async model callable provided but agent invoked synchronously. "
                    "Use agent.ainvoke() or agent.astream(), or provide a sync model callable."
                )

            messages = _get_state_value(state, "messages")
            structured_response_schema = self.response_format
            if isinstance(self.response_format, tuple):
                system_prompt, structured_response_schema = self.response_format
                messages = [SystemMessage(content=system_prompt)] + list(messages)

            resolved_model = self._resolve_model(state, runtime)
            model_with_structured_output = _get_model(
                resolved_model
            ).with_structured_output(
                cast(StructuredResponseSchema, structured_response_schema)
            )
            response = model_with_structured_output.invoke(messages, config)
            return {"structured_response": response}

        async def agenerate_structured_response(
            state: StateSchema, runtime: Runtime[ContextT], config: RunnableConfig
        ) -> StateSchema:
            messages = _get_state_value(state, "messages")
            structured_response_schema = self.response_format
            if isinstance(self.response_format, tuple):
                system_prompt, structured_response_schema = self.response_format
                messages = [SystemMessage(content=system_prompt)] + list(messages)

            resolved_model = await self._aresolve_model(state, runtime)
            model_with_structured_output = _get_model(
                resolved_model
            ).with_structured_output(
                cast(StructuredResponseSchema, structured_response_schema)
            )
            response = await model_with_structured_output.ainvoke(messages, config)
            return {"structured_response": response}

        return RunnableCallable(
            generate_structured_response, agenerate_structured_response
        )

    def create_model_router(self) -> Callable[[StateSchema], Union[str, list[Send]]]:
        """Create routing function for model node conditional edges."""

        def should_continue(state: StateSchema) -> Union[str, list[Send]]:
            messages = _get_state_value(state, "messages")
            last_message = messages[-1]

            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                if self.post_model_hook is not None:
                    return "post_model_hook"
                else:
                    return END
            else:
                if self.version == "v1":
                    return "tools"
                elif self.version == "v2":
                    if self.post_model_hook is not None:
                        return "post_model_hook"

                    if self._use_individual_tool_nodes:
                        # Route to individual tool nodes
                        tool_calls = [
                            self._tool_node.inject_tool_args(call, state, self.store)  # type: ignore[arg-type]
                            for call in last_message.tool_calls
                        ]
                        return [
                            Send(tool_call["name"], [tool_call])
                            for tool_call in tool_calls
                        ]
                    else:
                        # Use the original combined tools node
                        tool_calls = [
                            self._tool_node.inject_tool_args(call, state, self.store)  # type: ignore[arg-type]
                            for call in last_message.tool_calls
                        ]
                        return [Send("tools", [tool_call]) for tool_call in tool_calls]

        return should_continue

    def create_post_model_hook_router(
        self,
    ) -> Callable[[StateSchema], Union[str, list[Send]]]:
        """Create a routing function for post_model_hook node conditional edges."""

        def post_model_hook_router(state: StateSchema) -> Union[str, list[Send]]:
            messages = _get_state_value(state, "messages")
            tool_messages = [
                m.tool_call_id for m in messages if isinstance(m, ToolMessage)
            ]
            last_ai_message = next(
                m for m in reversed(messages) if isinstance(m, AIMessage)
            )
            pending_tool_calls = [
                c for c in last_ai_message.tool_calls if c["id"] not in tool_messages
            ]

            if pending_tool_calls:
                pending_tool_calls = [
                    self._tool_node.inject_tool_args(call, state, self.store)  # type: ignore[arg-type]
                    for call in pending_tool_calls
                ]

                if self._use_individual_tool_nodes:
                    return [
                        # TODO: Add validation for tool name being a valid node name
                        # and one that matches a tool.
                        Send(tool_call["name"], [tool_call])
                        for tool_call in pending_tool_calls
                    ]
                else:
                    return [
                        Send("tools", [tool_call]) for tool_call in pending_tool_calls
                    ]
            elif isinstance(messages[-1], ToolMessage):
                return self._get_entry_point()
            else:
                return END

        return post_model_hook_router

    def create_tools_router(self) -> Optional[Callable[[StateSchema], str]]:
        """Create a routing function for tools node conditional edges."""
        if not self._should_return_direct:
            return None

        def route_tool_responses(state: StateSchema) -> str:
            messages = _get_state_value(state, "messages")
            for m in reversed(messages):
                if not isinstance(m, ToolMessage):
                    break
                if m.name in self._should_return_direct:
                    return END

            if isinstance(m, AIMessage) and m.tool_calls:
                if any(
                    call["name"] in self._should_return_direct for call in m.tool_calls
                ):
                    return END

            return self._get_entry_point()

        return route_tool_responses

    def add_tool_node(self, tool: BaseTool) -> RunnableCallable:
        """Create a node that executes a specific tool.

        This method creates a node that wraps a single tool in a ToolNode
        and executes it, returning the result as {"messages": [message]}.

        Args:
            tool: The tool to wrap in a node.

        Returns:
            A RunnableCallable node that can be added to the graph.
        """
        tool_node = ToolNode([tool])
        return tool_node

    def _get_entry_point(self) -> str:
        """Get the workflow entry point."""
        return "pre_model_hook" if self.pre_model_hook else "agent"

    def _get_model_paths(self) -> list[str]:
        """Get possible edge destinations from model node."""
        paths = []
        if self._tool_calling_enabled:
            if self._use_individual_tool_nodes:
                paths.extend([tool.name for tool in self._tool_classes])
            else:
                paths.append("tools")
        paths.append(END)
        return paths

    def _get_post_model_hook_paths(self) -> list[str]:
        """Get possible edge destinations from post_model_hook node."""
        paths = []
        if self._tool_calling_enabled:
            if self._use_individual_tool_nodes:
                paths = [self._get_entry_point()] + [
                    tool.name for tool in self._tool_classes
                ]
            else:
                paths = [self._get_entry_point(), "tools"]
        paths.append(END)
        return paths

    def build(
        self,
    ) -> StateGraph:
        """Build the agent workflow graph."""
        workflow = StateGraph(
            state_schema=self._final_state_schema,
            context_schema=self.context_schema,
        )

        # Set entry point
        workflow.set_entry_point(self._get_entry_point())

        # Add nodes
        workflow.add_node(
            "agent", self.create_model_node(), input_schema=self._get_input_schema()
        )

        if self._tool_calling_enabled:
            if self._use_individual_tool_nodes:
                # Add individual tool nodes
                for tool in self._tool_classes:
                    tool_node = self.add_tool_node(tool)
                    workflow.add_node(tool.name, tool_node)
            else:
                # Add the combined tools node
                workflow.add_node("tools", self._tool_node)

        if self.pre_model_hook:
            workflow.add_node("pre_model_hook", self.pre_model_hook)  # type: ignore[arg-type]

        if self.post_model_hook:
            workflow.add_node("post_model_hook", self.post_model_hook)  # type: ignore[arg-type]

        # Add edges
        if self.pre_model_hook:
            workflow.add_edge("pre_model_hook", "agent")

        if self.post_model_hook:
            workflow.add_edge("agent", "post_model_hook")
            post_hook_paths = self._get_post_model_hook_paths()
            if len(post_hook_paths) == 1:
                # No need for a conditional edge if there's only one path
                workflow.add_edge("post_model_hook", post_hook_paths[0])
            else:
                workflow.add_conditional_edges(
                    "post_model_hook",
                    self.create_post_model_hook_router(),
                    path_map=post_hook_paths,
                )
        else:
            model_paths = self._get_model_paths()
            if len(model_paths) == 1:
                # No need for a conditional edge if there's only one path
                workflow.add_edge("agent", model_paths[0])
            else:
                workflow.add_conditional_edges(
                    "agent",
                    self.create_model_router(),
                    path_map=model_paths,
                )

        if self._tool_calling_enabled:
            if self._use_individual_tool_nodes:
                # Add edges for individual tool nodes
                tools_router = self.create_tools_router()
                for tool in self._tool_classes:
                    tool_node_name = tool.name
                    if tools_router:
                        workflow.add_conditional_edges(
                            tool_node_name,
                            tools_router,
                            path_map=[self._get_entry_point(), END],
                        )
                    else:
                        workflow.add_edge(tool_node_name, self._get_entry_point())
            else:
                # In some cases, tools can return directly. In these cases
                # we add a conditional edge from the tools node to the END node
                # instead of going to the entry point.
                tools_router = self.create_tools_router()
                if tools_router:
                    workflow.add_conditional_edges(
                        "tools",
                        tools_router,
                        path_map=[self._get_entry_point(), END],
                    )
                else:
                    workflow.add_edge("tools", self._get_entry_point())

        return workflow


def create_react_agent(
    model: Union[
        str,
        BaseChatModel,
        PreConfiguredChatModel,
        SyncOrAsync[[StateSchema, Runtime[ContextT]], BaseModel],
        SyncOrAsync[
            [StateSchema, Runtime[ContextT]],
            Awaitable[PreConfiguredChatModel],
        ],
    ],
    tools: Union[Sequence[Union[BaseTool, Callable, dict[str, Any]]], ToolNode],
    *,
    prompt: Optional[Prompt] = None,
    response_format: Optional[
        Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]
    ] = None,
    pre_model_hook: Optional[RunnableLike] = None,
    post_model_hook: Optional[RunnableLike] = None,
    state_schema: Optional[StateSchemaType] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v2",
    name: Optional[str] = None,
    use_individual_tool_nodes: bool = False,
    **deprecated_kwargs: Any,
) -> CompiledStateGraph:
    """Creates an agent graph that calls tools in a loop until a stopping condition is met.

    For more details on using `create_react_agent`, visit [Agents](https://langchain-ai.github.io/langgraph/agents/overview/) documentation.

    Args:
        model: The language model for the agent. Supports static and dynamic
            model selection.

            - **Static model**: A chat model instance (e.g., `ChatOpenAI()`) or
              string identifier (e.g., `"openai:gpt-4"`)
            - **Dynamic model**: A callable with signature
              `(state, runtime) -> BaseChatModel` that returns different models
              based on runtime context
              If the model has tools bound via `.bind_tools()` or other configurations,
              the return type should be a Runnable[LanguageModelInput, BaseMessage]
              Coroutines are also supported, allowing for asynchronous model selection.

            Dynamic functions receive graph state and runtime, enabling
            context-dependent model selection. Must return a `BaseChatModel`
            instance. For tool calling, bind tools using `.bind_tools()`.
            Bound tools must be a subset of the `tools` parameter.

            Dynamic model example:
            ```python
            from dataclasses import dataclass

            @dataclass
            class ModelContext:
                model_name: str = "gpt-3.5-turbo"

            # Instantiate models globally
            gpt4_model = ChatOpenAI(model="gpt-4")
            gpt35_model = ChatOpenAI(model="gpt-3.5-turbo")

            def select_model(state: AgentState, runtime: Runtime[ModelContext]) -> ChatOpenAI:
                model_name = runtime.context.model_name
                model = gpt4_model if model_name == "gpt-4" else gpt35_model
                return model.bind_tools(tools)
            ```

            !!! note "Dynamic Model Requirements"
                Ensure returned models have appropriate tools bound via
                `.bind_tools()` and support required functionality. Bound tools
                must be a subset of those specified in the `tools` parameter.

        tools: A list of tools or a ToolNode instance.
            If an empty list is provided, the agent will consist of a single LLM node without tool calling.
        prompt: An optional prompt for the LLM. Can take a few different forms:

            - str: This is converted to a SystemMessage and added to the beginning of the list of messages in state["messages"].
            - SystemMessage: this is added to the beginning of the list of messages in state["messages"].
            - Callable: This function should take in full graph state and the output is then passed to the language model.
            - Runnable: This runnable should take in full graph state and the output is then passed to the language model.

        response_format: An optional schema for the final agent output.

            If provided, output will be formatted to match the given schema and returned in the 'structured_response' state key.
            If not provided, `structured_response` will not be present in the output state.
            Can be passed in as:

                - an OpenAI function/tool schema,
                - a JSON Schema,
                - a TypedDict class,
                - or a Pydantic class.
                - a tuple (prompt, schema), where schema is one of the above.
                    The prompt will be used together with the model that is being used to generate the structured response.

            !!! Important
                `response_format` requires the model to support `.with_structured_output`

            !!! Note
                The graph will make a separate call to the LLM to generate the structured response after the agent loop is finished.
                This is not the only strategy to get structured responses, see more options in [this guide](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/).

        pre_model_hook: An optional node to add before the `agent` node (i.e., the node that calls the LLM).
            Useful for managing long message histories (e.g., message trimming, summarization, etc.).
            Pre-model hook must be a callable or a runnable that takes in current graph state and returns a state update in the form of
                ```python
                # At least one of `messages` or `llm_input_messages` MUST be provided
                {
                    # If provided, will UPDATE the `messages` in the state
                    "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), ...],
                    # If provided, will be used as the input to the LLM,
                    # and will NOT UPDATE `messages` in the state
                    "llm_input_messages": [...],
                    # Any other state keys that need to be propagated
                    ...
                }
                ```

            !!! Important
                At least one of `messages` or `llm_input_messages` MUST be provided and will be used as an input to the `agent` node.
                The rest of the keys will be added to the graph state.

            !!! Warning
                If you are returning `messages` in the pre-model hook, you should OVERWRITE the `messages` key by doing the following:

                ```python
                {
                    "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]
                    ...
                }
                ```
        post_model_hook: An optional node to add after the `agent` node (i.e., the node that calls the LLM).
            Useful for implementing human-in-the-loop, guardrails, validation, or other post-processing.
            Post-model hook must be a callable or a runnable that takes in current graph state and returns a state update.

            !!! Note
                Only available with `version="v2"`.
        state_schema: An optional state schema that defines graph state.
            Must have `messages` and `remaining_steps` keys.
            Defaults to `AgentState` that defines those two keys.
        context_schema: An optional schema for runtime context.
        checkpointer: An optional checkpoint saver object. This is used for persisting
            the state of the graph (e.g., as chat memory) for a single thread (e.g., a single conversation).
        store: An optional store object. This is used for persisting data
            across multiple threads (e.g., multiple conversations / users).
        interrupt_before: An optional list of node names to interrupt before.
            Should be one of the following: "agent", "tools".
            This is useful if you want to add a user confirmation or other interrupt before taking an action.
        interrupt_after: An optional list of node names to interrupt after.
            Should be one of the following: "agent", "tools".
            This is useful if you want to return directly or run additional processing on an output.
        debug: A flag indicating whether to enable debug mode.
        version: Determines the version of the graph to create.
            Can be one of:

            - `"v1"`: The tool node processes a single message. All tool
                calls in the message are executed in parallel within the tool node.
            - `"v2"`: The tool node processes a tool call.
                Tool calls are distributed across multiple instances of the tool
                node using the [Send](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
                API.
        name: An optional name for the CompiledStateGraph.
            This name will be automatically used when adding ReAct agent graph to another graph as a subgraph node -
            particularly useful for building multi-agent systems.
        use_individual_tool_nodes: A flag indicating whether to use individual tool nodes for each tool.
            If set to `True`, each tool will have its own node in the graph.
            This has been added for the beta period. The default behavior will change
            in v1.0.0 to use individual tool nodes.

    !!! warning "`config_schema` Deprecated"
        The `config_schema` parameter is deprecated in v0.6.0 and support will be removed in v2.0.0.
        Please use `context_schema` instead to specify the schema for run-scoped context.


    Returns:
        A compiled LangChain runnable that can be used for chat interactions.

    The "agent" node calls the language model with the messages list (after applying the prompt).
    If the resulting AIMessage contains `tool_calls`, the graph will then call the ["tools"][langgraph.prebuilt.tool_node.ToolNode].
    The "tools" node executes the tools (1 tool per `tool_call`) and adds the responses to the messages list
    as `ToolMessage` objects. The agent node then calls the language model again.
    The process repeats until no more `tool_calls` are present in the response.
    The agent then returns the full list of messages as a dictionary containing the key "messages".

    ``` mermaid
        sequenceDiagram
            participant U as User
            participant A as LLM
            participant T as Tools
            U->>A: Initial input
            Note over A: Prompt + LLM
            loop while tool_calls present
                A->>T: Execute tools
                T-->>A: ToolMessage for each tool_calls
            end
            A->>U: Return final state
    ```

    Example:
        ```python
        from langgraph.prebuilt import create_react_agent

        def check_weather(location: str) -> str:
            '''Return the weather forecast for the specified location.'''
            return f"It's always sunny in {location}"

        graph = create_react_agent(
            "anthropic:claude-3-7-sonnet-latest",
            tools=[check_weather],
            prompt="You are a helpful assistant",
        )
        inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
        for chunk in graph.stream(inputs, stream_mode="updates"):
            print(chunk)
        ```
    """
    # Handle deprecated config_schema parameter
    if (
        config_schema := deprecated_kwargs.pop("config_schema", MISSING)
    ) is not MISSING:
        warn(
            "`config_schema` is deprecated and will be removed. Please use `context_schema` instead.",
            category=LangGraphDeprecatedSinceV10,
        )
        if context_schema is None:
            context_schema = config_schema

    if len(deprecated_kwargs) > 0:
        raise TypeError(
            f"create_react_agent() got unexpected keyword arguments: {deprecated_kwargs}"
        )

    if version not in ("v1", "v2"):
        raise ValueError(
            f"Invalid version {version}. Supported versions are 'v1' and 'v2'."
        )

    # Create and configure the agent builder
    builder = _AgentBuilder(
        model=model,
        tools=tools,
        prompt=prompt,
        response_format=response_format,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        state_schema=state_schema,
        context_schema=context_schema,
        version=version,
        name=name,
        store=store,
        use_individual_tool_nodes=use_individual_tool_nodes,
    )

    # Build and compile the workflow
    workflow = builder.build()
    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
    )


# Keep for backwards compatibility
create_tool_calling_executor = create_react_agent

__all__ = [
    "create_react_agent",
    "create_tool_calling_executor",
    "AgentState",
    "AgentStatePydantic",
    "AgentStateWithStructuredResponse",
    "AgentStateWithStructuredResponsePydantic",
]
