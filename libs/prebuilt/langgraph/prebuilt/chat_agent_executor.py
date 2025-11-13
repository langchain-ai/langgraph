import inspect
import warnings
from collections.abc import Awaitable, Callable, Sequence
from typing import (
    Annotated,
    Any,
    Literal,
    TypeVar,
    cast,
    get_type_hints,
)

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
from langgraph._internal._runnable import RunnableCallable, RunnableLike
from langgraph._internal._typing import MISSING
from langgraph.errors import ErrorCode, create_error_message
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send
from langgraph.typing import ContextT
from langgraph.warnings import LangGraphDeprecatedSinceV10
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict, deprecated

from langgraph.prebuilt.tool_node import ToolCallWithContext, ToolNode

StructuredResponse = dict | BaseModel
StructuredResponseSchema = dict | type[BaseModel]


@deprecated(
    "AgentState has been moved to `langchain.agents`. Please update your import to `from langchain.agents import AgentState`.",
    category=LangGraphDeprecatedSinceV10,
)
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    remaining_steps: NotRequired[RemainingSteps]


@deprecated(
    "AgentStatePydantic has been deprecated in favor of AgentState in `langchain.agents`.",
    category=LangGraphDeprecatedSinceV10,
)
class AgentStatePydantic(BaseModel):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    remaining_steps: RemainingSteps = 25


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=LangGraphDeprecatedSinceV10,
        message="AgentState has been moved to `langchain.agents`.*",
    )

    @deprecated(
        "AgentStateWithStructuredResponse has been deprecated in favor of AgentState in `langchain.agents`.",
        category=LangGraphDeprecatedSinceV10,
    )
    class AgentStateWithStructuredResponse(AgentState):
        """The state of the agent with a structured response."""

        structured_response: StructuredResponse


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=LangGraphDeprecatedSinceV10,
        message="AgentStatePydantic has been deprecated in favor of AgentState in `langchain.agents`.",
    )

    @deprecated(
        "AgentStateWithStructuredResponsePydantic has been deprecated in favor of AgentState in `langchain.agents`.",
        category=LangGraphDeprecatedSinceV10,
    )
    class AgentStateWithStructuredResponsePydantic(AgentStatePydantic):
        """The state of the agent with a structured response."""

        structured_response: StructuredResponse


StateSchema = TypeVar("StateSchema", bound=AgentState | AgentStatePydantic)
StateSchemaType = type[StateSchema]

PROMPT_RUNNABLE_NAME = "Prompt"

Prompt = (
    SystemMessage
    | str
    | Callable[[StateSchema], LanguageModelInput]
    | Runnable[StateSchema, LanguageModelInput]
)


def _get_state_value(state: StateSchema, key: str, default: Any = None) -> Any:
    return (
        state.get(key, default)
        if isinstance(state, dict)
        else getattr(state, key, default)
    )


def _get_prompt_runnable(prompt: Prompt | None) -> Runnable:
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


def _should_bind_tools(
    model: LanguageModelLike, tools: Sequence[BaseTool], num_builtin: int = 0
) -> bool:
    if isinstance(model, RunnableSequence):
        model = next(
            (
                step
                for step in model.steps
                if isinstance(step, (RunnableBinding, BaseChatModel))
            ),
            model,
        )

    if not isinstance(model, RunnableBinding):
        return True

    if "tools" not in model.kwargs:
        return True

    bound_tools = model.kwargs["tools"]
    if len(tools) != len(bound_tools) - num_builtin:
        raise ValueError(
            "Number of tools in the model.bind_tools() and tools passed to create_react_agent must match"
            f" Got {len(tools)} tools, expected {len(bound_tools) - num_builtin}"
        )

    tool_names = set(tool.name for tool in tools)
    bound_tool_names = set()
    for bound_tool in bound_tools:
        # OpenAI-style tool
        if bound_tool.get("type") == "function":
            bound_tool_name = bound_tool["function"]["name"]
        # Anthropic-style tool
        elif bound_tool.get("name"):
            bound_tool_name = bound_tool["name"]
        else:
            # unknown tool type so we'll ignore it
            continue

        bound_tool_names.add(bound_tool_name)

    if missing_tools := tool_names - bound_tool_names:
        raise ValueError(f"Missing tools '{missing_tools}' in the model.bind_tools()")

    return False


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


@deprecated(
    "create_react_agent has been moved to `langchain.agents`. Please update your import to `from langchain.agents import create_agent`.",
    category=LangGraphDeprecatedSinceV10,
)
def create_react_agent(
    model: str
    | LanguageModelLike
    | Callable[[StateSchema, Runtime[ContextT]], BaseChatModel]
    | Callable[[StateSchema, Runtime[ContextT]], Awaitable[BaseChatModel]]
    | Callable[
        [StateSchema, Runtime[ContextT]], Runnable[LanguageModelInput, BaseMessage]
    ]
    | Callable[
        [StateSchema, Runtime[ContextT]],
        Awaitable[Runnable[LanguageModelInput, BaseMessage]],
    ],
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | ToolNode,
    *,
    prompt: Prompt | None = None,
    response_format: StructuredResponseSchema
    | tuple[str, StructuredResponseSchema]
    | None = None,
    pre_model_hook: RunnableLike | None = None,
    post_model_hook: RunnableLike | None = None,
    state_schema: StateSchemaType | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v2",
    name: str | None = None,
    **deprecated_kwargs: Any,
) -> CompiledStateGraph:
    """Creates an agent graph that calls tools in a loop until a stopping condition is met.

    For more details on using `create_react_agent`, visit [Agents](https://langchain-ai.github.io/langgraph/agents/overview/) documentation.

    Args:
        model: The language model for the agent. Supports static and dynamic
            model selection.

            - **Static model**: A chat model instance (e.g.,
                [`ChatOpenAI`][langchain_openai.ChatOpenAI]) or string identifier (e.g.,
                `"openai:gpt-4"`)
            - **Dynamic model**: A callable with signature
                `(state, runtime) -> BaseChatModel` that returns different models
                based on runtime context

                If the model has tools bound via `bind_tools` or other configurations,
                the return type should be a `Runnable[LanguageModelInput, BaseMessage]`
                Coroutines are also supported, allowing for asynchronous model selection.

            Dynamic functions receive graph state and runtime, enabling
            context-dependent model selection. Must return a `BaseChatModel`
            instance. For tool calling, bind tools using `.bind_tools()`.
            Bound tools must be a subset of the `tools` parameter.

            !!! example "Dynamic model"

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

        tools: A list of tools or a `ToolNode` instance.
            If an empty list is provided, the agent will consist of a single LLM node without tool calling.
        prompt: An optional prompt for the LLM. Can take a few different forms:

            - `str`: This is converted to a `SystemMessage` and added to the beginning of the list of messages in `state["messages"]`.
            - `SystemMessage`: this is added to the beginning of the list of messages in `state["messages"]`.
            - `Callable`: This function should take in full graph state and the output is then passed to the language model.
            - `Runnable`: This runnable should take in full graph state and the output is then passed to the language model.

        response_format: An optional schema for the final agent output.

            If provided, output will be formatted to match the given schema and returned in the 'structured_response' state key.

            If not provided, `structured_response` will not be present in the output state.

            Can be passed in as:

            - An OpenAI function/tool schema,
            - A JSON Schema,
            - A TypedDict class,
            - A Pydantic class.
            - A tuple `(prompt, schema)`, where schema is one of the above.
                The prompt will be used together with the model that is being used to
                generate the structured response.

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
            !!! Note
                `remaining_steps` is used to limit the number of steps the react agent can take.
                Calculated roughly as `recursion_limit` - `total_steps_taken`.
                If `remaining_steps` is less than 2 and tool calls are present in the response,
                the react agent will return a final AI Message with
                the content "Sorry, need more steps to process this request.".
                No `GraphRecusionError` will be raised in this case.

        context_schema: An optional schema for runtime context.
        checkpointer: An optional checkpoint saver object. This is used for persisting
            the state of the graph (e.g., as chat memory) for a single thread (e.g., a single conversation).
        store: An optional store object. This is used for persisting data
            across multiple threads (e.g., multiple conversations / users).
        interrupt_before: An optional list of node names to interrupt before.
            Should be one of the following: `"agent"`, `"tools"`.

            This is useful if you want to add a user confirmation or other interrupt before taking an action.
        interrupt_after: An optional list of node names to interrupt after.
            Should be one of the following: `"agent"`, `"tools"`.

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
        name: An optional name for the `CompiledStateGraph`.
            This name will be automatically used when adding ReAct agent graph to another graph as a subgraph node -
            particularly useful for building multi-agent systems.

    !!! warning "`config_schema` Deprecated"
        The `config_schema` parameter is deprecated in v0.6.0 and support will be removed in v2.0.0.
        Please use `context_schema` instead to specify the schema for run-scoped context.


    Returns:
        A compiled LangChain `Runnable` that can be used for chat interactions.

    The "agent" node calls the language model with the messages list (after applying the prompt).
    If the resulting AIMessage contains `tool_calls`, the graph will then call the ["tools"][langgraph.prebuilt.tool_node.ToolNode].
    The "tools" node executes the tools (1 tool per `tool_call`) and adds the responses to the messages list
    as `ToolMessage` objects. The agent node then calls the language model again.
    The process repeats until no more `tool_calls` are present in the response.
    The agent then returns the full list of messages as a dictionary containing the key `'messages'`.

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
    if (
        config_schema := deprecated_kwargs.pop("config_schema", MISSING)
    ) is not MISSING:
        warnings.warn(
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

    if state_schema is not None:
        required_keys = {"messages", "remaining_steps"}
        if response_format is not None:
            required_keys.add("structured_response")

        schema_keys = set(get_type_hints(state_schema))
        if missing_keys := required_keys - set(schema_keys):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

    if state_schema is None:
        state_schema = (
            AgentStateWithStructuredResponse
            if response_format is not None
            else AgentState
        )

    llm_builtin_tools: list[dict] = []
    if isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
        tool_node = tools
    else:
        llm_builtin_tools = [t for t in tools if isinstance(t, dict)]
        tool_node = ToolNode([t for t in tools if not isinstance(t, dict)])
        tool_classes = list(tool_node.tools_by_name.values())

    is_dynamic_model = not isinstance(model, (str, Runnable)) and callable(model)
    is_async_dynamic_model = is_dynamic_model and inspect.iscoroutinefunction(model)

    tool_calling_enabled = len(tool_classes) > 0

    if not is_dynamic_model:
        if isinstance(model, str):
            try:
                from langchain.chat_models import (  # type: ignore[import-not-found]
                    init_chat_model,
                )
            except ImportError:
                raise ImportError(
                    "Please install langchain (`pip install langchain`) to "
                    "use '<provider>:<model>' string syntax for `model` parameter."
                )

            model = cast(BaseChatModel, init_chat_model(model))

        if (
            _should_bind_tools(model, tool_classes, num_builtin=len(llm_builtin_tools))  # type: ignore[arg-type]
            and len(tool_classes + llm_builtin_tools) > 0
        ):
            model = cast(BaseChatModel, model).bind_tools(
                tool_classes + llm_builtin_tools  # type: ignore[operator]
            )

        static_model: Runnable | None = _get_prompt_runnable(prompt) | model  # type: ignore[operator]
    else:
        # For dynamic models, we'll create the runnable at runtime
        static_model = None

    # If any of the tools are configured to return_directly after running,
    # our graph needs to check if these were called
    should_return_direct = {t.name for t in tool_classes if t.return_direct}

    def _resolve_model(
        state: StateSchema, runtime: Runtime[ContextT]
    ) -> LanguageModelLike:
        """Resolve the model to use, handling both static and dynamic models."""
        if is_dynamic_model:
            return _get_prompt_runnable(prompt) | model(state, runtime)  # type: ignore[operator]
        else:
            return static_model

    async def _aresolve_model(
        state: StateSchema, runtime: Runtime[ContextT]
    ) -> LanguageModelLike:
        """Async resolve the model to use, handling both static and dynamic models."""
        if is_async_dynamic_model:
            resolved_model = await model(state, runtime)  # type: ignore[misc,operator]
            return _get_prompt_runnable(prompt) | resolved_model
        elif is_dynamic_model:
            return _get_prompt_runnable(prompt) | model(state, runtime)  # type: ignore[operator]
        else:
            return static_model

    def _are_more_steps_needed(state: StateSchema, response: BaseMessage) -> bool:
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
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

    def _get_model_input_state(state: StateSchema) -> StateSchema:
        if pre_model_hook is not None:
            messages = (
                _get_state_value(state, "llm_input_messages")
            ) or _get_state_value(state, "messages")
            error_msg = f"Expected input to call_model to have 'llm_input_messages' or 'messages' key, but got {state}"
        else:
            messages = _get_state_value(state, "messages")
            error_msg = (
                f"Expected input to call_model to have 'messages' key, but got {state}"
            )

        if messages is None:
            raise ValueError(error_msg)

        _validate_chat_history(messages)
        # we're passing messages under `messages` key, as this is expected by the prompt
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            state.messages = messages  # type: ignore
        else:
            state["messages"] = messages  # type: ignore

        return state

    # Define the function that calls the model
    def call_model(
        state: StateSchema, runtime: Runtime[ContextT], config: RunnableConfig
    ) -> StateSchema:
        if is_async_dynamic_model:
            msg = (
                "Async model callable provided but agent invoked synchronously. "
                "Use agent.ainvoke() or agent.astream(), or "
                "provide a sync model callable."
            )
            raise RuntimeError(msg)

        model_input = _get_model_input_state(state)

        if is_dynamic_model:
            # Resolve dynamic model at runtime and apply prompt
            dynamic_model = _resolve_model(state, runtime)
            response = cast(AIMessage, dynamic_model.invoke(model_input, config))  # type: ignore[arg-type]
        else:
            response = cast(AIMessage, static_model.invoke(model_input, config))  # type: ignore[union-attr]

        # add agent name to the AIMessage
        response.name = name

        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    async def acall_model(
        state: StateSchema, runtime: Runtime[ContextT], config: RunnableConfig
    ) -> StateSchema:
        model_input = _get_model_input_state(state)

        if is_dynamic_model:
            # Resolve dynamic model at runtime and apply prompt
            # (supports both sync and async)
            dynamic_model = await _aresolve_model(state, runtime)
            response = cast(AIMessage, await dynamic_model.ainvoke(model_input, config))  # type: ignore[arg-type]
        else:
            response = cast(AIMessage, await static_model.ainvoke(model_input, config))  # type: ignore[union-attr]

        # add agent name to the AIMessage
        response.name = name
        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    input_schema: StateSchemaType
    if pre_model_hook is not None:
        # Dynamically create a schema that inherits from state_schema and adds 'llm_input_messages'
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            # For Pydantic schemas
            from pydantic import create_model

            input_schema = create_model(
                "CallModelInputSchema",
                llm_input_messages=(list[AnyMessage], ...),
                __base__=state_schema,
            )
        else:
            # For TypedDict schemas
            class CallModelInputSchema(state_schema):  # type: ignore
                llm_input_messages: list[AnyMessage]

            input_schema = CallModelInputSchema
    else:
        input_schema = state_schema

    def generate_structured_response(
        state: StateSchema, runtime: Runtime[ContextT], config: RunnableConfig
    ) -> StateSchema:
        if is_async_dynamic_model:
            msg = (
                "Async model callable provided but agent invoked synchronously. "
                "Use agent.ainvoke() or agent.astream(), or provide a sync model callable."
            )
            raise RuntimeError(msg)

        messages = _get_state_value(state, "messages")
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        resolved_model = _resolve_model(state, runtime)
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
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        resolved_model = await _aresolve_model(state, runtime)
        model_with_structured_output = _get_model(
            resolved_model
        ).with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema)
        )
        response = await model_with_structured_output.ainvoke(messages, config)
        return {"structured_response": response}

    if not tool_calling_enabled:
        # Define a new graph
        workflow = StateGraph(state_schema=state_schema, context_schema=context_schema)
        workflow.add_node(
            "agent",
            RunnableCallable(call_model, acall_model),
            input_schema=input_schema,
        )
        if pre_model_hook is not None:
            workflow.add_node("pre_model_hook", pre_model_hook)  # type: ignore[arg-type]
            workflow.add_edge("pre_model_hook", "agent")
            entrypoint = "pre_model_hook"
        else:
            entrypoint = "agent"

        workflow.set_entry_point(entrypoint)

        if post_model_hook is not None:
            workflow.add_node("post_model_hook", post_model_hook)  # type: ignore[arg-type]
            workflow.add_edge("agent", "post_model_hook")

        if response_format is not None:
            workflow.add_node(
                "generate_structured_response",
                RunnableCallable(
                    generate_structured_response,
                    agenerate_structured_response,
                ),
            )
            if post_model_hook is not None:
                workflow.add_edge("post_model_hook", "generate_structured_response")
            else:
                workflow.add_edge("agent", "generate_structured_response")

        return workflow.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name,
        )

    # Define the function that determines whether to continue or not
    def should_continue(state: StateSchema) -> str | list[Send]:
        messages = _get_state_value(state, "messages")
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            if post_model_hook is not None:
                return "post_model_hook"
            elif response_format is not None:
                return "generate_structured_response"
            else:
                return END
        # Otherwise if there is, we continue
        else:
            if version == "v1":
                return "tools"
            elif version == "v2":
                if post_model_hook is not None:
                    return "post_model_hook"
                return [
                    Send(
                        "tools",
                        ToolCallWithContext(
                            __type="tool_call_with_context",
                            tool_call=call,
                            state=state,
                        ),
                    )
                    for call in last_message.tool_calls
                ]

    # Define a new graph
    workflow = StateGraph(
        state_schema=state_schema or AgentState, context_schema=context_schema
    )

    # Define the two nodes we will cycle between
    workflow.add_node(
        "agent",
        RunnableCallable(call_model, acall_model),
        input_schema=input_schema,
    )
    workflow.add_node("tools", tool_node)

    # Optionally add a pre-model hook node that will be called
    # every time before the "agent" (LLM-calling node)
    if pre_model_hook is not None:
        workflow.add_node("pre_model_hook", pre_model_hook)  # type: ignore[arg-type]
        workflow.add_edge("pre_model_hook", "agent")
        entrypoint = "pre_model_hook"
    else:
        entrypoint = "agent"

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point(entrypoint)

    agent_paths = []
    post_model_hook_paths = [entrypoint, "tools"]

    # Add a post model hook node if post_model_hook is provided
    if post_model_hook is not None:
        workflow.add_node("post_model_hook", post_model_hook)  # type: ignore[arg-type]
        agent_paths.append("post_model_hook")
        workflow.add_edge("agent", "post_model_hook")
    else:
        agent_paths.append("tools")

    # Add a structured output node if response_format is provided
    if response_format is not None:
        workflow.add_node(
            "generate_structured_response",
            RunnableCallable(
                generate_structured_response,
                agenerate_structured_response,
            ),
        )
        if post_model_hook is not None:
            post_model_hook_paths.append("generate_structured_response")
        else:
            agent_paths.append("generate_structured_response")
    else:
        if post_model_hook is not None:
            post_model_hook_paths.append(END)
        else:
            agent_paths.append(END)

    if post_model_hook is not None:

        def post_model_hook_router(state: StateSchema) -> str | list[Send]:
            """Route to the next node after post_model_hook.

            Routes to one of:
            * "tools": if there are pending tool calls without a corresponding message.
            * "generate_structured_response": if no pending tool calls exist and response_format is specified.
            * END: if no pending tool calls exist and no response_format is specified.
            """

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
                return [
                    Send(
                        "tools",
                        ToolCallWithContext(
                            __type="tool_call_with_context",
                            tool_call=call,
                            state=state,
                        ),
                    )
                    for call in pending_tool_calls
                ]
            elif isinstance(messages[-1], ToolMessage):
                return entrypoint
            elif response_format is not None:
                return "generate_structured_response"
            else:
                return END

        workflow.add_conditional_edges(
            "post_model_hook",
            post_model_hook_router,
            path_map=post_model_hook_paths,
        )

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        path_map=agent_paths,
    )

    def route_tool_responses(state: StateSchema) -> str:
        for m in reversed(_get_state_value(state, "messages")):
            if not isinstance(m, ToolMessage):
                break
            if m.name in should_return_direct:
                return END

        # handle a case of parallel tool calls where
        # the tool w/ `return_direct` was executed in a different `Send`
        if isinstance(m, AIMessage) and m.tool_calls:
            if any(call["name"] in should_return_direct for call in m.tool_calls):
                return END

        return entrypoint

    if should_return_direct:
        workflow.add_conditional_edges(
            "tools", route_tool_responses, path_map=[entrypoint, END]
        )
    else:
        workflow.add_edge("tools", entrypoint)

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
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
