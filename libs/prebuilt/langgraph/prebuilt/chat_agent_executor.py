import functools
import inspect
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
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
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

from langgraph.errors import ErrorCode, create_error_message
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send
from langgraph.utils.runnable import RunnableCallable, RunnableLike

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

    is_last_step: IsLastStep

    remaining_steps: RemainingSteps


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


def _convert_modifier_to_prompt(func: F) -> F:
    """Decorator that converts state_modifier kwarg to prompt kwarg."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        prompt = kwargs.get("prompt")
        state_modifier = kwargs.pop("state_modifier", None)
        if sum(p is not None for p in (prompt, state_modifier)) > 1:
            raise ValueError(
                "Expected only one of (prompt, state_modifier), got multiple values"
            )

        if state_modifier is not None:
            prompt = state_modifier

        kwargs["prompt"] = prompt
        return func(*args, **kwargs)

    return cast(F, wrapper)


def _should_bind_tools(model: LanguageModelLike, tools: Sequence[BaseTool]) -> bool:
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
    if len(tools) != len(bound_tools):
        raise ValueError(
            "Number of tools in the model.bind_tools() and tools passed to create_react_agent must match"
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


@_convert_modifier_to_prompt
def create_react_agent(
    model: Union[str, LanguageModelLike],
    tools: Union[Sequence[Union[BaseTool, Callable]], ToolNode],
    *,
    prompt: Optional[Prompt] = None,
    response_format: Optional[
        Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]
    ] = None,
    pre_model_hook: Optional[RunnableLike] = None,
    state_schema: Optional[StateSchemaType] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v1",
    name: Optional[str] = None,
) -> CompiledGraph:
    """Creates a graph that works with a chat model that utilizes tool calling.

    Args:
        model: The `LangChain` chat model that supports tool calling.
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
        state_schema: An optional state schema that defines graph state.
            Must have `messages` and `remaining_steps` keys.
            Defaults to `AgentState` that defines those two keys.
        config_schema: An optional schema for configuration.
            Use this to expose configurable parameters via agent.config_specs.
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

    Returns:
        A compiled LangChain runnable that can be used for chat interactions.

    The resulting graph looks like this:

    ``` mermaid
    stateDiagram-v2
        [*] --> Start
        Start --> Agent
        Agent --> Tools : continue
        Tools --> Agent
        Agent --> End : end
        End --> [*]

        classDef startClass fill:#ffdfba;
        classDef endClass fill:#baffc9;
        classDef otherClass fill:#fad7de;

        class Start startClass
        class End endClass
        class Agent,Tools otherClass
    ```

    The "agent" node calls the language model with the messages list (after applying the messages modifier).
    If the resulting AIMessage contains `tool_calls`, the graph will then call the ["tools"][langgraph.prebuilt.tool_node.ToolNode].
    The "tools" node executes the tools (1 tool per `tool_call`) and adds the responses to the messages list
    as `ToolMessage` objects. The agent node then calls the language model again.
    The process repeats until no more `tool_calls` are present in the response.
    The agent then returns the full list of messages as a dictionary containing the key "messages".

    ``` mermaid
        sequenceDiagram
            participant U as User
            participant A as Agent (LLM)
            participant T as Tools
            U->>A: Initial input
            Note over A: Messages modifier + LLM
            loop while tool_calls present
                A->>T: Execute tools
                T-->>A: ToolMessage for each tool_calls
            end
            A->>U: Return final state
    ```

    Examples:
        Use with a simple tool:

        ```pycon
        >>> from langchain_openai import ChatOpenAI
        >>> from langgraph.prebuilt import create_react_agent


        ... def check_weather(location: str) -> str:
        ...     '''Return the weather forecast for the specified location.'''
        ...     return f"It's always sunny in {location}"
        >>>
        >>> tools = [check_weather]
        >>> model = ChatOpenAI(model="gpt-4o")
        >>> graph = create_react_agent(model, tools=tools)
        >>> inputs = {"messages": [("user", "what is the weather in sf")]}
        >>> for s in graph.stream(inputs, stream_mode="values"):
        ...     message = s["messages"][-1]
        ...     if isinstance(message, tuple):
        ...         print(message)
        ...     else:
        ...         message.pretty_print()
        ('user', 'what is the weather in sf')
        ================================== Ai Message ==================================
        Tool Calls:
        check_weather (call_LUzFvKJRuaWQPeXvBOzwhQOu)
        Call ID: call_LUzFvKJRuaWQPeXvBOzwhQOu
        Args:
            location: San Francisco
        ================================= Tool Message =================================
        Name: check_weather
        It's always sunny in San Francisco
        ================================== Ai Message ==================================
        The weather in San Francisco is sunny.
        ```
        Add a system prompt for the LLM:

        ```pycon
        >>> system_prompt = "You are a helpful bot named Fred."
        >>> graph = create_react_agent(model, tools, prompt=system_prompt)
        >>> inputs = {"messages": [("user", "What's your name? And what's the weather in SF?")]}
        >>> for s in graph.stream(inputs, stream_mode="values"):
        ...     message = s["messages"][-1]
        ...     if isinstance(message, tuple):
        ...         print(message)
        ...     else:
        ...         message.pretty_print()
        ('user', "What's your name? And what's the weather in SF?")
        ================================== Ai Message ==================================
        Hi, my name is Fred. Let me check the weather in San Francisco for you.
        Tool Calls:
        check_weather (call_lqhj4O0hXYkW9eknB4S41EXk)
        Call ID: call_lqhj4O0hXYkW9eknB4S41EXk
        Args:
            location: San Francisco
        ================================= Tool Message =================================
        Name: check_weather
        It's always sunny in San Francisco
        ================================== Ai Message ==================================
        The weather in San Francisco is currently sunny. If you need any more details or have other questions, feel free to ask!
        ```

        Add a more complex prompt for the LLM:

        ```pycon
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> prompt = ChatPromptTemplate.from_messages([
        ...     ("system", "You are a helpful bot named Fred."),
        ...     ("placeholder", "{messages}"),
        ...     ("user", "Remember, always be polite!"),
        ... ])
        >>>
        >>> graph = create_react_agent(model, tools, prompt=prompt)
        >>> inputs = {"messages": [("user", "What's your name? And what's the weather in SF?")]}
        >>> for s in graph.stream(inputs, stream_mode="values"):
        ...     message = s["messages"][-1]
        ...     if isinstance(message, tuple):
        ...         print(message)
        ...     else:
        ...         message.pretty_print()
        ```

        Add complex prompt with custom graph state:

        ```pycon
        >>> from typing_extensions import TypedDict
        >>>
        >>> from langgraph.managed import IsLastStep
        >>> prompt = ChatPromptTemplate.from_messages(
        ...     [
        ...         ("system", "Today is {today}"),
        ...         ("placeholder", "{messages}"),
        ...     ]
        ... )
        >>>
        >>> class CustomState(TypedDict):
        ...     today: str
        ...     messages: Annotated[list[BaseMessage], add_messages]
        ...     is_last_step: IsLastStep
        >>>
        >>> graph = create_react_agent(model, tools, state_schema=CustomState, prompt=prompt)
        >>> inputs = {"messages": [("user", "What's today's date? And what's the weather in SF?")], "today": "July 16, 2004"}
        >>> for s in graph.stream(inputs, stream_mode="values"):
        ...     message = s["messages"][-1]
        ...     if isinstance(message, tuple):
        ...         print(message)
        ...     else:
        ...         message.pretty_print()
        ```

        Add thread-level "chat memory" to the graph:

        ```pycon
        >>> from langgraph.checkpoint.memory import MemorySaver
        >>> graph = create_react_agent(model, tools, checkpointer=MemorySaver())
        >>> config = {"configurable": {"thread_id": "thread-1"}}
        >>> def print_stream(graph, inputs, config):
        ...     for s in graph.stream(inputs, config, stream_mode="values"):
        ...         message = s["messages"][-1]
        ...         if isinstance(message, tuple):
        ...             print(message)
        ...         else:
        ...             message.pretty_print()
        >>> inputs = {"messages": [("user", "What's the weather in SF?")]}
        >>> print_stream(graph, inputs, config)
        >>> inputs2 = {"messages": [("user", "Cool, so then should i go biking today?")]}
        >>> print_stream(graph, inputs2, config)
        ('user', "What's the weather in SF?")
        ================================== Ai Message ==================================
        Tool Calls:
        check_weather (call_ChndaktJxpr6EMPEB5JfOFYc)
        Call ID: call_ChndaktJxpr6EMPEB5JfOFYc
        Args:
            location: San Francisco
        ================================= Tool Message =================================
        Name: check_weather
        It's always sunny in San Francisco
        ================================== Ai Message ==================================
        The weather in San Francisco is sunny. Enjoy your day!
        ================================ Human Message =================================
        Cool, so then should i go biking today?
        ================================== Ai Message ==================================
        Since the weather in San Francisco is sunny, it sounds like a great day for biking! Enjoy your ride!
        ```

        Add an interrupt to let the user confirm before taking an action:

        ```pycon
        >>> graph = create_react_agent(
        ...     model, tools, interrupt_before=["tools"], checkpointer=MemorySaver()
        >>> )
        >>> config = {"configurable": {"thread_id": "thread-1"}}

        >>> inputs = {"messages": [("user", "What's the weather in SF?")]}
        >>> print_stream(graph, inputs, config)
        >>> snapshot = graph.get_state(config)
        >>> print("Next step: ", snapshot.next)
        >>> print_stream(graph, None, config)
        ```

        Add cross-thread memory to the graph:

        ```pycon
        >>> from langgraph.prebuilt import InjectedStore
        >>> from langgraph.store.base import BaseStore

        >>> def save_memory(memory: str, *, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore()]) -> str:
        ...     '''Save the given memory for the current user.'''
        ...     # This is a **tool** the model can use to save memories to storage
        ...     user_id = config.get("configurable", {}).get("user_id")
        ...     namespace = ("memories", user_id)
        ...     store.put(namespace, f"memory_{len(store.search(namespace))}", {"data": memory})
        ...     return f"Saved memory: {memory}"

        >>> def prepare_model_inputs(state: AgentState, config: RunnableConfig, store: BaseStore):
        ...     # Retrieve user memories and add them to the system message
        ...     # This function is called **every time** the model is prompted. It converts the state to a prompt
        ...     user_id = config.get("configurable", {}).get("user_id")
        ...     namespace = ("memories", user_id)
        ...     memories = [m.value["data"] for m in store.search(namespace)]
        ...     system_msg = f"User memories: {', '.join(memories)}"
        ...     return [{"role": "system", "content": system_msg)] + state["messages"]

        >>> from langgraph.checkpoint.memory import MemorySaver
        >>> from langgraph.store.memory import InMemoryStore
        >>> store = InMemoryStore()
        >>> graph = create_react_agent(model, [save_memory], prompt=prepare_model_inputs, store=store, checkpointer=MemorySaver())
        >>> config = {"configurable": {"thread_id": "thread-1", "user_id": "1"}}

        >>> inputs = {"messages": [("user", "Hey I'm Will, how's it going?")]}
        >>> print_stream(graph, inputs, config)
        ('user', "Hey I'm Will, how's it going?")
        ================================== Ai Message ==================================
        Hello Will! It's nice to meet you. I'm doing well, thank you for asking. How are you doing today?

        >>> inputs2 = {"messages": [("user", "I like to bike")]}
        >>> print_stream(graph, inputs2, config)
        ================================ Human Message =================================
        I like to bike
        ================================== Ai Message ==================================
        That's great to hear, Will! Biking is an excellent hobby and form of exercise. It's a fun way to stay active and explore your surroundings. Do you have any favorite biking routes or trails you enjoy? Or perhaps you're into a specific type of biking, like mountain biking or road cycling?

        >>> config = {"configurable": {"thread_id": "thread-2", "user_id": "1"}}
        >>> inputs3 = {"messages": [("user", "Hi there! Remember me?")]}
        >>> print_stream(graph, inputs3, config)
        ================================ Human Message =================================
        Hi there! Remember me?
        ================================== Ai Message ==================================
        User memories:
        Hello! Of course, I remember you, Will! You mentioned earlier that you like to bike. It's great to hear from you again. How have you been? Have you been on any interesting bike rides lately?
        ```

        Add a timeout for a given step:

        ```pycon
        >>> import time
        ... def check_weather(location: str) -> str:
        ...     '''Return the weather forecast for the specified location.'''
        ...     time.sleep(2)
        ...     return f"It's always sunny in {location}"
        >>>
        >>> tools = [check_weather]
        >>> graph = create_react_agent(model, tools)
        >>> graph.step_timeout = 1 # Seconds
        >>> for s in graph.stream({"messages": [("user", "what is the weather in sf")]}):
        ...     print(s)
        TimeoutError: Timed out at step 2
        ```
    """
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

    if isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
        tool_node = tools
    else:
        tool_node = ToolNode(tools)
        # get the tool functions wrapped in a tool class from the ToolNode
        tool_classes = list(tool_node.tools_by_name.values())

    if isinstance(model, str):
        try:
            from langchain.chat_models import (  # type: ignore[import-not-found]
                init_chat_model,
            )
        except ImportError:
            raise ImportError(
                "Please install langchain (`pip install langchain`) to use '<provider>:<model>' string syntax for `model` parameter."
            )

        model = cast(BaseChatModel, init_chat_model(model))

    tool_calling_enabled = len(tool_classes) > 0

    if _should_bind_tools(model, tool_classes) and tool_calling_enabled:
        model = cast(BaseChatModel, model).bind_tools(tool_classes)

    model_runnable = _get_prompt_runnable(prompt) | model

    # If any of the tools are configured to return_directly after running,
    # our graph needs to check if these were called
    should_return_direct = {t.name for t in tool_classes if t.return_direct}

    def _are_more_steps_needed(state: StateSchema, response: BaseMessage) -> bool:
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        is_last_step = _get_state_value(state, "is_last_step", False)
        return (
            (remaining_steps is None and is_last_step and has_tool_calls)
            or (
                remaining_steps is not None
                and remaining_steps < 1
                and all_tools_return_direct
            )
            or (remaining_steps is not None and remaining_steps < 2 and has_tool_calls)
        )

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
    def call_model(state: StateSchema, config: RunnableConfig) -> StateSchema:
        state = _get_model_input_state(state)
        response = cast(AIMessage, model_runnable.invoke(state, config))
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

    async def acall_model(state: StateSchema, config: RunnableConfig) -> StateSchema:
        state = _get_model_input_state(state)
        response = cast(AIMessage, await model_runnable.ainvoke(state, config))
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
        state: StateSchema, config: RunnableConfig
    ) -> StateSchema:
        messages = _get_state_value(state, "messages")
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        model_with_structured_output = _get_model(model).with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema)
        )
        response = model_with_structured_output.invoke(messages, config)
        return {"structured_response": response}

    async def agenerate_structured_response(
        state: StateSchema, config: RunnableConfig
    ) -> StateSchema:
        messages = _get_state_value(state, "messages")
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        model_with_structured_output = _get_model(model).with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema)
        )
        response = await model_with_structured_output.ainvoke(messages, config)
        return {"structured_response": response}

    if not tool_calling_enabled:
        # Define a new graph
        workflow = StateGraph(state_schema, config_schema=config_schema)
        workflow.add_node(
            "agent",
            RunnableCallable(call_model, acall_model),
            input=input_schema,
        )
        if pre_model_hook is not None:
            workflow.add_node("pre_model_hook", pre_model_hook)
            workflow.add_edge("pre_model_hook", "agent")
            entrypoint = "pre_model_hook"
        else:
            entrypoint = "agent"

        workflow.set_entry_point(entrypoint)

        if response_format is not None:
            workflow.add_node(
                "generate_structured_response",
                RunnableCallable(
                    generate_structured_response, agenerate_structured_response
                ),
            )
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
    def should_continue(state: StateSchema) -> Union[str, list]:
        messages = _get_state_value(state, "messages")
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return END if response_format is None else "generate_structured_response"
        # Otherwise if there is, we continue
        else:
            if version == "v1":
                return "tools"
            elif version == "v2":
                tool_calls = [
                    tool_node.inject_tool_args(call, state, store)  # type: ignore[arg-type]
                    for call in last_message.tool_calls
                ]
                return [Send("tools", [tool_call]) for tool_call in tool_calls]

    # Define a new graph
    workflow = StateGraph(state_schema or AgentState, config_schema=config_schema)

    # Define the two nodes we will cycle between
    workflow.add_node(
        "agent", RunnableCallable(call_model, acall_model), input=input_schema
    )
    workflow.add_node("tools", tool_node)

    # Optionally add a pre-model hook node that will be called
    # every time before the "agent" (LLM-calling node)
    if pre_model_hook is not None:
        workflow.add_node("pre_model_hook", pre_model_hook)
        workflow.add_edge("pre_model_hook", "agent")
        entrypoint = "pre_model_hook"
    else:
        entrypoint = "agent"

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point(entrypoint)

    # Add a structured output node if response_format is provided
    if response_format is not None:
        workflow.add_node(
            "generate_structured_response",
            RunnableCallable(
                generate_structured_response, agenerate_structured_response
            ),
        )
        workflow.add_edge("generate_structured_response", END)
        should_continue_destinations = ["tools", "generate_structured_response"]
    else:
        should_continue_destinations = ["tools", END]

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        path_map=should_continue_destinations,
    )

    def route_tool_responses(state: StateSchema) -> str:
        for m in reversed(_get_state_value(state, "messages")):
            if not isinstance(m, ToolMessage):
                break
            if m.name in should_return_direct:
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
