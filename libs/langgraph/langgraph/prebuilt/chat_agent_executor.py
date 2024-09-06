from typing import (
    Annotated,
    Callable,
    Optional,
    Sequence,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool

from langgraph._api.deprecation import deprecated_parameter
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt.tool_node import ToolNode


# We create the AgentState that we will pass around
# This simply involves a list of messages
# We want steps to return messages to append to the list
# So we annotate the messages attribute with operator.add
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    is_last_step: IsLastStep


StateSchema = TypeVar("StateSchema", bound=AgentState)
StateSchemaType = Type[StateSchema]

STATE_MODIFIER_RUNNABLE_NAME = "StateModifier"

MessagesModifier = Union[
    SystemMessage,
    str,
    Callable[[Sequence[BaseMessage]], Sequence[BaseMessage]],
    Runnable[Sequence[BaseMessage], Sequence[BaseMessage]],
]

StateModifier = Union[
    SystemMessage,
    str,
    Callable[[StateSchema], Sequence[BaseMessage]],
    Runnable[StateSchema, Sequence[BaseMessage]],
]


def _get_state_modifier_runnable(state_modifier: Optional[StateModifier]) -> Runnable:
    state_modifier_runnable: Runnable
    if state_modifier is None:
        state_modifier_runnable = RunnableLambda(
            lambda state: state["messages"], name=STATE_MODIFIER_RUNNABLE_NAME
        )
    elif isinstance(state_modifier, str):
        _system_message: BaseMessage = SystemMessage(content=state_modifier)
        state_modifier_runnable = RunnableLambda(
            lambda state: [_system_message] + state["messages"],
            name=STATE_MODIFIER_RUNNABLE_NAME,
        )
    elif isinstance(state_modifier, SystemMessage):
        state_modifier_runnable = RunnableLambda(
            lambda state: [state_modifier] + state["messages"],
            name=STATE_MODIFIER_RUNNABLE_NAME,
        )
    elif callable(state_modifier):
        state_modifier_runnable = RunnableLambda(
            state_modifier, name=STATE_MODIFIER_RUNNABLE_NAME
        )
    elif isinstance(state_modifier, Runnable):
        state_modifier_runnable = state_modifier
    else:
        raise ValueError(
            f"Got unexpected type for `state_modifier`: {type(state_modifier)}"
        )

    return state_modifier_runnable


def _convert_messages_modifier_to_state_modifier(
    messages_modifier: MessagesModifier,
) -> StateModifier:
    state_modifier: StateModifier
    if isinstance(messages_modifier, (str, SystemMessage)):
        return messages_modifier
    elif callable(messages_modifier):

        def state_modifier(state: AgentState) -> Sequence[BaseMessage]:
            return messages_modifier(state["messages"])

        return state_modifier
    elif isinstance(messages_modifier, Runnable):
        state_modifier = (lambda state: state["messages"]) | messages_modifier
        return state_modifier
    raise ValueError(
        f"Got unexpected type for `messages_modifier`: {type(messages_modifier)}"
    )


def _get_model_preprocessing_runnable(
    state_modifier: Optional[StateModifier],
    messages_modifier: Optional[MessagesModifier],
) -> Runnable:
    # Add the state or message modifier, if exists
    if state_modifier is not None and messages_modifier is not None:
        raise ValueError(
            "Expected value for either state_modifier or messages_modifier, got values for both"
        )

    if state_modifier is None and messages_modifier is not None:
        state_modifier = _convert_messages_modifier_to_state_modifier(messages_modifier)

    return _get_state_modifier_runnable(state_modifier)


def _should_bind_tools(model: LanguageModelLike, tools: Sequence[BaseTool]) -> bool:
    if not isinstance(model, RunnableBinding):
        return False

    if not "tools" in model.kwargs:
        return False

    bound_tools = model.kwargs["tools"]
    if len(tools) != len(bound_tools):
        raise ValueError("Number of tools in the model.bind_tools() and tools passed to create_react_agent must match")

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

    return True


@deprecated_parameter("messages_modifier", "0.1.9", "state_modifier", removal="0.3.0")
def create_react_agent(
    model: LanguageModelLike,
    tools: Union[ToolExecutor, Sequence[BaseTool], ToolNode],
    *,
    state_schema: Optional[StateSchemaType] = None,
    messages_modifier: Optional[MessagesModifier] = None,
    state_modifier: Optional[StateModifier] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    interrupt_before: Optional[Sequence[str]] = None,
    interrupt_after: Optional[Sequence[str]] = None,
    debug: bool = False,
) -> CompiledGraph:
    """Creates a graph that works with a chat model that utilizes tool calling.

    Args:
        model: The `LangChain` chat model that supports tool calling.
        tools: A list of tools, a ToolExecutor, or a ToolNode instance.
        state_schema: An optional state schema that defines graph state.
            Must have `messages` and `is_last_step` keys.
            Defaults to `AgentState` that defines those two keys.
        messages_modifier: An optional
            messages modifier. This applies to messages BEFORE they are passed into the LLM.

            Can take a few different forms:

            - SystemMessage: this is added to the beginning of the list of messages.
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages.
            - Callable: This function should take in a list of messages and the output is then passed to the language model.
            - Runnable: This runnable should take in a list of messages and the output is then passed to the language model.
            !!! Warning
                `messages_modifier` parameter is deprecated as of version 0.1.9 and will be removed in 0.2.0
        state_modifier: An optional
            state modifier. This takes full graph state BEFORE the LLM is called and prepares the input to LLM.

            Can take a few different forms:

            - SystemMessage: this is added to the beginning of the list of messages in state["messages"].
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages in state["messages"].
            - Callable: This function should take in full graph state and the output is then passed to the language model.
            - Runnable: This runnable should take in full graph state and the output is then passed to the language model.
        checkpointer: An optional checkpoint saver object. This is useful for persisting
            the state of the graph (e.g., as chat memory).
        interrupt_before: An optional list of node names to interrupt before.
            Should be one of the following: "agent", "tools".
            This is useful if you want to add a user confirmation or other interrupt before taking an action.
        interrupt_after: An optional list of node names to interrupt after.
            Should be one of the following: "agent", "tools".
            This is useful if you want to return directly or run additional processing on an output.
        debug: A flag indicating whether to enable debug mode.

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
    If the resulting AIMessage contains `tool_calls`, the graph will then call the ["tools"][toolnode].
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
        >>> from datetime import datetime
        >>> from langchain_core.tools import tool
        >>> from langchain_openai import ChatOpenAI
        >>> from langgraph.prebuilt import create_react_agent
        >>>
        >>> @tool
        ... def check_weather(location: str, at_time: datetime | None = None) -> float:
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
        >>> graph = create_react_agent(model, tools, state_modifier=system_prompt)
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
        >>> def modify_state_messages(state: AgentState):
        ...     # You can do more complex modifications here
        ...     return prompt.invoke({"messages": state["messages"]})
        >>>
        >>> graph = create_react_agent(model, tools, state_modifier=modify_state_messages)
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
        >>> from typing import TypedDict
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
        ...     is_last_step: str
        >>>
        >>> graph = create_react_agent(model, tools, state_schema=CustomState, state_modifier=prompt)
        >>> inputs = {"messages": [("user", "What's today's date? And what's the weather in SF?")], "today": "July 16, 2004"}
        >>> for s in graph.stream(inputs, stream_mode="values"):
        ...     message = s["messages"][-1]
        ...     if isinstance(message, tuple):
        ...         print(message)
        ...     else:
        ...         message.pretty_print()
        ```

        Add "chat memory" to the graph:

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
        >>> def print_stream(graph, inputs, config):
        ...     for s in graph.stream(inputs, config, stream_mode="values"):
        ...         message = s["messages"][-1]
        ...         if isinstance(message, tuple):
        ...             print(message)
        ...         else:
        ...             message.pretty_print()

        >>> inputs = {"messages": [("user", "What's the weather in SF?")]}
        >>> print_stream(graph, inputs, config)
        >>> snapshot = graph.get_state(config)
        >>> print("Next step: ", snapshot.next)
        >>> print_stream(graph, None, config)
        ```

        Add a timeout for a given step:

        ```pycon
        >>> import time
        >>> @tool
        ... def check_weather(location: str, at_time: datetime | None = None) -> float:
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

    if state_schema is not None:
        if missing_keys := {"messages", "is_last_step"} - set(
            state_schema.__annotations__
        ):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

    if isinstance(tools, ToolExecutor):
        tool_classes = tools.tools
        tool_node = ToolNode(tool_classes)
    elif isinstance(tools, ToolNode):
        tool_classes = tools.tools_by_name.values()
        tool_node = tools
    else:
        tool_classes = tools
        tool_node = ToolNode(tool_classes)

    if _should_bind_tools(model, tool_classes):
        model = model.bind_tools(tool_classes)

    # Define the function that determines whether to continue or not
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    preprocessor = _get_model_preprocessing_runnable(state_modifier, messages_modifier)
    model_runnable = preprocessor | model

    # Define the function that calls the model
    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)
        if state["is_last_step"] and response.tool_calls:
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

    async def acall_model(state: AgentState, config: RunnableConfig):
        response = await model_runnable.ainvoke(state, config)
        if state["is_last_step"] and response.tool_calls:
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

    # Define a new graph
    workflow = StateGraph(state_schema or AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", RunnableLambda(call_model, acall_model))
    workflow.add_node("tools", tool_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )


# Keep for backwards compatibility
create_tool_calling_executor = create_react_agent

__all__ = [
    "create_react_agent",
    "create_tool_calling_executor",
    "AgentState",
]
