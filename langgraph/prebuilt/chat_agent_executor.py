import json
from typing import Annotated, Callable, Optional, Sequence, TypedDict, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function

from langgraph._api.deprecation import deprecated
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.prebuilt.tool_node import ToolNode


# We create the AgentState that we will pass around
# This simply involves a list of messages
# We want steps to return messages to append to the list
# So we annotate the messages attribute with operator.add
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    is_last_step: IsLastStep


@deprecated("0.0.44", "create_react_agent")
def create_function_calling_executor(
    model: LanguageModelLike, tools: Union[ToolExecutor, Sequence[BaseTool]]
) -> CompiledGraph:
    """Creates a graph that works with a chat model that utilizes function calling.

    Examples:
        ```pycon
        >>> # Since this is deprecated, you should use `create_react_agent` instead.
        >>> # Example usage:
        >>> from langgraph.prebuilt import create_react_agent
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_community.tools.tavily_search import TavilySearchResults
        >>>
        >>> tools = [TavilySearchResults(max_results=1)]
        >>> model = ChatOpenAI()
        >>>
        >>> app = create_react_agent(model, tools)
        >>>
        >>> inputs = {"messages": [("user", "what is the weather in sf")]}
        >>> for s in app.stream(inputs):
        ...     print(list(s.values())[0])
        ...     print("----")
        ```
    """
    if isinstance(tools, ToolExecutor):
        tool_executor = tools
        tool_classes = tools.tools
    else:
        tool_executor = ToolExecutor(tools)
        tool_classes = tools
    model = model.bind(functions=[convert_to_openai_function(t) for t in tool_classes])

    # Define the function that determines whether to continue or not
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    # Define the function that calls the model
    def call_model(state: AgentState, config: RunnableConfig):
        messages = state["messages"]
        response = model.invoke(messages, config)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    async def acall_model(state: AgentState, config: RunnableConfig):
        messages = state["messages"]
        response = await model.ainvoke(messages, config)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define the function to execute tools
    def _get_action(state: AgentState):
        messages = state["messages"]
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an AgentAction from the function_call
        return ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(
                last_message.additional_kwargs["function_call"]["arguments"]
            ),
        )

    def call_tool(state: AgentState, config: RunnableConfig):
        action = _get_action(state)
        # We call the tool_executor and get back a response
        response = tool_executor.invoke(action, config)
        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(content=str(response), name=action.tool)
        # We return a list, because this will get added to the existing list
        return {"messages": [function_message]}

    async def acall_tool(state: AgentState, config: RunnableConfig):
        action = _get_action(state)
        # We call the tool_executor and get back a response
        response = await tool_executor.ainvoke(action, config)
        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(content=str(response), name=action.tool)
        # We return a list, because this will get added to the existing list
        return {"messages": [function_message]}

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", RunnableLambda(call_model, acall_model))
    workflow.add_node("tools", RunnableLambda(call_tool, acall_tool))

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
    return workflow.compile()


def create_react_agent(
    model: LanguageModelLike,
    tools: Union[ToolExecutor, Sequence[BaseTool]],
    messages_modifier: Optional[Union[SystemMessage, str, Callable, Runnable]] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    interrupt_before: Optional[Sequence[str]] = None,
    interrupt_after: Optional[Sequence[str]] = None,
    debug: bool = False,
) -> CompiledGraph:
    """Creates a graph that works with a chat model that utilizes tool calling.

    Args:
        model: The chat model that supports OpenAI tool calling.
        tools: A list of tools or a ToolExecutor instance.
        messages_modifier: An optional
            messages modifier. This applies to messages BEFORE they are passed into the LLM.
            Can take a few different forms:
            - SystemMessage: this is added to the beginning of the list of messages.
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages.
            - Callable: This function should take in a list of messages and the output is then passed to the language model.
            - Runnable: This runnable should take in a list of messages and the output is then passed to the language model.
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
        >>> graph = create_react_agent(model, tools, messages_modifier=system_prompt)
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
        >>> def modify_messages(messages: list):
        ...     # You can do more complex modifications here
        ...     return prompt.invoke({"messages": messages})
        >>>
        >>> app = create_react_agent(model, tools, messages_modifier=modify_messages)
        >>> inputs = {"messages": [("user", "What's your name? And what's the weather in SF?")]}
        >>> for s in graph.stream(inputs, stream_mode="values"):
        ...     message = s["messages"][-1]
        ...     if isinstance(message, tuple):
        ...         print(message)
        ...     else:
        ...         message.pretty_print()
        ```

        Add "chat memory" to the graph:

        ```pycon
        >>> from langgraph.checkpoint import MemorySaver
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

    if isinstance(tools, ToolExecutor):
        tool_classes = tools.tools
    else:
        tool_classes = tools
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

    # Add the message modifier, if exists
    if messages_modifier is None:
        model_runnable = model
    elif isinstance(messages_modifier, str):
        _system_message: BaseMessage = SystemMessage(content=messages_modifier)
        model_runnable = (lambda messages: [_system_message] + messages) | model
    elif isinstance(messages_modifier, SystemMessage):
        model_runnable = (lambda messages: [messages_modifier] + messages) | model
    elif isinstance(messages_modifier, (Callable, Runnable)):
        model_runnable = messages_modifier | model
    else:
        raise ValueError(
            f"Got unexpected type for `messages_modifier`: {type(messages_modifier)}"
        )

    # Define the function that calls the model
    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        messages = state["messages"]
        response = model_runnable.invoke(messages, config)
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
        messages = state["messages"]
        response = await model_runnable.ainvoke(messages, config)
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
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", RunnableLambda(call_model, acall_model))
    workflow.add_node("tools", ToolNode(tools))

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
    "create_function_calling_executor",
    "AgentState",
]
