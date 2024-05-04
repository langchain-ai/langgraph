import json
from typing import Annotated, Callable, Optional, Sequence, TypedDict, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, FunctionMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function

from langgraph._api.deprecation import deprecated
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.prebuilt.tool_node import ToolNode


# We create the AgentState that we will pass around
# This simply involves a list of messages
# We want steps to return messages to append to the list
# So we annotate the messages attribute with operator.add
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


@deprecated("0.0.44", "create_tool_calling_executor")
def create_function_calling_executor(
    model: LanguageModelLike, tools: Union[ToolExecutor, Sequence[BaseTool]]
) -> CompiledGraph:
    """Creates a graph that works with a chat model that utilizes function calling.

    Examples:

        # Since this is deprecated, you should use `create_tool_calling_executor` instead.
        # Example usage:
        from langgraph.prebuilt import chat_agent_executor
        from langchain_openai import ChatOpenAI
        from langchain_community.tools.tavily_search import TavilySearchResults

        tools = [TavilySearchResults(max_results=1)]
        model = ChatOpenAI()

        app = chat_agent_executor.create_tool_calling_executor(model, tools)

        inputs = {"messages": [("user", "what is the weather in sf")]}
        for s in app.stream(inputs):
            print(list(s.values())[0])
            print("----")
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
    def call_model(state: AgentState):
        messages = state["messages"]
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    async def acall_model(state: AgentState):
        messages = state["messages"]
        response = await model.ainvoke(messages)
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

    def call_tool(state: AgentState):
        action = _get_action(state)
        # We call the tool_executor and get back a response
        response = tool_executor.invoke(action)
        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(content=str(response), name=action.tool)
        # We return a list, because this will get added to the existing list
        return {"messages": [function_message]}

    async def acall_tool(state: AgentState):
        action = _get_action(state)
        # We call the tool_executor and get back a response
        response = await tool_executor.ainvoke(action)
        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(content=str(response), name=action.tool)
        # We return a list, because this will get added to the existing list
        return {"messages": [function_message]}

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", RunnableLambda(call_model, acall_model))
    workflow.add_node("action", RunnableLambda(call_tool, acall_tool))

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
            "continue": "action",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("action", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile()


def create_tool_calling_executor(
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
        model (LanguageModelLike): The chat model that supports OpenAI tool calling.
        tools (Union[ToolExecutor, Sequence[BaseTool]]): A list of tools or a ToolExecutor instance.
        messages_modifier: (Optional[Union[SystemMessage, str, Callable, Runnable]]): An optional
            messages modifier. This applies to messages BEFORE they are passed into the LLM.
            Can take a few different forms:
            - SystemMessage: this is added to the beginning of the list of messages.
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages.
            - Callable: This function should take in a list of messages and the output is then passed to the language model.
            - Runnable: This runnable should take in a list of messages and the output is then passed to the language model.
        checkpointer (Optional[BaseCheckpointSaver]): An optional checkpoint saver object.
        interrupt_before (Optional[Sequence[str]]): An optional list of node names to interrupt before.
        interrupt_after (Optional[Sequence[str]]): An optional list of node names to interrupt after.
        debug (bool): A flag indicating whether to enable debug mode.

    Returns:
        Runnable: A compiled LangChain runnable that can be used for chat interactions.

    Examples:

            from langgraph.prebuilt import chat_agent_executor
            from langchain_openai import ChatOpenAI
            from langchain_community.tools.tavily_search import TavilySearchResults

            tools = [TavilySearchResults(max_results=1)]
            model = ChatOpenAI()

            app = chat_agent_executor.create_tool_calling_executor(model, tools)

            inputs = {"messages": [("user", "what is the weather in sf")]}
            for s in app.stream(inputs):
                print(list(s.values())[0])
                print("----")
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
    def call_model(state: AgentState):
        messages = state["messages"]
        response = model_runnable.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    async def acall_model(state: AgentState):
        messages = state["messages"]
        response = await model_runnable.ainvoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", RunnableLambda(call_model, acall_model))
    workflow.add_node("action", ToolNode(tools))

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
            "continue": "action",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("action", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )
