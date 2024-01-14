from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import FunctionMessage
from langchain_core.agents import AgentFinish, AgentAction
import json

from langchain.tools.render import format_tool_to_openai_function
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
import operator
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import StateGraph, END


def _get_tool_executor_and_functions(tools, response_format):
    if isinstance(tools, ToolExecutor):
        tool_executor = tools
        tool_classes = tools.tools
    else:
        tool_executor = ToolExecutor(tools)
        tool_classes = tools

    functions = [format_tool_to_openai_function(t) for t in tool_classes]
    if response_format is not None:
        functions.append(convert_pydantic_to_openai_function(response_format))
    return tool_executor, functions


def create_messages_executor(model, tools, response_format = None):
    tool_executor, functions = _get_tool_executor_and_functions(tools, response_format)
    model = model.bind_functions([format_tool_to_openai_function(t) for t in tools])

    # Define the function that determines whether to continue or not
    def should_continue(state):
        messages = state['messages']
        last_message = messages[-1]
        # If there is no function call, then we finish
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        # Otherwise if there is, we need to check what type of function call it is
        else:
            if response_format is None:
                return "continue"
            elif last_message.additional_kwargs["function_call"]["name"] == response_format.__name__:
                return "end"
            else:
                return "continue"

    # Define the function that calls the model
    def call_model(state):
        messages = state['messages']
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define the function to execute tools
    def call_tool(state):
        messages = state['messages']
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an AgentAction from the function_call
        action = AgentAction(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
            log="",
        )
        # We call the tool_executor and get back a response
        response = tool_executor.invoke(action)
        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(content=str(response), name=action.tool)
        # We return a list, because this will get added to the existing list
        return {"messages": [function_message]}

    # We create the AgentState that we will pass around
    # This simply involves a list of messages
    # We want steps to return messages to append to the list
    # So we annotate the messages attribute with operator.add
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)

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
            "end": END
        }
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge('action', 'agent')

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile()
