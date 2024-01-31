import json
import operator
import os
import sys
from pathlib import Path
from time import monotonic
from typing import Annotated, Sequence, TypedDict

from langchain.tools.render import format_tool_to_openai_function
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langsmith import Client

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.utils import load_variable, log

# Configure LangSmith
os.environ["OPENAI_API_KEY"] = load_variable("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = load_variable("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = load_variable("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = load_variable("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = load_variable("LANGCHAIN_API_KEY")

# Create the LangSmith client
client = Client()

# Load environment variables
TAVILY_API_KEY = load_variable("TAVILY_API_KEY")
LANGCHAIN_TRACING_V2 = load_variable("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = load_variable("LANGCHAIN_API_KEY")
MODEL_NAME = load_variable("MODEL_NAME")

# Create the Tavily search results tool
tools = [TavilySearchResults(max_results=1)]

# Create the tool executor
tool_executor = ToolExecutor(tools)

# Create the LLM model
model = ChatOpenAI(temperature=0, streaming=True)

# Bind the tool executor to the model
functions = [format_tool_to_openai_function(tool) for tool in tools]
model = model.bind_functions(functions)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
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
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}

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
app = workflow.compile()

# We can now call the runnable

# inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
# result = app.invoke(inputs)
# print(result)

log.info("Starting LM Graph processing...\n\n")
start_time = monotonic()

inputs = {"messages": [HumanMessage(content="when will dune 2 be released")]}
for output in app.stream(inputs):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")

# Print the run time
log.info("LM Graph processing finished.")
log.info(f"Run time: {monotonic() - start_time}")