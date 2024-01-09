import asyncio
from pprint import pprint

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.agents import AgentFinish
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI

from langgraph.graph import END, Graph

tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

# Construct the OpenAI Functions agent
agent_runnable = create_openai_functions_agent(llm, tools, prompt)


# Define the agent
# Note that here, we are using `.assign` to add the output of the agent to the dictionary
# This dictionary will be returned from the node
# The reason we don't want to return just the result of `agent_runnable` from this node is
# that we want to continue passing around all the other inputs
agent = RunnablePassthrough.assign(agent_outcome=agent_runnable)


# Define the function to execute tools
def execute_tools(data):
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    agent_action = data.pop("agent_outcome")
    # Get the tool to use
    tool_to_use = {t.name: t for t in tools}[agent_action.tool]
    # Call that tool on the input
    observation = tool_to_use.invoke(agent_action.tool_input)
    # We now add in the action and the observation to the `intermediate_steps` list
    # This is the list of all previous actions taken and their output
    data["intermediate_steps"].append((agent_action, observation))
    return data


# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(data["agent_outcome"], AgentFinish):
        return "exit"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"


# Define the graph


workflow = Graph()

# Add the agent node, we give it name `agent` which we will use later
workflow.add_node("agent", agent)
# Add the tools node, we give it name `tools` which we will use later
workflow.add_node("tools", execute_tools)

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
        "exit": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
chain = workflow.compile()


def main():
    for output in chain.stream(
        {"input": "what is the weather in sf", "intermediate_steps": []}
    ):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            pprint(value)
        print("\n---\n")


async def amain():
    async for output in chain.astream_log(
        {"input": "what is the weather in sf", "intermediate_steps": []},
        include_types=["llm"],
    ):
        for op in output.ops:
            if op["path"] == "/streamed_output/-":
                # this is the output from .stream()
                ...
            elif op["path"].startswith("/logs/") and op["path"].endswith(
                "/streamed_output/-"
            ):
                # these are tokens from the LLM
                print(op["value"])


asyncio.run(amain())
