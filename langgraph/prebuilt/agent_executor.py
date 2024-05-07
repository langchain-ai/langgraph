import operator
from typing import Annotated, Sequence, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage

from langgraph._api.deprecation import deprecated
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.utils import RunnableCallable


def _get_agent_state(input_schema=None):
    if input_schema is None:

        class AgentState(TypedDict):
            # The input string
            input: str
            # The list of previous messages in the conversation
            chat_history: Sequence[BaseMessage]
            # The outcome of a given call to the agent
            # Needs `None` as a valid type, since this is what this will start as
            agent_outcome: Union[AgentAction, AgentFinish, None]
            # List of actions and corresponding observations
            # Here we annotate this with `operator.add` to indicate that operations to
            # this state should be ADDED to the existing values (not overwrite it)
            intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    else:

        class AgentState(input_schema):
            # The outcome of a given call to the agent
            # Needs `None` as a valid type, since this is what this will start as
            agent_outcome: Union[AgentAction, AgentFinish, None]
            # List of actions and corresponding observations
            # Here we annotate this with `operator.add` to indicate that operations to
            # this state should be ADDED to the existing values (not overwrite it)
            intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    return AgentState


@deprecated(
    "0.0.44",
    alternative="create_react_executor",
    example="""
from langgraph.prebuilt import create_react_executor 

create_react_executor(...)
""",
)
def create_agent_executor(
    agent_runnable, tools, input_schema=None
) -> CompiledStateGraph:
    """This is a helper function for creating a graph that works with LangChain Agents.

    Args:
        agent_runnable (RunnableLike): The agent runnable.
        tools (list): A list of tools to be used by the agent.
        input_schema (dict, optional): The input schema for the agent. Defaults to None.

    Returns:
        The `CompiledStateGraph` object.


    Examples:

        # Since this is deprecated, you should use `create_react_executor` instead.
        # Example usage:
        from langgraph.prebuilt import create_react_executor
        from langchain_openai import ChatOpenAI
        from langchain_community.tools.tavily_search import TavilySearchResults

        tools = [TavilySearchResults(max_results=1)]
        model = ChatOpenAI()

        app = create_react_executor(model, tools)

        inputs = {"messages": [("user", "what is the weather in sf")]}
        for s in app.stream(inputs):
            print(list(s.values())[0])
            print("----")
    """

    if isinstance(tools, ToolExecutor):
        tool_executor = tools
    else:
        tool_executor = ToolExecutor(tools)

    state = _get_agent_state(input_schema)

    # Define logic that will be used to determine which conditional edge to go down

    def should_continue(data):
        # If the agent outcome is an AgentFinish, then we return `exit` string
        # This will be used when setting up the graph to define the flow
        if isinstance(data["agent_outcome"], AgentFinish):
            return "end"
        # Otherwise, an AgentAction is returned
        # Here we return `continue` string
        # This will be used when setting up the graph to define the flow
        else:
            return "continue"

    def run_agent(data):
        agent_outcome = agent_runnable.invoke(data)
        return {"agent_outcome": agent_outcome}

    async def arun_agent(data):
        agent_outcome = await agent_runnable.ainvoke(data)
        return {"agent_outcome": agent_outcome}

    # Define the function to execute tools
    def execute_tools(data):
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        agent_action = data["agent_outcome"]
        if not isinstance(agent_action, list):
            agent_action = [agent_action]
        output = tool_executor.batch(agent_action, return_exceptions=True)
        return {
            "intermediate_steps": [
                (action, str(out)) for action, out in zip(agent_action, output)
            ]
        }

    async def aexecute_tools(data):
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        agent_action = data["agent_outcome"]
        if not isinstance(agent_action, list):
            agent_action = [agent_action]
        output = await tool_executor.abatch(agent_action, return_exceptions=True)
        return {
            "intermediate_steps": [
                (action, str(out)) for action, out in zip(agent_action, output)
            ]
        }

    # Define a new graph
    workflow = StateGraph(state)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", RunnableCallable(run_agent, arun_agent))
    workflow.add_node("action", RunnableCallable(execute_tools, aexecute_tools))

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
