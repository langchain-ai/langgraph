from typing import Annotated, TypedDict
import operator
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutor




def create_agent_executor(agent_runnable, tools, input_schema=None):

    if isinstance(tools, ToolExecutor):
        tool_executor = tools
    else:
        tool_executor = ToolExecutor(tools)


    if input_schema is None:
        class AgentState(TypedDict):
            input: str
            agent_outcome: AgentAction | AgentFinish | None
            intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    else:
        class AgentState(input_schema):
            agent_outcome: AgentAction | AgentFinish | None
            intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    def should_continue(data):
        # If the agent outcome is an AgentFinish, then we return `exit` string
        # This will be used when setting up the graph to define the flow
        if isinstance(data['agent_outcome'], AgentFinish):
            return "end"
        # Otherwise, an AgentAction is returned
        # Here we return `continue` string
        # This will be used when setting up the graph to define the flow
        else:
            return "continue"

    def run_agent(data):
        agent_outcome = agent_runnable.invoke(data)
        return {"agent_outcome": agent_outcome}

    # Define the function to execute tools
    def execute_tools(data):
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        agent_action = data['agent_outcome']
        output = tool_executor.invoke(agent_action)
        return {"intermediate_steps": [(agent_action, str(output))]}

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", run_agent)
    workflow.add_node("action", execute_tools)

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
