import operator
import time
from typing import Annotated, Optional, Sequence, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor


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
            # Maximum number of iterations
            max_iterations: Optional[int]
            # Iteration count, tracked to enable early stopping
            iteration_count: int
            # # Maximum wall time
            # max_execution_time: Optional[float]
            # # Start time
            # start_time: float

    else:

        class AgentState(input_schema):
            # The outcome of a given call to the agent
            # Needs `None` as a valid type, since this is what this will start as
            agent_outcome: Union[AgentAction, AgentFinish, None]
            # List of actions and corresponding observations
            # Here we annotate this with `operator.add` to indicate that operations to
            # this state should be ADDED to the existing values (not overwrite it)
            intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
            # Maximum number of iterations
            max_iterations: Optional[int]
            # Iteration count, tracked to enable early stopping
            iteration_count: int
            # # Maximum wall time
            # max_execution_time: Optional[float]
            # # Start time
            # start_time: float

    return AgentState


def _should_abort(data) -> bool:
    """Check if exceeding max iterations."""
    return (
        (
            data.get("max_iterations") is not None
            and data.get("iteration_count", 0) >= data["max_iterations"]
        )
        # or (
        #     data.get("max_execution_time") is not None
        #     and time.time() - data["start_time"] >= data["max_execution_time"]
        # )
    )


def create_agent_executor(agent_runnable, tools, input_schema=None):
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
        # If exceeding max iterations
        elif _should_abort(data):
            return "abort"
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
        iteration_count = data.get("iteration_count", 0) + 1
        agent_action = data["agent_outcome"]
        if isinstance(agent_action, list):
            output = tool_executor.batch(agent_action, return_exceptions=True)
            return {
                "intermediate_steps": [
                    (action, str(out)) for action, out in zip(agent_action, output)
                ],
                "iteration_count": iteration_count,
            }
        output = tool_executor.invoke(agent_action)
        return {
            "intermediate_steps": [(agent_action, str(output))],
            "iteration_count": iteration_count,
        }

    async def aexecute_tools(data):
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        iteration_count = data.get("iteration_count", 0) + 1
        agent_action = data["agent_outcome"]
        if isinstance(agent_action, list):
            output = await tool_executor.abatch(agent_action, return_exceptions=True)
            return {
                "intermediate_steps": [
                    (action, str(out)) for action, out in zip(agent_action, output)
                ],
                "iteration_count": iteration_count,
            }
        output = await tool_executor.ainvoke(agent_action)
        return {
            "intermediate_steps": [(agent_action, str(output))],
            "iteration_count": iteration_count,
        }

    def abort(data):
        return {
            "agent_outcome": AgentFinish(
                return_values={"output": "Agent stopped due to max iterations."},
                log="Agent stopped due to max iterations.",
            )
        }

    # Define a new graph
    workflow = StateGraph(state)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", RunnableLambda(run_agent, arun_agent))
    workflow.add_node("action", RunnableLambda(execute_tools, aexecute_tools))
    workflow.add_node("abort", RunnableLambda(abort))

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
            # If `abort`, then we call the abort node.
            "abort": "abort",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("action", "agent")

    # End if we abort due to exceeding iteration or time constraints.
    workflow.add_edge("abort", END)

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile()
