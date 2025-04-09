# Handoffs

A common pattern in multi-agent interactions is **handoffs**, where one agent *hands off* control to another. Handoffs allow you to specify:

- **destination**: target agent to navigate to
- **payload**: information to pass to that agent

To implement handoffs with `create_react_agent`, you need to:

1. Create special handoff tools that can transfer control to a different agent

    ```python
    def transfer_to_bob():
        """Transfer to bob."""
        return Command(
            # name of the agent (node) to go to
            # highlight-next-line
            goto="bob",
            # data to send to the agent
            # highlight-next-line
            update={"my_state_key": "my_state_value"},
            # indicate to LangGraph that we need to navigate to
            # agent node in a parent graph
            # highlight-next-line
            update={"my_state_key": "my_state_value"},
            graph=Command.PARENT,
        )
    ```

1. Create individual agents that have access to handoff tools:

    ```python
    alice = create_react_agent(..., tools=[add, transfer_to_bob])
    bob = create_react_agent(..., tools=[multiply, transfer_to_alice])
    ```

1. Define a parent graph that contains individual agents as nodes:

    ```python
    from langgraph.graph import StateGraph, START, MessagesState
    multi_agent_graph = (
        StateGraph(MessagesState)
        .add_node(alice)
        .add_node(bob)
        ...
    )
    ```

Putting this together, here is how you can implement a simple multi-agent system with two agents - Alice and Bob:

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command

def make_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"

    @tool(name, description=description)
    def handoff_tool(
        # access individual agent's state
        # highlight-next-line
        state: Annotated[MessagesState, InjectedState],
        # highlight-next-line
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            # highlight-next-line
            goto=agent_name,
            # Take individual agent's messages and add them to the
            # parent, multi-agent graph, as part of the handoff.
            # The next agent will see the updated multi-agent graph state.
            # highlight-next-line
            update={"messages": state["messages"] + [tool_message]},
            # indicate to LangGraph that we need to navigate to 
            # agent node in a parent, multi-agent graph
            # highlight-next-line
            graph=Command.PARENT,
        )
    return handoff_tool

# Handoffs
transfer_to_alice = make_handoff_tool(
    agent_name="alice", 
    description="Use this to ask Alice, addition expert, for help."
)
transfer_to_bob = make_handoff_tool(
    agent_name="bob",
    description="Use this to ask Bob, multiplication expert, for help."
)

# Simple agent tools
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def multiply(a: int, b: int) -> int:
   """Multiply two numbers."""
   return a * b

# Define agents
alice = create_react_agent(
    model="claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[add, transfer_to_bob],
    prompt="You only now how to add, do not perform any other operations.",
    # highlight-next-line
    name="alice"
)
bob = create_react_agent(
    model="claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[multiply, transfer_to_alice],
    prompt="You only now how to multiply, do not perform any other operations.",
    # highlight-next-line
    name="bob"
)

# Define multi-agent graph
multi_agent_graph = (
    StateGraph(MessagesState)
    .add_node(alice)
    .add_node(bob)
    .add_edge(START, "alice")
    .compile()
)

# Tun the multi-agent graph
for chunk in multi_agent_graph.stream({"messages": "what's (3 x 7) + 15?"}):
    print(chunk)
    print("\n")
```

!!! Note
    This handoff implementation assumes that:

    - each agent receives overall message history (across all agents) in the multi-agent system as its input
    - each agent outputs its internal messages history to the overall message history of the multi-agent system