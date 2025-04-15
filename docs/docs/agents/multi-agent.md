# Multi-agent

A single agent might struggle if it needs to specialize in multiple domains or manage many tools. To tackle this, you can break your agent into smaller, independent agents and composing them into a [multi-agent system](../concepts/multi_agent.md).

In multi-agent systems, agents need to communicate between each other. They do so via [handoffs](#handoffs) — a primitive that describes which agent to hand control to and the payload to send to that agent.

Two of the most popular multi-agent architectures are:

- [supervisor](#supervisor) — individual agents are coordinated by a central supervisor agent. The supervisor controls all communication flow and task delegation, making decisions about which agent to invoke based on the current context and task requirements.
- [swarm](#swarm) — agents dynamically hand off control to one another based on their specializations. The system remembers which agent was last active, ensuring that on subsequent interactions, the conversation resumes with that agent.

## Supervisor

![Supervisor](./assets/supervisor.png)

Use [`langgraph-supervisor`](https://github.com/langchain-ai/langgraph-supervisor-py) library to create a supervisor multi-agent system:

```bash
pip install langgraph-supervisor
```

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
# highlight-next-line
from langgraph_supervisor import create_supervisor

alice = create_react_agent(
    model="openai:gpt-4o",
    tools=[add],
    prompt="You only know how to add, do not perform any other operations.",
    # highlight-next-line
    name="alice"
)

bob = create_react_agent(
    model="openai:gpt-4o",
    tools=[multiply],
    prompt="You only know how to multiply, do not perform any other operations.",
    # highlight-next-line
    name="bob"
)

supervisor = create_supervisor(
    agents=[alice, bob],
    model=ChatOpenAI(model="gpt-4o"),
    prompt="You manage Alice (knows how to add) and Bob (knows how to multiply). Assign work to them."
).compile()

for chunk in supervisor.stream({"messages": "what's (3 x 7) + 15?"}):
    print(chunk)
    print("\n")
```

## Swarm

![Swarm](./assets/swarm.png)

Use [`langgraph-swarm`](https://github.com/langchain-ai/langgraph-swarm-py) library to create a swarm multi-agent system:

```bash
pip install langgraph-swarm
```

```python
from langgraph.prebuilt import create_react_agent
# highlight-next-line
from langgraph_swarm import create_swarm, create_handoff_tool

alice = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[
        add,
        # highlight-next-line
        create_handoff_tool(
            agent_name="bob",
            description="Ask Bob, multiplication expert, for help."
        )
    ],
    prompt="You only know how to add, do not perform any other operations.",
    # highlight-next-line
    name="alice"
)

bob = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[
        multiply,
        # highlight-next-line
        create_handoff_tool(
            agent_name="alice",
            description="Ask Alice, addition expert, for help."
        )
    ],
    prompt="You only know how to multiply, do not perform any other operations.",
    # highlight-next-line
    name="bob"
)

swarm = create_swarm(
    agents=[alice, bob],
    default_active_agent="alice"
).compile()

for chunk in swarm.stream({"messages": "what's (3 x 7) + 15?"}):
    print(chunk)
    print("\n")
```

## Handoffs

A common pattern in multi-agent interactions is **handoffs**, where one agent *hands off* control to another. Handoffs allow you to specify:

- **destination**: target agent to navigate to
- **payload**: information to pass to that agent

This is used both by `langgraph-supervisor` (supervisor hands off to individual agents) and `langgraph-swarm` (an individual agent can hand off to other agents).

To implement handoffs with `create_react_agent`, you need to:

1. Create a special tool that can transfer control to a different agent

    ```python
    def transfer_to_bob():
        """Transfer to bob."""
        return Command(
            # name of the agent (node) to go to
            # highlight-next-line
            goto="bob",
            # data to send to the agent
            # highlight-next-line
            update={"messages": [...]},
            # indicate to LangGraph that we need to navigate to
            # agent node in a parent graph
            # highlight-next-line
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
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command

def create_handoff_tool(*, agent_name: str, description: str | None = None):
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
transfer_to_alice = create_handoff_tool(
    agent_name="alice",
    description="Ask Alice, addition expert, for help."
)
transfer_to_bob = create_handoff_tool(
    agent_name="bob",
    description="Ask Bob, multiplication expert, for help."
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
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[add, transfer_to_bob],
    prompt="You only know how to add, do not perform any other operations.",
    # highlight-next-line
    name="alice"
)
bob = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[multiply, transfer_to_alice],
    prompt="You only know how to multiply, do not perform any other operations.",
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