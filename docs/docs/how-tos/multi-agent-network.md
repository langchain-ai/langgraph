# How to build a multi-agent network

!!! info "Prerequisites" 
    This guide assumes familiarity with the following:

    - [How to implement handoffs between agents](../agent-handoffs.md)
    - [Multi-agent systems](../../concepts/multi_agent.md)
    - [Command](../../concepts/low_level/#command.md)
    - [LangGraph Glossary](../../concepts/low_level/.md)

In this how-to guide we will demonstrate how to implement a [multi-agent network](../../concepts/multi_agent#network.md) architecture where each agent can communicate with every other agent (many-to-many connections) and can decide which agent to call next. Individual agents will be defined as graph nodes.

To implement communication between the agents, we will be using [handoffs](../agent-handoffs.md):

```python
def agent(state) -> Command[Literal["agent", "another_agent"]]:
    # the condition for routing/halting can be anything, e.g. LLM tool call / structured output, etc.
    goto = get_next_agent(...)  # 'agent' / 'another_agent'
    return Command(
        # Specify which agent to call next
        goto=goto,
        # Update the graph state
        update={"my_state_key": "my_state_value"}
    )
```

## Setup

First, let's install the required packages


```
%%capture --no-stderr
%pip install -U langgraph langchain-anthropic
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Using a custom agent implementation

In this example we will build a team of travel assistant agents that can communicate with each other via handoffs.

We will create 2 agents:

* `travel_advisor`: can help with travel destination recommendations. Can ask `hotel_advisor` for help.
* `hotel_advisor`: can help with hotel recommendations. Can ask `travel_advisor` for help.

This is a fully-connected network - every agent can talk to any other agent. 

Each agent will have a corresponding node function that can conditionally return a `Command` object (the handoff). The node function will use an LLM with a system prompt and a tool that lets it signal when it needs to hand off to another agent. If the LLM responds with the tool calls, we will return a `Command(goto=<other_agent>)`.

> **Note**: while we're using tools for the LLM to signal that it needs a handoff, the condition for the handoff can be anything: a specific response text from the LLM, structured output from the LLM, any other custom logic, etc.

Now, let's define our agent nodes and graph!


```python
from typing_extensions import Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command


model = ChatAnthropic(model="claude-3-5-sonnet-latest")


# Define a helper for each of the agent nodes to call


@tool
def transfer_to_travel_advisor():
    """Ask travel advisor for help."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to another agent
    # (See the paragraph above)
    return


@tool
def transfer_to_hotel_advisor():
    """Ask hotel advisor for help."""
    return


def travel_advisor(
    state: MessagesState,
) -> Command[Literal["hotel_advisor", "__end__"]]:
    system_prompt = (
        "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). "
        "If you need hotel recommendations, ask 'hotel_advisor' for help."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_hotel_advisor]).invoke(messages)
    # If there are tool calls, the LLM needs to hand off to another agent
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        # NOTE: it's important to insert a tool message here because LLM providers are expecting
        # all AI messages to be followed by a corresponding tool result message
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="hotel_advisor", update={"messages": [ai_msg, tool_msg]})

    # If the expert has an answer, return it directly to the user
    return {"messages": [ai_msg]}


def hotel_advisor(
    state: MessagesState,
) -> Command[Literal["travel_advisor", "__end__"]]:
    system_prompt = (
        "You are a hotel expert that can provide hotel recommendations for a given destination. "
        "If you need help picking travel destinations, ask 'travel_advisor' for help."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_travel_advisor]).invoke(messages)
    # If there are tool calls, the LLM needs to hand off to another agent
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        # NOTE: it's important to insert a tool message here because LLM providers are expecting
        # all AI messages to be followed by a corresponding tool result message
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="travel_advisor", update={"messages": [ai_msg, tool_msg]})

    # If the expert has an answer, return it directly to the user
    return {"messages": [ai_msg]}


builder = StateGraph(MessagesState)
builder.add_node("travel_advisor", travel_advisor)
builder.add_node("hotel_advisor", hotel_advisor)
# we'll always start with a general travel advisor
builder.add_edge(START, "travel_advisor")

graph = builder.compile()

from IPython.display import display, Image

display(Image(graph.get_graph().draw_mermaid_png()))
```

![](data:image/png;base64,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)

First, let's invoke it with a generic input:


```python
from langchain_core.messages import convert_to_messages


def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")

    for node_name, node_update in update.items():
        print(f"Update from node {node_name}:")
        print("\n")

        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()
        print("\n")
```


```python
for chunk in graph.stream(
    {"messages": [("user", "i wanna go somewhere warm in the caribbean")]}
):
    pretty_print_messages(chunk)
```

You can see that in this case only the first agent (`travel_advisor`) ran. Let's now ask for more recommendations:


```python
for chunk in graph.stream(
    {
        "messages": [
            (
                "user",
                "i wanna go somewhere warm in the caribbean. pick one destination and give me hotel recommendations",
            )
        ]
    }
):
    pretty_print_messages(chunk)
```

Voila - `travel_advisor` picks a destination and then makes a decision to call `hotel_advisor` for more info!

## Using with a prebuilt ReAct agent

Let's now see how we can implement the same team of travel agents, but give each of the agents some tools to call. We'll be using prebuilt [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent] to implement the agents. First, let's create some of the tools that the agents will be using:


```python
import random
from typing_extensions import Literal


@tool
def get_travel_recommendations():
    """Get recommendation for travel destinations"""
    return random.choice(["aruba", "turks and caicos"])


@tool
def get_hotel_recommendations(location: Literal["aruba", "turks and caicos"]):
    """Get hotel recommendations for a given destination."""
    return {
        "aruba": [
            "The Ritz-Carlton, Aruba (Palm Beach)"
            "Bucuti & Tara Beach Resort (Eagle Beach)"
        ],
        "turks and caicos": ["Grace Bay Club", "COMO Parrot Cay"],
    }[location]
```

Let's also write a helper to create a handoff tool. See [this how-to guide](../agent-handoffs#implementing-handoffs-using-tools.md) for a more in-depth walkthrough of how to make a handoff tool.


```python
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState


def make_handoff_tool(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            # navigate to another agent node in the PARENT graph
            goto=agent_name,
            graph=Command.PARENT,
            # This is the state update that the agent `agent_name` will see when it is invoked.
            # We're passing agent's FULL internal message history AND adding a tool message to make sure
            # the resulting chat history is valid.
            update={"messages": state["messages"] + [tool_message]},
        )

    return handoff_to_agent
```

Now let's define our agent nodes and combine them into a graph:


```python
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command


model = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Define travel advisor ReAct agent
travel_advisor_tools = [
    get_travel_recommendations,
    make_handoff_tool(agent_name="hotel_advisor"),
]
travel_advisor = create_react_agent(
    model,
    travel_advisor_tools,
    prompt=(
        "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). "
        "If you need hotel recommendations, ask 'hotel_advisor' for help. "
        "You MUST include human-readable response before transferring to another agent."
    ),
)


def call_travel_advisor(
    state: MessagesState,
) -> Command[Literal["hotel_advisor", "__end__"]]:
    # You can also add additional logic like changing the input to the agent / output from the agent, etc.
    # NOTE: we're invoking the ReAct agent with the full history of messages in the state
    return travel_advisor.invoke(state)


# Define hotel advisor ReAct agent
hotel_advisor_tools = [
    get_hotel_recommendations,
    make_handoff_tool(agent_name="travel_advisor"),
]
hotel_advisor = create_react_agent(
    model,
    hotel_advisor_tools,
    prompt=(
        "You are a hotel expert that can provide hotel recommendations for a given destination. "
        "If you need help picking travel destinations, ask 'travel_advisor' for help."
        "You MUST include human-readable response before transferring to another agent."
    ),
)


def call_hotel_advisor(
    state: MessagesState,
) -> Command[Literal["travel_advisor", "__end__"]]:
    return hotel_advisor.invoke(state)


builder = StateGraph(MessagesState)
builder.add_node("travel_advisor", call_travel_advisor)
builder.add_node("hotel_advisor", call_hotel_advisor)
# we'll always start with a general travel advisor
builder.add_edge(START, "travel_advisor")

graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
```

![](data:image/png;base64,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)

Let's test it out using the same input as our original multi-agent system:


```python
for chunk in graph.stream(
    {
        "messages": [
            (
                "user",
                "i wanna go somewhere warm in the caribbean. pick one destination and give me hotel recommendations",
            )
        ]
    },
    subgraphs=True,
):
    pretty_print_messages(chunk)
```
