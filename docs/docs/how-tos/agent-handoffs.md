# How to implement handoffs between agents

!!! info "Prerequisites"
    This guide assumes familiarity with the following:

    - [Multi-agent systems](../../concepts/multi_agent.md)
    - [Command](../../concepts/low_level/#command.md)
    - [LangGraph Glossary](../../concepts/low_level/.md)
    

In multi-agent architectures, agents can be represented as graph nodes. Each agent node executes its step(s) and decides whether to finish execution or route to another agent, including potentially routing to itself (e.g., running in a loop). A natural pattern in multi-agent interactions is [handoffs](../../concepts/multi_agent#handoffs.md), where one agent hands off control to another. Handoffs allow you to specify:

- **destination**: target agent to navigate to - node name in LangGraph
- **payload**: information to pass to that agent - state update in LangGraph

To implement handoffs in LangGraph, agent nodes can return `Command` object that allows you to [combine both control flow and state updates](../command.md):

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

One of the most common agent types is a tool-calling agent. For those types of agents, one pattern is wrapping a handoff in a tool call, e.g.:

```python
@tool
def transfer_to_bob(state):
    """Transfer to bob."""
    return Command(
        goto="bob",
        update={"my_state_key": "my_state_value"},
        # Each tool-calling agent is implemented as a subgraph.
        # As a result, to navigate to another agent (a sibling sub-graph), 
        # we need to specify that navigation is w/ respect to the parent graph.
        graph=Command.PARENT,
    )
```

This guide shows how you can:

- implement handoffs using `Command`: agent node makes some decision (usually LLM-based), and explicitly returns a handoff via `Command`. These are useful when you need fine-grained control over how an agent routes to another agent. It could be well suited for implementing a supervisor agent in a supervisor architecture.
- implement handoffs using tools: a tool-calling agent has access to tools that can return a handoff via `Command`. The tool-executing node in the agent recognizes `Command` objects returned by the tools and routes accordingly. Handoff tool a general-purpose primitive that is useful in any multi-agent systems that contain tool-calling agents.

## Setup


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

## Implement handoffs using `Command`

Let's implement a system with two agents:

- an addition expert (can only add numbers)
- a multiplication expert (can only multiply numbers).

In this example the agents will be relying on the LLM for doing math. In a more realistic [follow-up example](#using-with-a-custom-agent.md), we will give the agents tools for doing math.

When the addition expert needs help with multiplication, it hands off to the multiplication expert and vice-versa. This is an example of a simple multi-agent network.

Each agent will have a corresponding node function that can conditionally return a `Command` object (e.g. our handoff). The node function will use an LLM with a system prompt and a tool that lets it signal when it needs to hand off to another agent. If the LLM responds with the tool calls, we will return a `Command(goto=<other_agent>)`.

> **Note**: while we're using tools for the LLM to signal that it needs a handoff, the condition for the handoff can be anything: a specific response text from the LLM, structured output from the LLM, any other custom logic, etc.


```python
from typing_extensions import Literal
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command

model = ChatAnthropic(model="claude-3-5-sonnet-latest")


@tool
def transfer_to_multiplication_expert():
    """Ask multiplication agent for help."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to another agent
    # (See the paragraph above)
    return


@tool
def transfer_to_addition_expert():
    """Ask addition agent for help."""
    return


def addition_expert(
    state: MessagesState,
) -> Command[Literal["multiplication_expert", "__end__"]]:
    system_prompt = (
        "You are an addition expert, you can ask the multiplication expert for help with multiplication. "
        "Always do your portion of calculation before the handoff."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_multiplication_expert]).invoke(messages)
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
        return Command(
            goto="multiplication_expert", update={"messages": [ai_msg, tool_msg]}
        )

    # If the expert has an answer, return it directly to the user
    return {"messages": [ai_msg]}


def multiplication_expert(
    state: MessagesState,
) -> Command[Literal["addition_expert", "__end__"]]:
    system_prompt = (
        "You are a multiplication expert, you can ask an addition expert for help with addition. "
        "Always do your portion of calculation before the handoff."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_addition_expert]).invoke(messages)
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="addition_expert", update={"messages": [ai_msg, tool_msg]})

    return {"messages": [ai_msg]}
```

Let's now combine both of these nodes into a single graph. Note that there are no edges between the agents! If the expert has an answer, it will return it directly to the user, otherwise it will route to the other expert for help.


```python
builder = StateGraph(MessagesState)
builder.add_node("addition_expert", addition_expert)
builder.add_node("multiplication_expert", multiplication_expert)
# we'll always start with the addition expert
builder.add_edge(START, "addition_expert")
graph = builder.compile()
```

Finally, let's define a helper function to render the streamed outputs nicely:


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

Let's run the graph with an expression that requires both addition and multiplication:


```python
for chunk in graph.stream(
    {"messages": [("user", "what's (3 + 5) * 12")]},
):
    pretty_print_messages(chunk)
```

You can see that the addition expert first handled the expression in the parentheses, and then handed off to the multiplication expert to finish the calculation.

Now let's see how we can implement this same system using special handoff tools and give our agents actual math tools.

## Implement handoffs using tools

### Implement a handoff tool

In the previous example we explicitly defined custom handoffs in each of the agent nodes. Another pattern is to create special **handoff tools** that directly return `Command` objects. When an agent calls a tool like this, it hands the control off to a different agent. Specifically, the tool-executing node in the agent recognizes the `Command` objects returned by the tools and routes control flow accordingly. **Note**: unlike the previous example, a tool-calling agent is not a single node but another graph that can be added to the multi-agent graph as a subgraph node.

There are a few important considerations when implementing handoff tools:

- since each agent is a __subgraph__ node in another graph, and the tools will be called in one of the agent subgraph nodes (e.g. tool executor), we need to specify `graph=Command.PARENT` in the `Command`, so that LangGraph knows to navigate outside of the agent subgraph
- we can optionally specify a state update that will be applied to the parent graph state before the next agent is called
    - these state updates can be used to control [how much of the chat message history](../../concepts/multi_agent#shared-message-list.md) the target agent sees. For example, you might choose to just share the last AI messages from the current agent, or its full internal chat history, etc. In the examples below we'll be sharing the full internal chat history.

- we can optionally provide the following to the tool (in the tool function signature):
    - graph state (using [`InjectedState`][langgraph.prebuilt.tool_node.InjectedState])
    - graph long-term memory (using [`InjectedStore`][langgraph.prebuilt.tool_node.InjectedStore])
    - the current tool call ID (using [`InjectedToolCallId`](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.base.InjectedToolCallId.html))
      
    These are not necessary but are useful for creating the state update passed to the next agent.


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
        # # optionally pass current graph state to the tool (will be ignored by the LLM)
        state: Annotated[dict, InjectedState],
        # optionally pass the current tool call ID (will be ignored by the LLM)
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
            # the resulting chat history is valid. See the paragraph above for more information.
            update={"messages": state["messages"] + [tool_message]},
        )

    return handoff_to_agent
```

### Using with a custom agent

To demonstrate how to use handoff tools, let's first implement a simple version of the prebuilt [create_react_agent][langgraph.prebuilt.chat_agent_executor.create_react_agent]. This is useful in case you want to have a custom tool-calling agent implementation and want to leverage handoff tools.


```python
from typing_extensions import Literal
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command


def make_agent(model, tools, system_prompt=None):
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    def call_model(state: MessagesState) -> Command[Literal["call_tools", "__end__"]]:
        messages = state["messages"]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        response = model_with_tools.invoke(messages)
        if len(response.tool_calls) > 0:
            return Command(goto="call_tools", update={"messages": [response]})

        return {"messages": [response]}

    # NOTE: this is a simplified version of the prebuilt ToolNode
    # If you want to have a tool node that has full feature parity, please refer to the source code
    def call_tools(state: MessagesState) -> Command[Literal["call_model"]]:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tool_call in tool_calls:
            tool_ = tools_by_name[tool_call["name"]]
            tool_input_fields = tool_.get_input_schema().model_json_schema()[
                "properties"
            ]

            # this is simplified for demonstration purposes and
            # is different from the ToolNode implementation
            if "state" in tool_input_fields:
                # inject state
                tool_call = {**tool_call, "args": {**tool_call["args"], "state": state}}

            tool_response = tool_.invoke(tool_call)
            if isinstance(tool_response, ToolMessage):
                results.append(Command(update={"messages": [tool_response]}))

            # handle tools that return Command directly
            elif isinstance(tool_response, Command):
                results.append(tool_response)

        # NOTE: nodes in LangGraph allow you to return list of updates, including Command objects
        return results

    graph = StateGraph(MessagesState)
    graph.add_node(call_model)
    graph.add_node(call_tools)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_tools", "call_model")

    return graph.compile()
```

Let's also define math tools that we'll give our agents:


```python
@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b
```

Let's test the agent implementation out to make sure it's working as expected:


```python
agent = make_agent(model, [add, multiply])

for chunk in agent.stream({"messages": [("user", "what's (3 + 5) * 12")]}):
    pretty_print_messages(chunk)
```

Now, we can implement our multi-agent system with the multiplication and addition expert agents. This time we'll give them the tools for doing math, as well as our special handoff tools:


```python
addition_expert = make_agent(
    model,
    [add, make_handoff_tool(agent_name="multiplication_expert")],
    system_prompt="You are an addition expert, you can ask the multiplication expert for help with multiplication.",
)
multiplication_expert = make_agent(
    model,
    [multiply, make_handoff_tool(agent_name="addition_expert")],
    system_prompt="You are a multiplication expert, you can ask an addition expert for help with addition.",
)

builder = StateGraph(MessagesState)
builder.add_node("addition_expert", addition_expert)
builder.add_node("multiplication_expert", multiplication_expert)
builder.add_edge(START, "addition_expert")
graph = builder.compile()
```

Let's run the graph with the same multi-step calculation input as before:


```python
for chunk in graph.stream(
    {"messages": [("user", "what's (3 + 5) * 12")]}, subgraphs=True
):
    pretty_print_messages(chunk)
```

We can see that after the addition expert is done with the first part of the calculation (after calling the `add` tool), it decides to hand off to the multiplication expert, which computes the final result.

## Using with a prebuilt ReAct agent

If you don't need extra customization, you can use the prebuilt [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent], which includes built-in support for handoff tools through [`ToolNode`][langgraph.prebuilt.tool_node.ToolNode].


```python
from langgraph.prebuilt import create_react_agent

addition_expert = create_react_agent(
    model,
    [add, make_handoff_tool(agent_name="multiplication_expert")],
    prompt="You are an addition expert, you can ask the multiplication expert for help with multiplication.",
)

multiplication_expert = create_react_agent(
    model,
    [multiply, make_handoff_tool(agent_name="addition_expert")],
    prompt="You are a multiplication expert, you can ask an addition expert for help with addition.",
)

builder = StateGraph(MessagesState)
builder.add_node("addition_expert", addition_expert)
builder.add_node("multiplication_expert", multiplication_expert)
builder.add_edge(START, "addition_expert")
graph = builder.compile()
```

We can now verify that the prebuilt ReAct agent works exactly the same as the custom agent above:


```python
for chunk in graph.stream(
    {"messages": [("user", "what's (3 + 5) * 12")]}, subgraphs=True
):
    pretty_print_messages(chunk)
```
