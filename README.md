# 🦜🕸️LangGraph

![Version](https://img.shields.io/pypi/v/langgraph)
[![Downloads](https://static.pepy.tech/badge/langgraph/month)](https://pepy.tech/project/langgraph)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph/issues)
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.com/channels/1038097195422978059/1170024642245832774)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://langchain-ai.github.io/langgraph/)

⚡ Building language agents as graphs ⚡

> [!NOTE]
> Looking for the JS version? Click [here](https://github.com/langchain-ai/langgraphjs) ([JS docs](https://langchain-ai.github.io/langgraphjs/)).

> [!TIP]
> Looking to deploy your LangGraph application? [Join the waitlist](https://www.langchain.com/langgraph-cloud-beta) for [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/), our managed service for deploying and hosting LangGraph applications.

## Overview

[LangGraph](https://langchain-ai.github.io/langgraph/) is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence. LangGraph allows you to define flows that involve cycles, essential for most agentic architectures, differentiating it from DAG-based solutions. As a very low-level framework, it provides fine-grained control over both the flow and state of your application, crucial for creating reliable agents. Additionally, LangGraph includes built-in persistence, enabling advanced human-in-the-loop and memory features.

LangGraph is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/). LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.

### Key Features

- **Cycles and Branching**: Implement loops and conditionals in your apps.
- **Persistence**: Automatically save state after each step in the graph. Pause and resume the graph execution at any point to support error recovery, human-in-the-loop workflows, time travel and more.
- **Human-in-the-Loop**: Interrupt graph execution to approve or edit next action planned by the agent.
- **Streaming Support**: Stream outputs as they are produced by each node (including token streaming).
- **Integration with LangChain**: LangGraph integrates seamlessly with [LangChain](https://github.com/langchain-ai/langchain/) and [LangSmith](https://docs.smith.langchain.com/) (but does not require them).


## Installation

```shell
pip install -U langgraph
```

## Example

One of the central concepts of LangGraph is state. Each graph execution creates a state that is passed between nodes in the graph as they execute, and each node updates this internal state with its return value after it executes. The way that the graph updates its internal state is defined by either the type of graph chosen or a custom function.

Let's take a look at a simple example of an agent that can use a search tool.

```shell
pip install langchain-anthropic
```

```shell
export ANTHROPIC_API_KEY=sk-...
```

Optionally, we can set up [LangSmith](https://docs.smith.langchain.com/) for best-in-class observability.

```shell
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_sk_...
```

```python
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return ["It's 60 degrees and foggy."]
    return ["It's 90 degrees and sunny."]


tools = [search]

tool_node = ToolNode(tools)

model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0).bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState):
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

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
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content
```

```
"Based on the search results, I can tell you that the current weather in San Francisco is:\n\nTemperature: 60 degrees Fahrenheit\nConditions: Foggy\n\nSan Francisco is known for its microclimates and frequent fog, especially during the summer months. The temperature of 60°F (about 15.5°C) is quite typical for the city, which tends to have mild temperatures year-round. The fog, often referred to as "Karl the Fog" by locals, is a characteristic feature of San Francisco\'s weather, particularly in the mornings and evenings.\n\nIs there anything else you\'d like to know about the weather in San Francisco or any other location?"
```

Now when we pass the same `"thread_id"`, the conversation context is retained via the saved state (i.e. stored list of messages)

```python
final_state = app.invoke(
    {"messages": [HumanMessage(content="what about ny")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content
```

```
"Based on the search results, I can tell you that the current weather in New York City is:\n\nTemperature: 90 degrees Fahrenheit (approximately 32.2 degrees Celsius)\nConditions: Sunny\n\nThis weather is quite different from what we just saw in San Francisco. New York is experiencing much warmer temperatures right now. Here are a few points to note:\n\n1. The temperature of 90°F is quite hot, typical of summer weather in New York City.\n2. The sunny conditions suggest clear skies, which is great for outdoor activities but also means it might feel even hotter due to direct sunlight.\n3. This kind of weather in New York often comes with high humidity, which can make it feel even warmer than the actual temperature suggests.\n\nIt's interesting to see the stark contrast between San Francisco's mild, foggy weather and New York's hot, sunny conditions. This difference illustrates how varied weather can be across different parts of the United States, even on the same day.\n\nIs there anything else you'd like to know about the weather in New York or any other location?"
```

### Step-by-step Breakdown

1. <details>
    <summary>Initialize the model and tools.</summary>

    - we use `ChatAnthropic` as our LLM. **NOTE:** we need make sure the model knows that it has these tools available to call. We can do this by converting the LangChain tools into the format for OpenAI tool calling using the `.bind_tools()` method.
    - we define the tools we want to use - a search tool in our case. It is really easy to create your own tools - see documentation here on how to do that [here](https://python.langchain.com/docs/modules/agents/tools/custom_tools).
   </details>

2. <details>
    <summary>Initialize graph with state.</summary>

    - we initialize graph (`StateGraph`) by passing state schema (in our case `MessagesState`)
    - `MessagesState` is a prebuilt state schema that has one attribute -- a list of LangChain `Message` objects, as well as logic for merging the updates from each node into the state
   </details>

3. <details>
    <summary>Define graph nodes.</summary>

    There are two main nodes we need:

      - The `agent` node: responsible for deciding what (if any) actions to take.
      - The `tools` node that invokes tools: if the agent decides to take an action, this node will then execute that action.
   </details>

4. <details>
    <summary>Define entry point and graph edges.</summary>

      First, we need to set the entry point for graph execution - `agent` node.

      Then we define one normal and one conditional edge. Conditional edge means that the destination depends on the contents of the graph's state (`MessageState`). In our case, the destination is not known until the agent (LLM) decides.

      - Conditional edge: after the agent is called, we should either:
        - a. Run tools if the agent said to take an action, OR
        - b. Finish (respond to the user) if the agent did not ask to run tools
      - Normal edge: after the tools are invoked, the graph should always return to the agent to decide what to do next
   </details>

5. <details>
    <summary>Compile the graph.</summary>

    - When we compile the graph, we turn it into a LangChain [Runnable](https://python.langchain.com/v0.2/docs/concepts/#runnable-interface), which automatically enables calling `.invoke()`, `.stream()` and `.batch()` with your inputs
    - We can also optionally pass checkpointer object for persisting state between graph runs, and enabling memory, human-in-the-loop workflows, time travel and more. In our case we use `MemorySaver` - a simple in-memory checkpointer
    </details>

6. <details>
   <summary>Execute the graph.</summary>

    1. LangGraph adds the input message to the internal state, then passes the state to the entrypoint node, `"agent"`.
    2. The `"agent"` node executes, invoking the chat model.
    3. The chat model returns an `AIMessage`. LangGraph adds this to the state.
    4. Graph cycles the following steps until there are no more `tool_calls` on `AIMessage`:

        - If `AIMessage` has `tool_calls`, `"tools"` node executes
        - The `"agent"` node executes again and returns `AIMessage`

    5. Execution progresses to the special `END` value and outputs the final state.
    And as a result, we get a list of all our chat messages as output.
   </details>


## Documentation

* [Tutorials](https://langchain-ai.github.io/langgraph/tutorials/): Learn to build with LangGraph through guided examples.
* [How-to Guides](https://langchain-ai.github.io/langgraph/how-tos/): Accomplish specific things within LangGraph, from streaming, to adding memory & persistence, to common design patterns (branching, subgraphs, etc.), these are the place to go if you want to copy and run a specific code snippet.
* [Conceptual Guides](https://langchain-ai.github.io/langgraph/concepts/): In-depth explanations of the key concepts and principles behind LangGraph, such as nodes, edges, state and more.
* [API Reference](https://langchain-ai.github.io/langgraph/reference/graphs/): Review important classes and methods, simple examples of how to use the graph and checkpointing APIs, higher-level prebuilt components and more.
* [Cloud (beta)](https://langchain-ai.github.io/langgraph/cloud/): With one click, deploy LangGraph applications to LangGraph Cloud.

## Contributing

For more information on how to contribute, see [here](https://github.com/langchain-ai/langgraph/blob/main/CONTRIBUTING.md).
