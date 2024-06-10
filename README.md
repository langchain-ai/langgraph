# 🦜🕸️LangGraph

![Version](https://img.shields.io/pypi/v/langgraph)
[![Downloads](https://static.pepy.tech/badge/langgraph/month)](https://pepy.tech/project/langgraph)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph/issues)
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.com/channels/1038097195422978059/1170024642245832774)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://langchain-ai.github.io/langgraph/)

⚡ Building language agents as graphs ⚡

## Overview

[LangGraph](https://langchain-ai.github.io/langgraph/) is a library for building stateful, multi-actor applications with LLMs.
Inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/), LangGraph lets you coordinate and checkpoint multiple chains (or actors) across cyclic computational steps using regular python functions (or [JS](https://github.com/langchain-ai/langgraphjs)). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/).

### Why LangGraph?

The main purpose of LangGraph is adding **cycles** and **persistence** to your LLM application. Cycles are important for agentic behaviors, where you call an LLM in a loop, asking it what action to take next.

LangGraph is framework agnostic (each node is a regular python function). It extends the core Runnable API (shared interface for streaming, async, and batch calls) to make it easy to:

* Seamlesssly manage state across multiple turns of conversation or tool usage
* Flexibly route between nodes based on dynamic criteria
* Smoothly switch between LLMs and human intervention
* Add persistence for long-running, multi-session applications

**NOTE**: If you only need simple Directed Acyclic Graphs (DAGs), you can already accomplish this using [LangChain Expression Language](https://python.langchain.com/docs/expression_language/). But for more complex, stateful applications with nonlinear flows, LangGraph is the perfect tool for the job.

### Key Features

- **Cycles and Branching**: Implement loops and conditionals in your apps.
- **Persistence**: Automatically save state after each step in the graph. Pause and resume the graph execution at any point to support error recovery, human-in-the-loop workflows, time travel and more.
- **Human-in-the-Loop**: Interrupt graph execution to approve or edit next action planned by the agent.
- **Streaming Support**: Stream outputs as they are produced by each node (including token streaming).
- **Integration with LangChain**: LangGraph integrates seamlessly with [LangChain](https://github.com/langchain-ai/langchain/) and [LangSmith](https://docs.smith.langchain.com/).


## Installation

```shell
pip install -U langgraph
```

## Example

One of the central concepts of LangGraph is state. Each graph execution creates a state that is passed between nodes in the graph as they execute, and each node updates this internal state with its return value after it executes. The way that the graph updates its internal state is defined by either the type of graph chosen or a custom function.

Let's take a look at a simple example of an agent that can search the web using [Tavily Search API](https://tavily.com/).

```shell
pip install langchain_openai langchain_community
```

```shell
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
```

Optionally, we can set up [LangSmith](https://docs.smith.langchain.com/) for best-in-class observability.

```shell
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY=ls__...
```

```python
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode


# Define the tools for the agent to use
tools = [TavilySearchResults(max_results=1)]
tool_node = ToolNode(tools)

model = ChatOpenAI(temperature=0).bind_tools(tools)

def add_messages(left: list, right: list):
    """Add-don't-overwrite."""
    return left + right


# Define graph state
class AgentState(TypedDict):
    # The `add_messages` function within the annotation defines
    # *how* updates should be merged into the state.
    messages: Annotated[list, add_messages]


# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return "__end__"


# Define the function that calls the model
def call_model(state: AgentState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)

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

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

# Use the Runnable
final_state = app.invoke({"messages": [HumanMessage(content="what is the weather in sf")]})
final_state["messages"][-1].content
```

```
'The current weather in San Francisco is as follows:\n- Temperature: 60.1°F (15.6°C)\n- Condition: Partly cloudy\n- Wind: 5.6 mph (9.0 kph) from SSW\n- Humidity: 83%\n- Visibility: 9.0 miles (16.0 km)\n- UV Index: 4.0\n\nFor more details, you can visit [Weather API](https://www.weatherapi.com/).'
```

### Step-by-step Breakdown:

1. <details>
    <summary>Initialize the model, tools and the graph.</summary>

    - we use `ChatOpenAI` as our LLM. **NOTE:** we need make sure the model knows that it has these tools available to call. We can do this by converting the LangChain tools into the format for OpenAI tool calling using the `.bind_tools()` method.
    - we define the tools we want to use -- a web search tool in our case. It is really easy to create your own tools - see documentation here on how to do that [here](https://python.langchain.com/docs/modules/agents/tools/custom_tools).
   </details>
2. <details>
    <summary>Define graph nodes.</summary>

    There are two main nodes we need for this:
      - The `agent` node: responsible for deciding what (if any) actions to take.
      - The `tools` node that invokes tools: if the agent decides to take an action, this node will then execute that action.
   </details>
3. <details
    <summary>Define graph edges</summary>
      We define both normal and conditional edges. Conditional edge means that the destination depends on the contents of the graph's State. In our case, the destination is not known until the agent (LLM) decides.

      Our graph has one of each type of edge:
      - Conditional edge: after the agent is called, we should either:
        - a. Run tools if the agent said to take an action, OR
        - b. Finish (respond to the user) if the agent did not ask to run tools
      - Normal edge: after the tools are invoked, the graph should always return to the agent to decide what to do next
   </details>
4. Set the entry point for graph execution (`agent`).
5. <details>
    <summary>Compile the graph.</summary>

    When we compile the graph, we are translating it to low-level [Pregel](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/) operations
    </details>
6. <details>
   <summary>Execute the graph</summary>

    1. LangGraph adds the input message to the internal state, then passes the state to the entrypoint node, `"agent"`.
    2. The `"agent"` node executes, invoking the chat model.
    3. The chat model returns an `AIMessage`. LangGraph adds this to the state.
    4. Graph cycles the following steps until there are no more `tool_calls` on `AIMessage`:
      - If `AIMessage` has `tool_calls`, `"tools"` node executes
      - The `"agent"` node executes again and returns `AIMessage`
    5. Execution progresses to the special `END` value and outputs the final state.
    And as a result, we get a list of all our chat messages as output.
   </details>

## Advanced usage

For more advanced examples of LangGraph agents with with tool calling, conditional edges and cycles see [How-to Guides](https://langchain-ai.github.io/langgraph/how-tos/)


## Documentation

* [Tutorials](https://langchain-ai.github.io/langgraph/tutorials/): Learn to build with LangGraph through guided examples.
* [How-to Guides](https://langchain-ai.github.io/langgraph/how-tos/): Accomplish specific things within LangGraph, from streaming, to adding memory & persistence, to common design patterns (branching, subgraphs, etc.), these are the place to go if you want to copy and run a specific code snippet.
* [Conceptual Guides](https://langchain-ai.github.io/langgraph/concepts/): In-depth explanations of the key concepts and principles behind LangGraph, such as nodes, edges, state and more.
* [API Reference](https://langchain-ai.github.io/langgraph/reference/graphs/): Review important classes and methods, simple examples of how to use the graph and checkpointing APIs, higher-level prebuilt components and more.

## Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see [here](https://python.langchain.com/v0.2/docs/contributing/).