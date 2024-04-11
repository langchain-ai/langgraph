# 🦜🕸️LangGraph

[![Downloads](https://static.pepy.tech/badge/langgraph/month)](https://pepy.tech/project/langgraph)

⚡ Build language agents as graphs ⚡


## Overview

Suppose you're building a customer support assistant. You want your assistant to be able to:

1. Use tools to respond to questions
2. Connect with a human if needed
3. Be able to pause the process indefinitely and resume whenever the human responds

LangGraph makes this all easy. First install:

```shell
pip install -U langgraph
```

Then define your assistant:

```python
import json

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt.tool_node import ToolNode


# Define the function that determines whether to continue or not
def should_continue(messages):
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    else:
        return "action"


# Define a new graph
workflow = MessageGraph()

tools = [TavilySearchResults(max_results=1)]
model = ChatAnthropic(model="claude-3-haiku-20240307").bind_tools(tools)
workflow.add_node("agent", model)
workflow.add_node("action", ToolNode(tools))

workflow.set_entry_point("agent")

# Conditional agent -> action OR agent -> END
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# Always transition `action` -> `agent`
workflow.add_edge("action", "agent")

memory = SqliteSaver.from_conn_string(":memory:")

# Setting the interrupt means that any time an action is called, the machine will stop
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])

# Run the graph
thread = {"configurable": {"thread_id": "4"}}
for event in app.stream("what is the weather in sf currently", thread):
    for v in event.values():
        print(v)

# The graph has paused before the 'action' node and can be resumed at any time
# ...
# Resume from where you left off
for event in app.stream(None, thread):
    for v in event.values():
        print(v)
```

The graph orchestrates everything:

- The `MessageGraph` contains the agent's "Memory"
- Conditional edges enable dynamic routing between the chatbot, tools, and the user
- Persistence makes it easy to stop, resume, and even rewind for full control over your application

With LangGraph, you can build complex, stateful agents without getting bogged down in manual state and interrupt management. Just define your nodes, edges, and state schema - and let the graph take care of the rest.


## Tutorials

Consult the [Tutorials](tutorials/index.md) to learn more about implementing advanced 

- **Agent Executors**: Chat and Langchain agents
- **Planning Agents**: Plan-and-Execute, ReWOO, LLMCompiler  
- **Reflection & Critique**: Improving quality via reflection
- **Multi-Agent Systems**: Collaboration, supervision, teams
- **Research & QA**: Web research, retrieval-augmented QA  
- **Applications**: Chatbots, code assist, web tasks
- **Evaluation & Analysis**: Simulation, self-discovery, swarms


## How-To Guides

Check out the [How-To Guides](how-tos/index.md) for instructions on handling common tasks with LangGraph.

- Manage State
- Tool Integration  
- Human-in-the-Loop
- Async Execution
- Streaming Responses
- Subgraphs & Branching
- Persistence, Visualization, Time Travel 
- Benchmarking

## Concepts

- [Graphs](concepts.md#graphs)
- [State](concepts.md#state): The data structure passed between nodes, allowing you to persist context 
- [Nodes](concepts.#nodes): The building blocks of your graph - LLMs, tools, or custom logic 
- [Edges](concepts.md#edges): The connections that define the flow of data between your nodes
- [Conditional Edges](concepts.md#conditional_edges): Special edges that let you dynamically route between nodes based on state
- [Persistence](concepts.md#persistence): Save and resume your graph's state for long-running applications


## Why LangGraph?

LangGraph is framework agnostic (each node is a regular python function). It extends the core Runnable API (shared interface for streaming, async, and batch calls) to make it easy to:

- Seamless state management across multiple turns of conversation or tool usage
- The ability to flexibly route between nodes based on dynamic criteria 
- Smooth switching between LLMs and human intervention  
- Persistence for long-running, multi-session applications

If you're building a straightforward DAG, Runnables are a great fit. But for more complex, stateful applications with nonlinear flows, LangGraph is the perfect tool for the job.