---
hide_comments: true
---
# 🦜🕸️LangGraph

[![Downloads](https://static.pepy.tech/badge/langgraph/month)](https://pepy.tech/project/langgraph)

⚡ Build language agents as graphs ⚡

!!! note "Python version :material-language-python:"

    Looking for the JS version? Click [:fontawesome-brands-square-js: here](https://github.com/langchain-ai/langgraphjs) ([:simple-readme: JS docs](https://langchain-ai.github.io/langgraphjs/)).

## Overview

Suppose you're building a customer support assistant. You want your assistant to be able to:

1. Use tools to respond to questions
2. Connect with a human if needed
3. Be able to pause the process indefinitely and resume whenever the human responds

LangGraph makes this all easy. First install:

```bash
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

memory = SqliteSaver.from_conn_string(":memory:") # Here we only save in-memory

# Setting the interrupt means that any time an action is called, the machine will stop
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])
```

Now, run the graph:

```python
# Run the graph
thread = {"configurable": {"thread_id": "4"}}
for event in app.stream("what is the weather in sf currently", thread, stream_mode="values"):
    for v in event.values():
        print(v)

```
We configured the graph to **wait** before executing the `action`. The `SqliteSaver` persists the state. Resume at any time.

```python
for event in app.stream(None, thread, stream_mode="values"):
    for v in event.values():
        print(v)
```

The graph orchestrates everything:

- The `MessageGraph` contains the agent's "Memory"
- Conditional edges enable dynamic routing between the chatbot, tools, and the user
- Persistence makes it easy to stop, resume, and even rewind for full control over your application

With LangGraph, you can build complex, stateful agents without getting bogged down in manual state and interrupt management. Just define your nodes, edges, and state schema - and let the graph take care of the rest.


## Tutorials

Consult the [Tutorials](tutorials/index.md) to learn more about building with LangGraph, including advanced use cases.


## How-To Guides

Check out the [How-To Guides](how-tos/index.md) for instructions on handling common tasks with LangGraph

## Reference

For documentation on the core APIs, check out the [Reference](reference/graphs.md) docs.

## Conceptual Guides

Once you've learned the basics, if you want to further understand LangGraph's core abstractions, check out the [Conceptual Guides](./concepts/index.md).

## Why LangGraph?

LangGraph is framework agnostic (each node is a regular python function). It extends the core Runnable API (shared interface for streaming, async, and batch calls) to make it easy to:

- Seamless state management across multiple turns of conversation or tool usage
- The ability to flexibly route between nodes based on dynamic criteria 
- Smooth switching between LLMs and human intervention  
- Persistence for long-running, multi-session applications

If you're building a straightforward DAG, Runnables are a great fit. But for more complex, stateful applications with nonlinear flows, LangGraph is the perfect tool for the job.