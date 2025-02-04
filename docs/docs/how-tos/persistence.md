# How to add thread-level persistence to your graph

<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide assumes familiarity with the following:
        <ul>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/persistence/">
                    Persistence
                </a>
            </li>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/memory/">
                    Memory
                </a>
            </li>
            <li>
                <a href="https://python.langchain.com/docs/concepts/#chat-models/">
                    Chat Models
                </a>
            </li>        
        </ul>
    </p>
</div> 

Many AI applications need memory to share context across multiple interactions. In LangGraph, this kind of memory can be added to any [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph) using [thread-level persistence](https://langchain-ai.github.io/langgraph/concepts/persistence) .

When creating any LangGraph graph, you can set it up to persist its state by adding a [checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#basecheckpointsaver) when compiling the graph:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph.compile(checkpointer=checkpointer)
```

This guide shows how you can add thread-level persistence to your graph.

<div class="admonition tip">
    <p class="admonition-title">Note</p>
    <p>
        If you need memory that is <b>shared</b> across multiple conversations or users (cross-thread persistence), check out this <a href="https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/">how-to guide</a>.
    </p>
</div>

## Setup

First we need to install the packages required


```
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_anthropic
```

Next, we need to set API key for Anthropic (the LLM we will use).


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

## Define graph

We will be using a single-node graph that calls a [chat model](https://python.langchain.com/docs/concepts/#chat-models).

Let's first define the model we'll be using:


```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
```

Now we can define our `StateGraph` and add our model-calling node:


```python
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, MessagesState, START


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile()
```

If we try to use this graph, the context of the conversation will not be persisted across interactions:


```python
input_message = {"role": "user", "content": "hi! I'm bob"}
for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

input_message = {"role": "user", "content": "what's my name?"}
for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

## Add persistence

To add in persistence, we need to pass in a [Checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver) when compiling the graph.


```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
# If you're using LangGraph Cloud or LangGraph Studio, you don't need to pass the checkpointer when compiling the graph, since it's done automatically.
```

<div class="admonition tip">
    <p class="admonition-title">Note</p>
    <p>
        If you're using LangGraph Cloud or LangGraph Studio, you <strong>don't need</strong> to pass checkpointer when compiling the graph, since it's done automatically.
    </p>
</div>

We can now interact with the agent and see that it remembers previous messages!


```python
config = {"configurable": {"thread_id": "1"}}
input_message = {"role": "user", "content": "hi! I'm bob"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

You can always resume previous threads:


```python
input_message = {"role": "user", "content": "what's my name?"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

If we want to start a new conversation, we can pass in a different `thread_id`. Poof! All the memories are gone!


```python
input_message = {"role": "user", "content": "what's my name?"}
for chunk in graph.stream(
    {"messages": [input_message]},
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
```
