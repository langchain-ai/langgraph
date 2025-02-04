# How to stream LLM tokens from specific nodes

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Streaming](../../concepts/streaming/.md)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

A common use case when [streaming LLM tokens](../streaming-tokens.md) is to only stream them from specific nodes. To do so, you can use `stream_mode="messages"` and filter the outputs by the `langgraph_node` field in the streamed metadata:

```python
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

model = ChatOpenAI()

def node_a(state: State):
    model.invoke(...)
    ...

def node_b(state: State):
    model.invoke(...)
    ...

graph = (
    StateGraph(State)
    .add_node(node_a)
    .add_node(node_b)
    ...
    .compile()
    
for msg, metadata in graph.stream(
    inputs,
    # highlight-next-line
    stream_mode="messages"
):
    # stream from 'node_a'
    # highlight-next-line
    if metadata["langgraph_node"] == "node_a":
        print(msg)
```

!!! note "Streaming from a specific LLM invocation"

    If you need to instead filter streamed LLM tokens to a specific LLM invocation, check out [this guide](../streaming-tokens#filter-to-specific-llm-invocation.md)

## Setup

First we need to install the packages required


```
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

## Example


```python
from typing import TypedDict
from langgraph.graph import START, StateGraph, MessagesState
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    topic: str
    joke: str
    poem: str


def write_joke(state: State):
    topic = state["topic"]
    joke_response = model.invoke(
        [{"role": "user", "content": f"Write a joke about {topic}"}]
    )
    return {"joke": joke_response.content}


def write_poem(state: State):
    topic = state["topic"]
    poem_response = model.invoke(
        [{"role": "user", "content": f"Write a short poem about {topic}"}]
    )
    return {"poem": poem_response.content}


graph = (
    StateGraph(State)
    .add_node(write_joke)
    .add_node(write_poem)
    # write both the joke and the poem concurrently
    .add_edge(START, "write_joke")
    .add_edge(START, "write_poem")
    .compile()
)
```


```python
for msg, metadata in graph.stream(
    {"topic": "cats"},
    # highlight-next-line
    stream_mode="messages",
):
    # highlight-next-line
    if msg.content and metadata["langgraph_node"] == "write_poem":
        print(msg.content, end="|", flush=True)
```
