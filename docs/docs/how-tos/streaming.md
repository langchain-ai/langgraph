# How to stream

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Streaming](../../concepts/streaming/.md)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.

LangGraph is built with first class support for streaming. There are several different ways to stream back outputs from a graph run:

- `"values"`: Emit all values in the state after each step.
- `"updates"`: Emit only the node names and updates returned by the nodes after each step.
    If multiple updates are made in the same step (e.g. multiple nodes are run) then those updates are emitted separately.
- `"custom"`: Emit custom data from inside nodes using `StreamWriter`.
- [`"messages"`](../streaming-tokens.md): Emit LLM messages token-by-token together with metadata for any LLM invocations inside nodes.
- `"debug"`: Emit debug events with as much information as possible for each step.

You can stream outputs from the graph by using `graph.stream(..., stream_mode=<stream_mode>)` method, e.g.:

=== "Sync"

    ```python
    for chunk in graph.stream(inputs, stream_mode="updates"):
        print(chunk)
    ```

=== "Async"

    ```python
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        print(chunk)
    ```

You can also combine multiple streaming mode by providing a list to `stream_mode` parameter:

=== "Sync"

    ```python
    for chunk in graph.stream(inputs, stream_mode=["updates", "custom"]):
        print(chunk)
    ```

=== "Async"

    ```python
    async for chunk in graph.astream(inputs, stream_mode=["updates", "custom"]):
        print(chunk)
    ```

## Setup


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

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

Let's define a simple graph with two nodes:

## Define graph


```python
from typing import TypedDict
from langgraph.graph import StateGraph, START


class State(TypedDict):
    topic: str
    joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}


graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .compile()
)
```

## Stream all values in the state (stream_mode="values") {#values}

Use this to stream **all values** in the state after each step.


```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    # highlight-next-line
    stream_mode="values",
):
    print(chunk)
```

## Stream state updates from the nodes (stream_mode="updates") {#updates}

Use this to stream only the **state updates** returned by the nodes after each step. The streamed outputs include the name of the node as well as the update.


```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    # highlight-next-line
    stream_mode="updates",
):
    print(chunk)
```

## Stream debug events (stream_mode="debug") {#debug}

Use this to stream **debug events** with as much information as possible for each step. Includes information about tasks that were scheduled to be executed as well as the results of the task executions.


```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    # highlight-next-line
    stream_mode="debug",
):
    print(chunk)
```

## Stream LLM tokens ([stream_mode="messages"](../streaming-tokens.md)) {#messages}

Use this to stream **LLM messages token-by-token** together with metadata for any LLM invocations inside nodes or tasks. Let's modify the above example to include LLM calls:


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")


def generate_joke(state: State):
    # highlight-next-line
    llm_response = llm.invoke(
        # highlight-next-line
        [
            # highlight-next-line
            {"role": "user", "content": f"Generate a joke about {state['topic']}"}
            # highlight-next-line
        ]
        # highlight-next-line
    )
    return {"joke": llm_response.content}


graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .compile()
)
```


```python
for message_chunk, metadata in graph.stream(
    {"topic": "ice cream"},
    # highlight-next-line
    stream_mode="messages",
):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)
```


```python
metadata
```






## Stream custom data (stream_mode="custom") {#custom}

Use this to stream custom data from inside nodes using [`StreamWriter`][langgraph.types.StreamWriter].


```python
from langgraph.types import StreamWriter


# highlight-next-line
def generate_joke(state: State, writer: StreamWriter):
    # highlight-next-line
    writer({"custom_key": "Writing custom data while generating a joke"})
    return {"joke": f"This is a joke about {state['topic']}"}


graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .compile()
)
```


```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    # highlight-next-line
    stream_mode="custom",
):
    print(chunk)
```

## Configure multiple streaming modes (stream_mode="custom") {#multiple}

Use this to combine multiple streaming modes. The outputs are streamed as tuples `(stream_mode, streamed_output)`.


```python
for stream_mode, chunk in graph.stream(
    {"topic": "ice cream"},
    # highlight-next-line
    stream_mode=["updates", "custom"],
):
    print(f"Stream mode: {stream_mode}")
    print(chunk)
    print("\n")
```
