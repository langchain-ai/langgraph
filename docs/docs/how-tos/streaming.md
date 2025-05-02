# Stream outputs

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Streaming](../../concepts/streaming/)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.

!!! tip "Set up [LangSmith](https://smith.langchain.com) for LangGraph Development"

    Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started [here](https://docs.smith.langchain.com).

## Using the streaming API

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
    async for mode, chunk in graph.astream(inputs, stream_mode=["updates", "custom"]):
        print(chunk)
    ```


| Mode       | Description                                                                                                                                                                                                                                                                      |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`values`](../how-tos/streaming.ipynb#values)   | Streams the full value of the state after each step of the graph.                                                                                                                                                                           |
| [`updates`](../how-tos/streaming.ipynb#updates) | Streams the updates to the state after each step of the graph. If multiple updates are made in the same step (e.g., multiple nodes are run), those updates are streamed separately.                                                              |
| [`custom`](../how-tos/streaming.ipynb#custom)   | Streams custom data from inside your graph nodes.                                                                                                                                                                                          |
| [`messages`](../how-tos/streaming-tokens.ipynb) | Streams LLM tokens and metadata for the graph node where the LLM is invoked.                                                                                                                                                               |
| [`debug`](../how-tos/streaming.ipynb#debug)     | Streams as much information as possible throughout the execution of the graph.                                                                                                                                                             |


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

## Stream LLM tokens ([stream_mode="messages"](../streaming-tokens)) {#messages}

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

## Configure multiple streaming modes {#multiple}

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

# How to stream LLM tokens from your graph

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Streaming](../../concepts/streaming/)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

When building LLM applications with LangGraph, you might want to stream individual LLM tokens from the LLM calls inside LangGraph nodes. You can do so via `graph.stream(..., stream_mode="messages")`:

```python
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
def call_model(state: State):
    model.invoke(...)
    ...

graph = (
    StateGraph(State)
    .add_node(call_model)
    ...
    .compile()
    
for msg, metadata in graph.stream(inputs, stream_mode="messages"):
    print(msg)
```

The streamed outputs will be tuples of `(message chunk, metadata)`:

* message chunk is the token streamed by the LLM
* metadata is a dictionary with information about the graph node where the LLM was called as well as the LLM invocation metadata

!!! note "Using without LangChain"

    If you need to stream LLM tokens **without using LangChain**, you can use [`stream_mode="custom"`](../streaming/#custom) to stream the outputs from LLM provider clients directly. Check out the [example below](#example-without-langchain) to learn more.

!!! warning "Async in Python < 3.11"
    
    When using Python < 3.11 with async code, please ensure you manually pass the `RunnableConfig` through to the chat model when invoking it like so: `model.ainvoke(..., config)`.
    The stream method collects all events from your nested code using a streaming tracer passed as a callback. In 3.11 and above, this is automatically handled via [contextvars](https://docs.python.org/3/library/contextvars.html); prior to 3.11, [asyncio's tasks](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task) lacked proper `contextvar` support, meaning that the callbacks will only propagate if you manually pass the config through. We do this in the `call_model` function below.

## Setup

First we need to install the packages required


```python
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai
```

Next, we need to set API keys for OpenAI (the LLM we will use).


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

!!! note Manual Callback Propagation

    Note that in `call_model(state: State, config: RunnableConfig):` below, we a) accept the [`RunnableConfig`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig) in the node function and b) pass it in as the second arg for `model.ainvoke(..., config)`. This is optional for python >= 3.11.

## Example

Below we demonstrate an example with two LLM calls in a single node.


```python
from typing import TypedDict
from langgraph.graph import START, StateGraph, MessagesState
from langchain_openai import ChatOpenAI


# Note: we're adding the tags here to be able to filter the model outputs down the line
joke_model = ChatOpenAI(model="gpt-4o-mini", tags=["joke"])
poem_model = ChatOpenAI(model="gpt-4o-mini", tags=["poem"])


class State(TypedDict):
    topic: str
    joke: str
    poem: str


# highlight-next-line
async def call_model(state, config):
    topic = state["topic"]
    print("Writing joke...")
    # Note: Passing the config through explicitly is required for python < 3.11
    # Since context var support wasn't added before then: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    joke_response = await joke_model.ainvoke(
        [{"role": "user", "content": f"Write a joke about {topic}"}],
        # highlight-next-line
        config,
    )
    print("\n\nWriting poem...")
    poem_response = await poem_model.ainvoke(
        [{"role": "user", "content": f"Write a short poem about {topic}"}],
        # highlight-next-line
        config,
    )
    return {"joke": joke_response.content, "poem": poem_response.content}


graph = StateGraph(State).add_node(call_model).add_edge(START, "call_model").compile()
```


```python
async for msg, metadata in graph.astream(
    {"topic": "cats"},
    # highlight-next-line
    stream_mode="messages",
):
    if msg.content:
        print(msg.content, end="|", flush=True)
```


```python
metadata
```

### Filter to specific LLM invocation

You can see that we're streaming tokens from all of the LLM invocations. Let's now filter the streamed tokens to include only a specific LLM invocation. We can use the streamed metadata and filter events using the tags we've added to the LLMs previously:


```python
async for msg, metadata in graph.astream(
    {"topic": "cats"},
    stream_mode="messages",
):
    # highlight-next-line
    if msg.content and "joke" in metadata.get("tags", []):
        print(msg.content, end="|", flush=True)
```

## Example without LangChain


```python
from openai import AsyncOpenAI

openai_client = AsyncOpenAI()
model_name = "gpt-4o-mini"


async def stream_tokens(model_name: str, messages: list[dict]):
    response = await openai_client.chat.completions.create(
        messages=messages, model=model_name, stream=True
    )

    role = None
    async for chunk in response:
        delta = chunk.choices[0].delta

        if delta.role is not None:
            role = delta.role

        if delta.content:
            yield {"role": role, "content": delta.content}


# highlight-next-line
async def call_model(state, config, writer):
    topic = state["topic"]
    joke = ""
    poem = ""

    print("Writing joke...")
    async for msg_chunk in stream_tokens(
        model_name, [{"role": "user", "content": f"Write a joke about {topic}"}]
    ):
        joke += msg_chunk["content"]
        metadata = {**config["metadata"], "tags": ["joke"]}
        chunk_to_stream = (msg_chunk, metadata)
        # highlight-next-line
        writer(chunk_to_stream)

    print("\n\nWriting poem...")
    async for msg_chunk in stream_tokens(
        model_name, [{"role": "user", "content": f"Write a short poem about {topic}"}]
    ):
        poem += msg_chunk["content"]
        metadata = {**config["metadata"], "tags": ["poem"]}
        chunk_to_stream = (msg_chunk, metadata)
        # highlight-next-line
        writer(chunk_to_stream)

    return {"joke": joke, "poem": poem}


graph = StateGraph(State).add_node(call_model).add_edge(START, "call_model").compile()
```

!!! note "stream_mode="custom""

    When streaming LLM tokens without LangChain, we recommend using [`stream_mode="custom"`](../streaming/#stream-modecustom). This allows you to explicitly control which data from the LLM provider APIs to include in LangGraph streamed outputs, including any additional metadata.


```python
async for msg, metadata in graph.astream(
    {"topic": "cats"},
    # highlight-next-line
    stream_mode="custom",
):
    print(msg["content"], end="|", flush=True)
```


```python
metadata
```

To filter to the specific LLM invocation, you can use the streamed metadata:


```python
async for msg, metadata in graph.astream(
    {"topic": "cats"},
    stream_mode="custom",
):
    # highlight-next-line
    if "poem" in metadata.get("tags", []):
        print(msg["content"], end="|", flush=True)
```

# How to stream LLM tokens from specific nodes

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Streaming](../../concepts/streaming/)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

A common use case when [streaming LLM tokens](../streaming-tokens) is to only stream them from specific nodes. To do so, you can use `stream_mode="messages"` and filter the outputs by the `langgraph_node` field in the streamed metadata:

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

    If you need to instead filter streamed LLM tokens to a specific LLM invocation, check out [this guide](../streaming-tokens#filter-to-specific-llm-invocation)

## Setup

First we need to install the packages required


```python
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

# How to stream from subgraphs

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Subgraphs](../../concepts/low_level/#subgraphs)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

If you have created a graph with [subgraphs](../subgraph), you may wish to stream outputs from those subgraphs. To do so, you can specify `subgraphs=True` in parent graph's `.stream()` method:


```python
for chunk in parent_graph.stream(
    {"foo": "foo"},
    # highlight-next-line
    subgraphs=True
):
    print(chunk)
```

## Setup

First let's install the required packages


```python
%%capture --no-stderr
%pip install -U langgraph
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Example

Let's define a simple example:


```python
from langgraph.graph import START, StateGraph
from typing import TypedDict


# Define subgraph
class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str


def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}


def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()


# Define parent graph
class ParentState(TypedDict):
    foo: str


def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}


builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()
```

Let's now stream the outputs from the graph:


```python
for chunk in graph.stream({"foo": "foo"}, stream_mode="updates"):
    print(chunk)
```

You can see that we're only emitting the updates from the parent graph nodes (`node_1` and `node_2`). To emit the updates from the _subgraph_ nodes you can specify `subgraphs=True`:


```python
for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    # highlight-next-line
    subgraphs=True,
):
    print(chunk)
```

Voila! The streamed outputs now contain updates from both the parent graph and the subgraph. **Note** that we are receiving not just the node updates, but we also the namespaces which tell us what graph (or subgraph) we are streaming from.

# How to stream data from within a tool

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Streaming](../../concepts/streaming/)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)
    - [Tools](https://python.langchain.com/docs/concepts/tools/)

If your graph calls tools that use LLMs or any other streaming APIs, you might want to surface partial results during the execution of the tool, especially if the tool takes a longer time to run.

1. To stream **arbitrary** data from inside a tool you can use [`stream_mode="custom"`](../streaming#custom) and `get_stream_writer()`:

    ```python
    # highlight-next-line
    from langgraph.config import get_stream_writer
    
    def tool(tool_arg: str):
        writer = get_stream_writer()
        for chunk in custom_data_stream():
            # stream any arbitrary data
            # highlight-next-line
            writer(chunk)
        ...
    
    for chunk in graph.stream(
        inputs,
        # highlight-next-line
        stream_mode="custom"
    ):
        print(chunk)
    ```

2. To stream LLM tokens generated by a tool calling an LLM you can use [`stream_mode="messages"`](../streaming#messages):

    ```python
    from langgraph.graph import StateGraph, MessagesState
    from langchain_openai import ChatOpenAI
    
    model = ChatOpenAI()
    
    def tool(tool_arg: str):
        model.invoke(tool_arg)
        ...
    
    def call_tools(state: MessagesState):
        tool_call = get_tool_call(state)
        tool_result = tool(**tool_call["args"])
        ...
    
    graph = (
        StateGraph(MessagesState)
        .add_node(call_tools)
        ...
        .compile()
    
    for msg, metadata in graph.stream(
        inputs,
        # highlight-next-line
        stream_mode="messages"
    ):
        print(msg)
    ```

!!! note "Using without LangChain"

    If you need to stream data from inside tools **without using LangChain**, you can use [`stream_mode="custom"`](../streaming/#custom). Check out the [example below](#example-without-langchain) to learn more.

!!! warning "Async in Python < 3.11"
    
    When using Python < 3.11 with async code, please ensure you manually pass the `RunnableConfig` through to the chat model when invoking it like so: `model.ainvoke(..., config)`.
    The stream method collects all events from your nested code using a streaming tracer passed as a callback. In 3.11 and above, this is automatically handled via [contextvars](https://docs.python.org/3/library/contextvars.html); prior to 3.11, [asyncio's tasks](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task) lacked proper `contextvar` support, meaning that the callbacks will only propagate if you manually pass the config through. We do this in the `call_model` function below.

## Setup

First, let's install the required packages and set our API keys


```python
%%capture --no-stderr
%pip install -U langgraph langchain-openai
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

## Streaming custom data

We'll use a [prebuilt ReAct agent][langgraph.prebuilt.chat_agent_executor.create_react_agent] for this guide:


```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from langgraph.config import get_stream_writer


@tool
async def get_items(place: str) -> str:
    """Use this tool to list items one might find in a place you're asked about."""
    # highlight-next-line
    writer = get_stream_writer()

    # this can be replaced with any actual streaming logic that you might have
    items = ["books", "penciles", "pictures"]
    for chunk in items:
        # highlight-next-line
        writer({"custom_tool_data": chunk})

    return ", ".join(items)


llm = ChatOpenAI(model_name="gpt-4o-mini")
tools = [get_items]
# contains `agent` (tool-calling LLM) and `tools` (tool executor) nodes
agent = create_react_agent(llm, tools=tools)
```

Let's now invoke our agent with an input that requires a tool call:


```python
inputs = {
    "messages": [  # noqa
        {"role": "user", "content": "what items are in the office?"}
    ]
}
async for chunk in agent.astream(
    inputs,
    # highlight-next-line
    stream_mode="custom",
):
    print(chunk)
```

## Streaming LLM tokens


```python
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig


@tool
async def get_items(
    place: str,
    # Manually accept config (needed for Python <= 3.10)
    # highlight-next-line
    config: RunnableConfig,
) -> str:
    """Use this tool to list items one might find in a place you're asked about."""
    # Attention: when using async, you should be invoking the LLM using ainvoke!
    # If you fail to do so, streaming will NOT work.
    response = await llm.ainvoke(
        [
            {
                "role": "user",
                "content": (
                    f"Can you tell me what kind of items i might find in the following place: '{place}'. "
                    "List at least 3 such items separating them by a comma. And include a brief description of each item."
                ),
            }
        ],
        # highlight-next-line
        config,
    )
    return response.content


tools = [get_items]
# contains `agent` (tool-calling LLM) and `tools` (tool executor) nodes
agent = create_react_agent(llm, tools=tools)
```


```python
inputs = {
    "messages": [  # noqa
        {"role": "user", "content": "what items are in the bedroom?"}
    ]
}
async for msg, metadata in agent.astream(
    inputs,
    # highlight-next-line
    stream_mode="messages",
):
    if (
        isinstance(msg, AIMessageChunk)
        and msg.content
        # Stream all messages from the tool node
        # highlight-next-line
        and metadata["langgraph_node"] == "tools"
    ):
        print(msg.content, end="|", flush=True)
```

## Example without LangChain

You can also stream data from within tool invocations **without using LangChain**. Below example demonstrates how to do it for a graph with a single tool-executing node. We'll leave it as an exercise for the reader to [implement ReAct agent from scratch](../react-agent-from-scratch) without using LangChain.


```python
import operator
import json

from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START

from openai import AsyncOpenAI

openai_client = AsyncOpenAI()
model_name = "gpt-4o-mini"


async def stream_tokens(model_name: str, messages: list[dict]):
    response = await openai_client.chat.completions.create(
        messages=messages, model=model_name, stream=True
    )
    role = None
    async for chunk in response:
        delta = chunk.choices[0].delta

        if delta.role is not None:
            role = delta.role

        if delta.content:
            yield {"role": role, "content": delta.content}


# this is our tool
async def get_items(place: str) -> str:
    """Use this tool to list items one might find in a place you're asked about."""
    # highlight-next-line
    writer = get_stream_writer()
    response = ""
    async for msg_chunk in stream_tokens(
        model_name,
        [
            {
                "role": "user",
                "content": (
                    "Can you tell me what kind of items "
                    f"i might find in the following place: '{place}'. "
                    "List at least 3 such items separating them by a comma. "
                    "And include a brief description of each item."
                ),
            }
        ],
    ):
        response += msg_chunk["content"]
        # highlight-next-line
        writer(msg_chunk)

    return response


class State(TypedDict):
    messages: Annotated[list[dict], operator.add]


# this is the tool-calling graph node
async def call_tool(state: State):
    ai_message = state["messages"][-1]
    tool_call = ai_message["tool_calls"][-1]

    function_name = tool_call["function"]["name"]
    if function_name != "get_items":
        raise ValueError(f"Tool {function_name} not supported")

    function_arguments = tool_call["function"]["arguments"]
    arguments = json.loads(function_arguments)

    function_response = await get_items(**arguments)
    tool_message = {
        "tool_call_id": tool_call["id"],
        "role": "tool",
        "name": function_name,
        "content": function_response,
    }
    return {"messages": [tool_message]}


graph = (
    StateGraph(State)  # noqa
    .add_node(call_tool)
    .add_edge(START, "call_tool")
    .compile()
)
```

Let's now invoke our graph with an AI message that contains a tool call:


```python
inputs = {
    "messages": [
        {
            "content": None,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "arguments": '{"place":"bedroom"}',
                        "name": "get_items",
                    },
                    "type": "function",
                }
            ],
        }
    ]
}

async for chunk in graph.astream(
    inputs,
    # highlight-next-line
    stream_mode="custom",
):
    print(chunk["content"], end="|", flush=True)
```

# How to stream from subgraphs

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Subgraphs](../../concepts/low_level/#subgraphs)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

If you have created a graph with [subgraphs](../subgraph), you may wish to stream outputs from those subgraphs. To do so, you can specify `subgraphs=True` in parent graph's `.stream()` method:


```python
for chunk in parent_graph.stream(
    {"foo": "foo"},
    # highlight-next-line
    subgraphs=True
):
    print(chunk)
```

## Setup

First let's install the required packages


```python
%%capture --no-stderr
%pip install -U langgraph
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Example

Let's define a simple example:


```python
from langgraph.graph import START, StateGraph
from typing import TypedDict


# Define subgraph
class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str


def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}


def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()


# Define parent graph
class ParentState(TypedDict):
    foo: str


def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}


builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()
```

Let's now stream the outputs from the graph:


```python
for chunk in graph.stream({"foo": "foo"}, stream_mode="updates"):
    print(chunk)
```

You can see that we're only emitting the updates from the parent graph nodes (`node_1` and `node_2`). To emit the updates from the _subgraph_ nodes you can specify `subgraphs=True`:


```python
for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    # highlight-next-line
    subgraphs=True,
):
    print(chunk)
```

Voila! The streamed outputs now contain updates from both the parent graph and the subgraph. **Note** that we are receiving not just the node updates, but we also the namespaces which tell us what graph (or subgraph) we are streaming from.
