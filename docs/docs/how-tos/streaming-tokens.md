# How to stream LLM tokens from your graph

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Streaming](../../concepts/streaming/.md)
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

    If you need to stream LLM tokens **without using LangChain**, you can use [`stream_mode="custom"`](../streaming/#custom.md) to stream the outputs from LLM provider clients directly. Check out the [example below](#example-without-langchain.md) to learn more.

!!! warning "Async in Python < 3.11"
    
    When using Python < 3.11 with async code, please ensure you manually pass the `RunnableConfig` through to the chat model when invoking it like so: `model.ainvoke(..., config)`.
    The stream method collects all events from your nested code using a streaming tracer passed as a callback. In 3.11 and above, this is automatically handled via [contextvars](https://docs.python.org/3/library/contextvars.html); prior to 3.11, [asyncio's tasks](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task) lacked proper `contextvar` support, meaning that the callbacks will only propagate if you manually pass the config through. We do this in the `call_model` function below.

## Setup

First we need to install the packages required


```
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

    When streaming LLM tokens without LangChain, we recommend using [`stream_mode="custom"`](../streaming/#stream-modecustom.md). This allows you to explicitly control which data from the LLM provider APIs to include in LangGraph streamed outputs, including any additional metadata.


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
