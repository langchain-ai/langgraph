# Stream outputs

!!! tip "Set up [LangSmith](https://smith.langchain.com) for LangGraph Development"

    Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started [here](https://docs.smith.langchain.com).

## Streaming API

The entrypoint for streaming is the `stream` method (or the async `astream` method) on any LangGraph graph. This method returns an iterator that yields the streamed outputs from the graph.

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

### Stream multiple modes

You can also combine multiple streaming mode by providing a list to `stream_mode` parameter:

=== "Sync"

    ```python
    for mode, chunk in graph.stream(inputs, stream_mode=["updates", "custom"]):
        print(chunk)
    ```

=== "Async"

    ```python
    async for mode, chunk in graph.astream(inputs, stream_mode=["updates", "custom"]):
        print(chunk)
    ```

### Available modes

| Mode                                            | Description                                                                                                                                                                         |
|-------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`values`](../how-tos/streaming.md#values)      | Streams the full value of the state after each step of the graph.                                                                                                                   |
| [`updates`](../how-tos/streaming.md#updates)    | Streams the updates to the state after each step of the graph. If multiple updates are made in the same step (e.g., multiple nodes are run), those updates are streamed separately. |
| [`custom`](../how-tos/streaming.md#custom)      | Streams custom data from inside your graph nodes.                                                                                                                                   |
| [`messages`](../how-tos/streaming-tokens.ipynb) | Streams LLM tokens and metadata for the graph node where the LLM is invoked.                                                                                                        |
| [`debug`](../how-tos/streaming.md#debug)        | Streams as much information as possible throughout the execution of the graph.                                                                                                      |

## State 

You can use the stream modes `updates` and `values` to stream the state of the graph as it changes during execution.

* `updates` streams the **updates** to the state after each step of the graph.
* `values` streams the **full value** of the state after each step of the graph.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


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
  .add_edge("generate_joke", END)
  .compile()
)
```

=== "updates"

    Use this to stream only the **state updates** returned by the nodes after each step. The streamed outputs include the name of the node as well as the update.


    ```python
    for chunk in graph.stream(
        {"topic": "ice cream"},
        # highlight-next-line
        stream_mode="updates",
    ):
        print(chunk)
    ```

===  "values"

    Use this to stream the **full state** of the graph after each step.

    ```python
    for chunk in graph.stream(
        {"topic": "ice cream"},
        # highlight-next-line
        stream_mode="values",
    ):
        print(chunk)
    ```


## Subgraphs

If you have created a graph with [subgraphs](../subgraph), you may wish to stream outputs from those subgraphs. To do so, you can specify `subgraphs=True` in parent graph's `.stream()` method:

```python
for chunk in parent_graph.stream(
    {"foo": "foo"},
    # highlight-next-line
    subgraphs=True # (1)!
):
    print(chunk)
```

1. Set `subgraphs=True` to stream outputs from subgraphs.

??? example "Streaming from subgraphs"

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

      for chunk in graph.stream(
          {"foo": "foo"},
          stream_mode="updates",
          # highlight-next-line
          subgraphs=True, # (1)!
      ):
          print(chunk)
      ```
      
      1. Set `subgraphs=True` to stream outputs from subgraphs.


      Voila! The streamed outputs now contain updates from both the parent graph and the subgraph. **Note** that we are receiving not just the node updates, but we also the namespaces which tell us what graph (or subgraph) we are streaming from.

## Debugging {#debug}

Use the `debug` streaming mode to stream as much information as possible throughout the execution of the graph. The streamed outputs include the name of the node as well as the full state.

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    # highlight-next-line
    stream_mode="debug",
):
    print(chunk)
```


## LLM tokens {#messages}

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

Use the `messages` streaming mode to stream LLM messages **token-by-token** from **anywhere** in your graph. 

You can stream LLM tokens from:

* nodes
* tools
* subgraphs
* tasks


The streamed outputs will be tuples of `(message chunk, metadata)`:

* message chunk is the token streamed by the LLM
* metadata is a dictionary with information about the graph node where the LLM was called as well as the LLM invocation metadata

!!! tip "Using without LangChain"

    If you need to stream LLM tokens **without using LangChain**, you can use [`stream_mode="custom"`](../streaming/#custom) to stream the outputs from LLM provider clients directly. See [stream arbitrary chat models](#stream-arbitrary-chat-models) for details.

```python
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START


@dataclass
class MyState:
    topic: str
    joke: str = ""


llm = init_chat_model(model="openai:gpt-4o-mini")

def call_model(state: MyState):
    """Call the LLM to generate a joke about a topic"""
    # highlight-next-line
    llm_response = llm.invoke( # (1)!
        [
            {"role": "user", "content": f"Generate a joke about {state.topic}"}
        ]
    )
    return {"joke": llm_response.content}

graph = (
    StateGraph(MyState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

for message_chunk, metadata in graph.stream( # (2)!
    {"topic": "ice cream"},
    # highlight-next-line
    stream_mode="messages",
):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)
```

1. Note that even though the LLM is run using `invoke` rather than `stream`. When using the `messages` stream mode, LangGraph will 
2. The "messages" stream mode returns an iterator of tuples `(message_chunk, metadata)` where `message_chunk` is the token streamed by the LLM and `metadata` is a dictionary with information about the graph node where the LLM was called and other information.


### Filter by LLM invocation

You can associate `tags` with LLM invocations to filter the streamed tokens by LLM invocation.

```python
from typing import TypedDict

from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph

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


graph = (
      StateGraph(State)
      .add_node(call_model)
      .add_edge(START, "call_model")
      .compile()
)
```

You can see that we're streaming tokens from all of the LLM invocations. Let's now filter the streamed tokens to include only a specific LLM invocation. We can use the streamed metadata and filter events using the tags we've added to the LLMs previously:

```python
async for msg, metadata in graph.astream(
    {"topic": "cats"},
    # highlight-next-line
    stream_mode="messages",
):
    if msg.content:
        print(msg.content, end="|", flush=True)
```

### Filter by node

A common use case when [streaming LLM tokens](../streaming-tokens) is to only stream them from specific nodes. To do so, you can use `stream_mode="messages"` and filter the outputs by the `langgraph_node` field in the streamed metadata:

```python
# highlight-next-line
for msg, metadata in graph.stream( # (1)!
    {"topic": "cats"},
    stream_mode="messages",
):
    # highlight-next-line
    if msg.content and metadata["langgraph_node"] == "write_poem": # (2)!
        print(msg.content, end="|", flush=True)
```

1. The "messages" stream mode returns a tuple of `(message_chunk, metadata)` where `message_chunk` is the token streamed by the LLM and `metadata` is a dictionary with information about the graph node where the LLM was called and other information.
2. Filter the streamed tokens by the `langgraph_node` field in the metadata to only include the tokens from the `write_poem` node.

??? example "Streaming LLM tokens from specific nodes"

      ```python
      from typing import TypedDict
      from langgraph.graph import START, StateGraph 
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

      # highlight-next-line
      for msg, metadata in graph.stream( # (1)!
          {"topic": "cats"},
          stream_mode="messages",
      ):
          # highlight-next-line
          if msg.content and metadata["langgraph_node"] == "write_poem": # (2)!
              print(msg.content, end="|", flush=True)
      ```

      1. The "messages" stream mode returns a tuple of `(message_chunk, metadata)` where `message_chunk` is the token streamed by the LLM and `metadata` is a dictionary with information about the graph node where the LLM was called and other information.
      2. Filter the streamed tokens by the `langgraph_node` field in the metadata to only include the tokens from the `write_poem` node.

## Stream custom data

Sometimes you need to send **custom, user-defined data** from inside a LangGraph node or tool during execution. This is useful when you want to surface intermediate results, log details, or push any non-standard outputs as part of the streaming flow.

To stream custom data, you need to:

1. **Write custom data using the stream writer** — use `get_stream_writer()` to access the writer.
2. **Set `stream_mode="custom"` when calling `.stream()` or `.astream()`** — this ensures the graph routes your custom data to the stream. You can also combine multiple streaming modes (e.g., `["updates", "custom"]`), but at least one of them must be `"custom"` for your custom data to appear.

Below are examples showing how to use this inside both **nodes** and **tools**.


!!! warning "Async with Python < 3.11"

      If you are using Python < 3.11 and are running LangGraph asynchronously,
      `get_stream_writer()` won't work since it uses [contextvar](https://docs.python.org/3/library/contextvars.html) propagation (only available in [Python >= 3.11](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)).

      Instead of using `get_stream_writer()`, you should include `writer` in the function signature of your node or tool, and pass it in when invoking the function.

### From a node

```python
from langgraph.config import get_stream_writer

# Example node function
def node(state):
    writer = get_stream_writer()
    # Stream a custom key-value pair
    writer({"custom_key": "Generating custom data inside node"})
    ...
    # do some processing


# Define a graph that uses this node

# Usage
for chunk in graph.stream(inputs, stream_mode="custom"):
    print(chunk)
```

### From a tool

```python
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

@tool
def query_database(query: str) -> str:
    """Query the database."""
    writer = get_stream_writer() # (1)!
    # highlight-next-line
    writer({"data": "Retrieved 0/100 records", "type": "progress"}) # (2)!
    # Do some work like fetching data from a database
    # ...
    # ...
    # highlight-next-line
    writer({"data": "Retrieved 100/100 records", "type": "progress"}) # (3)!
    return "some-answer" 


graph = ... # define a graph that uses this tool

for chunk in graph.stream(inputs, stream_mode="custom"): # (4)!
    print(chunk)
```

1. Get the stream writer to send custom data.
2. Stream a custom key-value pair with progress information.
3. Stream another custom key-value pair with progress information.
4. Set `stream_mode="custom"` to receive the custom data in the stream.

## Use with any LLM

You can use `stream_mode="custom"` to stream data from **any LLM API**  — even if that API does **not** implement the LangChain chat model interface.

This lets you integrate raw LLM clients or external services that provide their own streaming interfaces, making LangGraph highly flexible for custom setups.

```python
from langgraph.config import get_stream_writer

def call_arbitrary_model(state):
    """Example node that calls an arbitrary model and streams the output"""
    # highlight-next-line
    writer = get_stream_writer() # (1)!
    # Assume you have a streaming client that yields chunks
    for chunk in your_custom_streaming_client(state["topic"]): # (2)!
        # highlight-next-line
        writer({"custom_llm_chunk": chunk}) # (3)!
    return {"result": "completed"}

graph = (
    StateGraph(State)
    .add_node(call_arbitrary_model)
    # Add other nodes and edges as needed
    .compile()
)

for chunk in graph.stream(
    {"topic": "cats"},
    # highlight-next-line
    stream_mode="custom", # (4)!
):
    # The chunk will contain the custom data streamed from the llm
    print(chunk)
```

1. Get the stream writer to send custom data.
2. Generate LLM tokens using your custom streaming client.
3. Use the writer to send custom data to the stream.
4. Set `stream_mode="custom"` to receive the custom data in the stream.


??? example "Streaming arbitrary chat model"
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
          StateGraph(State)  
          .add_node(call_tool)
          .add_edge(START, "call_tool")
          .compile()
      )
      ```

      Let's invoke the graph with an AI message that includes a tool call:

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
          stream_mode="custom",
      ):
          print(chunk["content"], end="|", flush=True)
      ```


## Disable streaming for specific chat models

Some chat models, including the new O1 models from OpenAI (depending on when you're reading this), do not support streaming.

If you're using the `stream` API while also using a chat model that doesn't support streaming, you may encounter issues. For example,

this could happen if you have an LLM application that leverages different chat models, some of which support streaming and others that do not. In such cases, you may want to disable streaming for specific chat models that do not support streaming.

To disable streaming for specific chat models, you can set `disable_streaming=True` when initializing the model:

=== "init_chat_model"
      
      ```python
      from langchain.chat_models import init_chat_model

      model = init_chat_model(
            "anthropic:claude-3-7-sonnet-latest",
            # highlight-next-line
            disable_streaming=True # (1)!
      )
      ```

      1. Set `disable_streaming=True` to disable streaming for the chat model.

=== "ChatModel interface"

      ```python
      from langchain_openai import ChatOpenAI

      llm = ChatOpenAI(model="o1-preview", disable_streaming=True) # (1)!
      ```

      1. Set `disable_streaming=True` to disable streaming for the chat model.




## Async

When using Python < 3.11 with async code, please ensure you manually pass the `RunnableConfig` through to the chat model when invoking it like so: `model.ainvoke(..., config)`.
The stream method collects all events from your nested code using a streaming tracer passed as a callback. In 3.11 and above, this is automatically handled via [contextvars](https://docs.python.org/3/library/contextvars.html); prior to 3.11, [asyncio's tasks](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task) lacked proper `contextvar` support, meaning that the callbacks will only propagate if you manually pass the config through. We do this in the `call_model` function below.

!!! warning "Async in Python < 3.11"

    When using Python < 3.11 with async code, please ensure you manually pass the `RunnableConfig` through to the chat model when invoking it like so: `model.ainvoke(..., config)`.
    The stream method collects all events from your nested code using a streaming tracer passed as a callback. In 3.11 and above, this is automatically handled via [contextvars](https://docs.python.org/3/library/contextvars.html); prior to 3.11, [asyncio's tasks](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task) lacked proper `contextvar` support, meaning that the callbacks will only propagate if you manually pass the config through. We do this in the `call_model` function below.

!!! note Manual Callback Propagation

    Note that in `call_model(state: State, config: RunnableConfig):` below, we a) accept the [`RunnableConfig`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig) in the node function and b) pass it in as the second arg for `model.ainvoke(..., config)`. This is optional for python >= 3.11.

```python
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

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

inputs = {
    "messages": [  
        {"role": "user", "content": "what items are in the bedroom?"}
    ]
}
async for msg, metadata in agent.astream(
    inputs,
    stream_mode="messages",
):
    if (
        isinstance(msg, AIMessageChunk)
        and msg.content
        # Stream all messages from the tool node
        and metadata["langgraph_node"] == "tools"
    ):
        print(msg.content, end="|", flush=True)
```
