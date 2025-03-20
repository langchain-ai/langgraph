# Streaming

Building a responsive app for end-users? Real-time updates are key to keeping users engaged as your app progresses.

There are three main types of data you’ll want to stream:

1. Workflow progress (e.g., get state updates after each graph node is executed).
2. LLM tokens as they’re generated.
3. Custom updates (e.g., "Fetched 10/100 records").

## Streaming graph outputs (`.stream` and `.astream`)

`.stream` and `.astream` are sync and async methods for streaming back outputs from a graph run.
There are several different modes you can specify when calling these methods (e.g. `graph.stream(..., mode="...")):

- [`"values"`](../how-tos/streaming.ipynb#values): This streams the full value of the state after each step of the graph.
- [`"updates"`](../how-tos/streaming.ipynb#updates): This streams the updates to the state after each step of the graph. If multiple updates are made in the same step (e.g. multiple nodes are run) then those updates are streamed separately.
- [`"custom"`](../how-tos/streaming.ipynb#custom): This streams custom data from inside your graph nodes.
- [`"messages"`](../how-tos/streaming-tokens.ipynb): This streams LLM tokens and metadata for the graph node where LLM is invoked.
- [`"debug"`](../how-tos/streaming.ipynb#debug): This streams as much information as possible throughout the execution of the graph.

You can also specify multiple streaming modes at the same time by passing them as a list. When you do this, the streamed outputs will be tuples `(stream_mode, data)`. For example:

```python
graph.stream(..., stream_mode=["updates", "messages"])
```

```
...
('messages', (AIMessageChunk(content='Hi'), {'langgraph_step': 3, 'langgraph_node': 'agent', ...}))
...
('updates', {'agent': {'messages': [AIMessage(content="Hi, how can I help you?")]}})
```

The below visualization shows the difference between the `values` and `updates` modes:

![values vs updates](../static/values_vs_updates.png)


## LangGraph Platform

Streaming is critical for making LLM applications feel responsive to end users. When creating a streaming run, the streaming mode determines what data is streamed back to the API client. LangGraph Platform supports five streaming modes:

- `values`: Stream the full state of the graph after each [super-step](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs) is executed. See the [how-to guide](../cloud/how-tos/stream_values.md) for streaming values.
- `messages-tuple`: Stream LLM tokens for any messages generated inside a node. This mode is primarily meant for powering chat applications. See the [how-to guide](../cloud/how-tos/stream_messages.md) for streaming messages.
- `updates`: Streams updates to the state of the graph after each node is executed. See the [how-to guide](../cloud/how-tos/stream_updates.md) for streaming updates.
- `debug`: Stream debug events throughout graph execution. See the [how-to guide](../cloud/how-tos/stream_debug.md) for streaming debug events.
- `events`: Stream all events (including the state of the graph) that occur during graph execution. See the [how-to guide](../cloud/how-tos/stream_events.md) for streaming events. This mode is only useful for users migrating large LCEL applications to LangGraph. Generally, this mode is not necessary for most applications.

You can also specify multiple streaming modes at the same time. See the [how-to guide](../cloud/how-tos/stream_multiple.md) for configuring multiple streaming modes at the same time.

See the [API reference](../cloud/reference/api/api_ref.html#tag/threads-runs/POST/threads/{thread_id}/runs/stream) for how to create streaming runs.

Streaming modes `values`, `updates`, `messages-tuple` and `debug` are very similar to modes available in the LangGraph library - for a deeper conceptual explanation of those, you can see the [previous section](#streaming-graph-outputs-stream-and-astream).

Streaming mode `events` is the same as using `.astream_events` in the LangGraph library - for a deeper conceptual explanation of this, you can see the [previous section](#streaming-graph-outputs-stream-and-astream).

All events emitted have two attributes:

- `event`: This is the name of the event
- `data`: This is data associated with the event