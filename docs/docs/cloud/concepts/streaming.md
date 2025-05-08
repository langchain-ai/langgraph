# Streaming

Streaming is critical for making LLM applications feel responsive to end users.  
When creating a streaming run, the **streaming mode** determines what kinds of data are streamed back to the API client.

## Supported streaming modes

LangGraph Platform supports the following streaming modes:

| Mode                 | Description                                                                                                                                                    | LangGraph Library Method                                                |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **`values`**         | Stream the full graph state after each [super-step](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs). [Guide](../how-tos/streaming.md#stream-graph-state) | `.stream()` / `.astream()` with `stream_mode="values"`                  |
| **`updates`**        | Stream only the updates to the graph state after each node. [Guide](../how-tos/streaming.md#stream-graph-state)                                                              | `.stream()` / `.astream()` with `stream_mode="updates"`                 |
| **`messages-tuple`** | Stream LLM tokens for any messages generated inside the graph (useful for chat apps). [Guide](../how-tos/streaming.md#messages)                                   | `.stream()` / `.astream()` with `stream_mode="messages"`                |
| **`debug`**          | Stream debug information throughout graph execution. [Guide](../how-tos/streaming.md#debug)                                                                       | `.stream()` / `.astream()` with `stream_mode="debug"`                   |
| **`custom`**         | Stream custom data. [Guide](../../how-tos/streaming.md#stream-custom-data)                                                                                                                                             | `.stream()` / `.astream()` with `stream_mode="custom"`                  |
| **`events`**         | Stream all events (including the state of the graph); mainly useful when migrating large LCEL apps. [Guide](../how-tos/streaming.md#stream-events)                       | `.astream_events()`                                                     |

✅ You can also **combine multiple modes** at the same time.  See the [how-to guide](../how-tos/streaming.md#stream-multiple-modes) for configuration details.

## API Reference

For API usage and implementation, refer to the [API reference](../reference/api/api_ref.html#tag/thread-runs/POST/threads/{thread_id}/runs/stream). 

