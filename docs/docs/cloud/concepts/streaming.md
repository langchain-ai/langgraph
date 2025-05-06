# Streaming

Streaming is critical for making LLM applications feel responsive to end users.  
When creating a streaming run, the **streaming mode** determines what kinds of data are streamed back to the API client.

## Supported streaming modes

LangGraph Cloud supports five streaming modes:

| Mode                 | Description                                                                                                                                                    |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`values`**         | Stream the full graph state after each [super-step](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs). [Guide](../how-tos/stream_values.md) |
| **`updates`**        | Stream only the updates to the graph state after each node. [Guide](../how-tos/stream_updates.md)                                                              |
| **`messages-tuple`** | Stream LLM tokens for any messages generated inside the graph (useful for chat apps). [Guide](../how-tos/stream_messages.md)                                   |
| **`debug`**          | Stream debug information throughout graph execution. [Guide](../how-tos/stream_debug.md)                                                                       |
| **`custom`**         | Stream custom data                                                                                                                                            |
| **`events`**         | Stream all events (including the state of the graph); mainly useful when migrating large LCEL apps. [Guide](../how-tos/stream_events.md)                       |

✅ You can also **combine multiple modes** at the same time.  See the [how-to guide](../how-tos/stream_multiple.md) for configuration details.

## Mapping to LangGraph library

- The `values`, `updates`, `messages-tuple`, and `debug` modes map to `.stream()` / `.astream()` in the LangGraph Python library.
- The `events` mode corresponds to using `.astream_events()` in the LangGraph library.

## API Reference

For API usage and implementation, refer to the [API reference](../reference/api/api_ref.html#tag/thread-runs/POST/threads/{thread_id}/runs/stream). 

