# API Concepts
This page describes the high-level concepts of the LangGraph Cloud API. The conceptual guide of LangGraph (Python library) is [here](../../concepts/index.md).

## Models
The LangGraph Cloud API consists of 3 core models: [Assistants](#assistants), [Threads](#threads), and [Runs](#runs).

### Assistants
An assistant is a configured instance of a [`CompiledGraph`](../../../reference/graphs/#compiledgraph). It abstracts the cognitive architecture of the graph and contains instance specific configuration and metadata. Multiple assistants can reference the same graph but can contain different configuration and metadata, which may differentiate the behavior of the assistants. An assistant (i.e. the graph) is invoked as part of a run.

The LangGraph Cloud API provides several endpoints for creating and managing assistants. See the [API reference](../reference/api_ref.md) for more details.

### Threads
A thread contains the accumulated state of a group of runs. If a run is executed on a thread, then the [state](../../../concepts/#persistence) of the underlying graph of the assistant will be persisted to the thread. A thread's current and historical state can be retrieved. To persist state, a thread must be created prior to executing a run.

The LangGraph Cloud API provides several endpoints for creating and managing threads and thread state. See the [API reference](../reference/api_ref.md) for more details.

### Runs
A run is an invocation of an assistant. Each run may have its own input, configuration, and metadata, which may affect execution and output of the underlying graph. A run can optionally be executed on a thread.

The LangGraph Cloud API provides several endpoints for creating and managing runs. See the [API reference](../reference/api_ref.md) for more details.

## Features
The LangGraph Cloud API offers several features to support complex agent architectures.

### Streaming
Streaming is critical for making LLM applications feel responsive to end users. When creating a streaming run, the streaming mode determines what data is streamed back to the API client. The LangGraph Cloud API supports five streaming modes. 

- **values**: Stream the full state of the graph after each node is executed. See the [How-to Guide](../../how-tos/cloud_examples/stream_values/) for streaming values.
- **messages**: Stream complete messages (at the end of node execution) as well as tokens for any messages generated inside a node. This mode is primarily meant for powering chat applications. See the [How-to Guide](../../how-tos/cloud_examples/stream_messages/) for streaming messages.
- **updates**: Streams updates to the state of the graph after each node is executed. See the [How-to Guide](../../how-tos/cloud_examples/stream_updates/) for streaming updates.
- **events**: Stream all events (including the state of the graph) after each node is executed. See the [How-to Guide](../../how-tos/cloud_examples/stream_events/) for streaming events.
- **debug**: Stream debug events after each node is executed. See the [How-to Guide](../../how-tos/cloud_examples/stream_debug/) for streaming debug events.

See the [API Reference](../reference/api_ref.md) for how to create streaming runs.

### Human-in-the-Loop
There are many occasions where the graph cannot run completely autonomously. For instance, the user might need to input some additional arguments to a function call, or select the next edge for the graph to continue on. In these instances, we need to insert some human in the loop interaction, which you can learn about in [this how-to](../how_tos/cloud_examples/human-in-the-loop_cloud).

### Multi-Tasking
Many times users might interact with your graph in unintended ways. For instance, a user interacting with a graph that has chat output could send one message and before the graph has finished running send a second message. To solve this issue of "double-texting" (i.e. prompting the graph a second time before the first run has finished), Langgraph has provided four different solutions, all of which are covered in the [Double Texting how-tos](../how_tos/cloud_examples/interrupt_concurrent/).
