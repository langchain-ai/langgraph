# LangGraph Checkpoint

This library defines the base interface for LangGraph checkpointers. Checkpointers provide a persistence layer for LangGraph. They allow you to interact with and manage the graph's state. When you use a graph with a checkpointer, the checkpointer saves a _checkpoint_ of the graph state at every superstep, enabling several powerful capabilities like human-in-the-loop, "memory" between interactions and more.

## Key concepts

### Checkpoint

Checkpoint is a snapshot of the graph state at a given point in time. Checkpoint tuple refers to an object containing checkpoint and the associated config, metadata and pending writes.

### Thread

Threads enable the checkpointing of multiple different runs, making them essential for multi-tenant chat applications and other scenarios where maintaining separate states is necessary. A thread is a unique ID assigned to a series of checkpoints saved by a checkpointer. When using a checkpointer, you must specify a `thread_id` and optionally `checkpoint_id` when running the graph.

- `thread_id` is simply the ID of a thread. This is always required.
- `checkpoint_id` can optionally be passed. This identifier refers to a specific checkpoint within a thread. This can be used to kick off a run of a graph from some point halfway through a thread.

You must pass these when invoking the graph as part of the configurable part of the config, e.g.

```python
{"configurable": {"thread_id": "1"}}  # valid config
{"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}  # also valid config
```

### Serde

`langgraph_checkpoint` also defines protocol for serialization/deserialization (serde) and provides an default implementation (`langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer`) that handles a wide variety of types, including LangChain and LangGraph primitives, datetimes, enums and more.

### Pending writes

When a graph node fails mid-execution at a given superstep, LangGraph stores pending checkpoint writes from any other nodes that completed successfully at that superstep, so that whenever we resume graph execution from that superstep we don't re-run the successful nodes.

## Interface

Each checkpointer should conform to `langgraph.checkpoint.base.BaseCheckpointSaver` interface and must implement the following methods:

- `.put` - Store a checkpoint with its configuration and metadata.
- `.put_writes` - Store intermediate writes linked to a checkpoint (i.e. pending writes).
- `.get_tuple` - Fetch a checkpoint tuple using for a given configuration (`thread_id` and `checkpoint_id`).
- `.list` - List checkpoints that match a given configuration and filter criteria.
- `.delete_thread()` - Delete all checkpoints and writes associated with a thread.
- `.get_next_version()` - Generate the next version ID for a channel.

If the checkpointer will be used with asynchronous graph execution (i.e. executing the graph via `.ainvoke`, `.astream`, `.abatch`), checkpointer must implement asynchronous versions of the above methods (`.aput`, `.aput_writes`, `.aget_tuple`, `.alist`). Similarly, the checkpointer must implement `.adelete_thread()` if asynchronous thread cleanup is desired. The base class provides a default implementation of `.get_next_version()` that generates an integer sequence starting from 1, but this method should be overridden for custom versioning schemes.

## Usage

```python
from langgraph.checkpoint.memory import InMemorySaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

checkpointer = InMemorySaver()
checkpoint = {
    "v": 4,
    "ts": "2024-07-31T20:14:19.804150+00:00",
    "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
    "channel_values": {
      "my_key": "meow",
      "node": "node"
    },
    "channel_versions": {
      "__start__": 2,
      "my_key": 3,
      "start:node": 3,
      "node": 3
    },
    "versions_seen": {
      "__input__": {},
      "__start__": {
        "__start__": 1
      },
      "node": {
        "start:node": 2
      }
    },
}

# store checkpoint
checkpointer.put(write_config, checkpoint, {}, {})

# load checkpoint
checkpointer.get(read_config)

# list checkpoints
list(checkpointer.list(read_config))
```
