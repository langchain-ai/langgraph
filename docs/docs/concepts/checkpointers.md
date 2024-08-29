# Checkpointer

LangGraph has a built-in persistence layer, implemented through [checkpointers][basecheckpointsaver]. When you use a checkpointer with a graph, you can interact with and manage the graph's state after the execution. The checkpointer saves a _checkpoint_, or a snapshot of the graph state at every super-step, enabling several powerful capabilities.

### Capabilities enabled by checkpointers

## Memory

## Human-in-the-loop

## Fault-tolerance

## Threads

Threads enable the checkpointing of multiple different runs, making them essential for multi-tenant chat applications and other scenarios where maintaining separate states is necessary. A thread is a unique ID assigned to a series of checkpoints saved by a checkpointer. When using a checkpointer, you must specify a `thread_id` and optionally a `checkpoint_id` when running the graph.

* `thread_id` is simply the ID of a thread. This is always required
* `checkpoint_id` can optionally be passed. This identifier refers to a specific checkpoint within a thread. This can be used to kick of a run of a graph from some point halfway through a thread.

You must pass these when invoking the graph as part of the `configurable` portion of the config, e.g.

```python
# {"configurable": {"thread_id": "1"}}  # valid config
# {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}  # also valid config

config = {"configurable": {"thread_id": "1"}}
graph.invoke(inputs, config=config)
```

See [this guide](../how-tos/persistence.ipynb) for how to use threads.

## Checkpointer state

When interacting with the checkpointer state, you **must** specify a [thread identifier](#threads) and optionally a checkpoint ID. Each checkpoint saved by the checkpointer has two properties:

- **values**: This is the value of the state at this point in time.
- **next**: This is a tuple of the nodes to execute next in the graph.
- **tasks**: This is a tuple of `PregelTask` objects that contain information about next tasks to be executed (upon interrupt/error). Will include error information, as well as additional data associated with interrupts, if interrupt was [dynamically triggered](../how-tos/human_in_the_loop/dynamic_breakpoints.ipynb) from within a node.

### Get state

You can get the state of a checkpointer by calling `graph.get_state(config)`. This will return a `StateSnapshot` object that corresponds to the latest checkpoint associated with the provided thread ID, or a checkpoint associated with a checkpoint ID for the thread, if provided.

```python
# get the latest state snapshot
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

# get a state snapshot for a specific checkpoint_id
config = {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}
graph.get_state(config)
```

### Get state history

You can get the full history of the graph execution for a given thread by calling `graph.get_state_history(config)`. This will 

### Update state

You can also interact with the state directly and update it. This takes three different components:

- config
- values
- `as_node`

**config**

The config should contain `thread_id` specifying which thread to update.

**values**

These are the values that will be used to update the state. Note that this update is treated exactly as any update from a node is treated. This means that these values will be passed to the [reducer](#reducers) functions that are part of the state. So this does NOT automatically overwrite the state. Let's walk through an example.

Let's assume you have defined the state of your graph as:

```python
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

Let's now assume the current state of the graph is

```
{"foo": 1, "bar": ["a"]}
```

If you update the state as below:

```
graph.update_state(config, {"foo": 2, "bar": ["b"]})
```

Then the new state of the graph will be:

```
{"foo": 2, "bar": ["a", "b"]}
```

The `foo` key is completely changed (because there is no reducer specified for that key, so it overwrites it). However, there is a reducer specified for the `bar` key, and so it appends `"b"` to the state of `bar`.

**`as_node`**

The final thing you specify when calling `update_state` is `as_node`. This update will be applied as if it came from node `as_node`. If `as_node` is not provided, it will be set to the last node that updated the state, if not ambiguous.

The reason this matters is that the next steps in the graph to execute depend on the last node to have given an update, so this can be used to control which node executes next.

## Serde

langgraph_checkpoint also defines [protocol][serializerprotocol] for serialization/deserialization (serde) and provides an default implementation ([JsonPlusSerializer][jsonplusserializer]) that handles a wide variety of types, including LangChain and LangGraph primitives, datetimes, enums and more.

## Pending writes

When a graph node fails mid-execution at a given superstep, LangGraph stores pending checkpoint writes from any other nodes that completed successfully at that superstep, so that whenever we resume graph execution from that superstep we don't re-run the successful nodes.

## Checkpointer libraries

LangGraph provides several checkpointer implementations, all implemented via standalone, installable libraries:

* `langgraph-checkpoint`: The base interface for checkpointer savers (BaseCheckpointSaver ) and serialization/deserialization interface (SerializationProtocol). Includes in-memory checkpointer implementation (MemorySaver) for experimentation.
* `langgraph-checkpoint-sqlite`: An implementation of LangGraph checkpointer that uses SQLite database. Ideal for experimentation and local workflows.
* `langgraph-checkpoint-postgres`: An advanced checkpointer that uses Postgres database, used in LangGraph Cloud. Ideal for using in production.

### Checkpointer interface

Each checkpointer conforms to [BaseCheckpointSaver][basecheckpointsaver] interface and implements the following methods:

`.put` - Store a checkpoint with its configuration and metadata.  
`.put_writes` - Store intermediate writes linked to a checkpoint (i.e. pending writes).  
`.get_tuple` - Fetch a checkpoint tuple using for a given configuration (`thread_id` and `checkpoint_id`). This is used to populate `StateSnapshot` in `graph.get_state()`.
`.list` - List checkpoints that match a given configuration and filter criteria. This is used to populate state history in `graph.get_state_history()`

If the checkpointer is used with asynchronous graph execution (i.e. executing the graph via `.ainvoke`, `.astream`, `.abatch`), asynchronous versions of the above methods will be used (`.aput`, `.aput_writes`, `.aget_tuple`, `.alist`).

!!! note Note
    For running your graph asynchronously, you can use [MemorySaver][memorysaver], or async versions of Sqlite/Postgres checkpointers -- [AsyncSqliteSaver][asyncsqlitesaver]/[AsyncPostgresSaver][asyncpostgressaver] checkpointers.

First, checkpointers facilitate [human-in-the-loop workflows](agentic_concepts.md#human-in-the-loop) workflows by allowing humans to inspect, interrupt, and approve steps.Checkpointers are needed for these workflows as the human has to be able to view the state of a graph at any point in time, and the graph has to be to resume execution after the human has made any updates to the state.

Second, it allows for ["memory"](agentic_concepts.md#memory) between interactions. You can use checkpointers to create threads and save the state of a thread after a graph executes. In the case of repeated human interactions (like conversations) any follow up messages can be sent to that checkpoint, which will retain its memory of previous ones.

See [this guide](../how-tos/persistence.ipynb) for how to add a checkpointer to your graph.