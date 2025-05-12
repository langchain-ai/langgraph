# Threads

A thread contains the accumulated state of a sequence of [runs](./runs.md). If a run is executed on a thread, then the [state](../../concepts/low_level.md#state) of the underlying graph of the assistant will be persisted to the thread.

A thread's current and historical state can be retrieved. To persist state, a thread must be created prior to executing a run.

The state of a thread at a particular point in time is called a [checkpoint](../../concepts/persistence.md#checkpoints). Checkpoints can be used to restore the state of a thread at a later time.

For more on threads and checkpoints, see this section of the [LangGraph conceptual guide](../../concepts/persistence.md).

The LangGraph Cloud API provides several endpoints for creating and managing threads and thread state. See the [API reference](../../cloud/reference/api/api_ref.html#tag/threads) for more details.