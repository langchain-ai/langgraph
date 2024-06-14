# API Concepts
This page discusses high-level concepts of the LangGraph Deploy API. The complete API specification is available locally at [`http://localhost:8123/docs`](http://localhost:8123/docs) or [here]().

## Assistant
An assistant is a configured instance of a [`CompiledGraph`](../../reference/graphs/#compiledgraph). It abstracts the cognitive architecture of the graph and contains instance specific configuration and metadata. Multiple assistants can reference the same graph but can contain different configuration and metadata, which may differentiate the behavior of the assistants.

An assistant (i.e. the graph) is invoked as part of a [run](#run).

## Thread
A thread contains the accumulated state of a group of [runs](#run). If a run is executed on a thread, then the [state](../../concepts/#state-management) of the underlying graph of the [assistant](#assistant) will be persisted to the thread. A thread's current and historical state can be retrieved.

To persist state, a thread must be created prior to executing a run.

## Run
A run is an invocation of an [assistant](#assistant). Each run may have its own input, configuration, and metadata, which may affect execution and output of the underlying graph. A run can optionally be executed on a [thread](#thread).

### Streaming

### Human-in-the-Loop

### Multi-Tasking

### Webhooks
