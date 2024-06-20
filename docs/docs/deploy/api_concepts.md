# API Concepts
This page discusses high-level concepts of the LangGraph Cloud.

## Assistant
An assistant is a configured instance of a [`CompiledGraph`](../../reference/graphs/#compiledgraph). It abstracts the cognitive architecture of the graph and contains instance specific configuration and metadata. Multiple assistants can reference the same graph but can contain different configuration and metadata, which may differentiate the behavior of the assistants.

An assistant (i.e. the graph) is invoked as part of a [run](#run).

## Thread
A thread contains the accumulated state of a group of [runs](#run). If a run is executed on a thread, then the [state](../../concepts/#state-management) of the underlying graph of the [assistant](#assistant) will be persisted to the thread. A thread's current and historical state can be retrieved.

To persist state, a thread must be created prior to executing a run.

## Run
A run is an invocation of an [assistant](#assistant). Each run may have its own input, configuration, and metadata, which may affect execution and output of the underlying graph. A run can optionally be executed on a [thread](#thread).

## Streaming
Streaming is critical in making applications based on LLMs feel responsive to end-users. There are three different ways to stream with graphs: by [values](../how_tos/cloud_examples/stream_values/), by [messages](../how_tos/cloud_examples/stream_messages/), and by [updates](../how_tos/cloud_examples/stream_updates/).

## Human-in-the-Loop
There are many occasions where the graph cannot run completely autonomously. For instance, the user might need to input some additional arguments to a function call, or select the next edge for the graph to continue on. In these instances, we need to insert some human in the loop interaction, which you can learn about in [this how-to](../how_tos/cloud_examples/human-in-the-loop_cloud).

## Multi-Tasking
Many times users might interact with your graph in unintended ways. For instance, a user interacting with a graph that has chat output could send one message and before the graph has finished running send a second message. To solve this issue of "double-texting" (i.e. prompting the graph a second time before the first run has finished), Langgraph has provided four different solutions, all of which are covered in the [Double Texting how-tos](../how_tos/cloud_examples/interrupt_concurrent/).
