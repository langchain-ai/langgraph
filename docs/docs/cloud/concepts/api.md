# API Concepts

This page describes the high-level concepts of the LangGraph Cloud API. The conceptual guide of LangGraph (Python library) is [here](../../concepts/index.md).

## Data Models

The LangGraph Cloud API consists of a few core data models: [Assistants](#assistants), [Threads](#threads), [Runs](#runs), and [Cron Jobs](#cron-jobs).

### Assistants

An assistant is a configured instance of a [`CompiledGraph`][compiledgraph]. It abstracts the cognitive architecture of the graph and contains instance specific configuration and metadata. Multiple assistants can reference the same graph but can contain different configuration and metadata, which may differentiate the behavior of the assistants. An assistant (i.e. the graph) is invoked as part of a run.

The LangGraph Cloud API provides several endpoints for creating and managing assistants. See the <a href="../reference/api/api_ref.html#tag/assistantscreate" target="_blank">API reference</a> for more details.

#### Configuring Assistants

One of the benefits of LangGraph API is that it lets you create agents with different configurations. This is useful when you want to allow users to use different models or prompts when using your graph. This also allows you the ability to quickly test out different configurations without having to rewrite your graph every time. See <a href="https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/configuration_cloud/">this</a> how-to for information on how to configure a deployed graph. 

### Threads

A thread contains the accumulated state of a group of runs. If a run is executed on a thread, then the [state][state] of the underlying graph of the assistant will be persisted to the thread. A thread's current and historical state can be retrieved. To persist state, a thread must be created prior to executing a run.

The state of a thread at a particular point in time is called a checkpoint.

For more on threads and checkpoints, see this section of the [LangGraph conceptual guide](../../concepts/low_level.md#checkpointer).

The LangGraph Cloud API provides several endpoints for creating and managing threads and thread state. See the <a href="../reference/api/api_ref.html#tag/threadscreate" target="_blank">API reference</a> for more details.

### Runs

A run is an invocation of an assistant. Each run may have its own input, configuration, and metadata, which may affect execution and output of the underlying graph. A run can optionally be executed on a thread.

The LangGraph Cloud API provides several endpoints for creating and managing runs. See the <a href="../reference/api/api_ref.html#tag/runscreate" target="_blank">API reference</a> for more details.

### Cron Jobs

It's often useful to run graphs on some schedule. LangGraph Cloud supports cron jobs, which run on a user defined schedule. The user specifies a schedule, an assistant, and some input. After than, on the specified schedule LangGraph cloud will:

- Create a new thread with the specified assistant
- Send the specified input to that thread

Note that this sends the same input to the thread every time. See the [how-to guide](../how-tos/cloud_examples/cron_jobs.ipynb) for creating cron jobs.

The LangGraph Cloud API provides several endpoints for creating and managing cron jobs. See the <a href="../reference/api/api_ref.html#tag/runscreate/POST/threads/{thread_id}/runs/crons" target="_blank">API reference</a> for more details.

## Features

The LangGraph Cloud API offers several features to support complex agent architectures.

### Streaming

Streaming is critical for making LLM applications feel responsive to end users. When creating a streaming run, the streaming mode determines what data is streamed back to the API client. The LangGraph Cloud API supports five streaming modes.

- `values`: Stream the full state of the graph after each node is executed. See the [how-to guide](../how-tos/stream_values.md) for streaming values.
- `messages`: Stream complete messages (at the end of node execution) as well as tokens for any messages generated inside a node. This mode is primarily meant for powering chat applications. This is only an option if your graph contains a `messages` key. See the [how-to guide](../how-tos/stream_messages.md) for streaming messages.
- `updates`: Streams updates to the state of the graph after each node is executed. See the [how-to guide](../how-tos/stream_updates.md) for streaming updates.
- `events`: Stream all events (including the state of the graph) after each node is executed. See the [how-to guide](../how-tos/stream_events.md) for streaming events. This can be used to do token-by-token streaming for LLMs.
- `debug`: Stream debug events after each node is executed. See the [how-to guide](../how-tos/stream_debug.md) for streaming debug events.

You can also specify multiple streaming modes at the same time. See the [how-to guide](../how-tos/stream_multiple.md) for configuring multiple streaming modes at the same time.

See the <a href="../reference/api/api_ref.html#tag/runscreate/POST/threads/{thread_id}/runs/stream" target="_blank">API reference</a> for how to create streaming runs.

### Human-in-the-Loop

There are many occasions where the graph cannot run completely autonomously. For instance, the user might need to input some additional arguments to a function call, or select the next edge for the graph to continue on. In these instances, we need to insert some human in the loop interaction, which you can learn about in the [human in the loop how-tos](../how-tos/index.md#human-in-the-loop).

### Double Texting

Many times users might interact with your graph in unintended ways. For instance, a user may send one message and before the graph has finished running send a second message. To solve this issue of "double-texting" (i.e. prompting the graph a second time before the first run has finished), LangGraph has provided four different solutions, all of which are covered in the [Double Texting how-tos](../how-tos/index.md#double-texting). These options are:

- `reject`: This is the simplest option, this just rejects any follow up runs and does not allow double texting. See the [how-to guide](../how-tos/reject_concurrent.md) for configuring the reject double text option.
- `enqueue`: This is a relatively simple option which continues the first run until it completes the whole run, then sends the new input as a separate run. See the [how-to guide](../how-tos/enqueue_concurrent.md) for configuring the enqueue double text option.
- `interrupt`: This option interrupts the current execution but saves all the work done up until that point. It then inserts the user input and continues from there. If you enable this option, your graph should be able to handle weird edge cases that may arise. See the [how-to guide](../how-tos/interrupt_concurrent.md) for configuring the interrupt double text option.
- `rollback`: This option rolls back all work done up until that point. It then sends the user input in, basically as if it just followed the original run input. See the [how-to guide](../how-tos/rollback_concurrent.md) for configuring the rollback double text option.

### Stateless Runs

All runs use the built-in checkpointer to store checkpoints for runs. However, it can often be useful to just kick off a run without worrying about explicitly creating a thread and without wanting to keep those checkpointers around. Stateless runs allow you to do this by exposing an endpoint that:

- Takes in user input
- Under the hood, creates a thread
- Runs the agent but skips all checkpointing steps
- Cleans up the thread afterwards

Stateless runs are still retried as regular retries are per node, while everything still in memory, so doesn't use checkpoints.

The only difference is in stateless background runs, if the task worker dies halfway (not because the run itself failed, for some external reason) then the whole run will be retried like any background run, but

- whereas a stateful background run would retry from the last successful checkpoint
- a stateless background run would retry from the beginning

See the [how-to guide](../how-tos/cloud_examples/stateless_runs.ipynb) for creating stateless runs.

### Webhooks

For all types of runs, langgraph cloud supports completion webhooks. When you create the run you can pass a webhook URL to be called when the completes (successfully or not). This is especially useful for background runs and cron jobs, as the webhook can give you an indication the run has completed and you can perform further actions for your appilcation.

See this [how-to guide](../how-tos/cloud_examples/webhooks.ipynb) to learn about how to use webhooks with LangGraph Cloud.

## Deployment

The LangGraph Cloud offers several features to support secure and robost deployments.

### Authentication

LangGraph applications deployed to LangGraph Cloud are automatically configured with LangSmith authentication. In order to call the API, a valid <a href="https://docs.smith.langchain.com/how_to_guides/setup/create_account_api_key#api-keys" target="_blank">LangSmith API key</a> is required.

### Local Testing

Before deploying your app in production to LangGraph Cloud, you may wish to test out your graph locally in order to ensure that everything is running as expected. Luckily, LangGraph makes this easy for you through use of the LangGraph CLI. Read more in this [how-to guide](../deployment/test_locally.md) or look at the [CLI reference](../reference/cli.md) to learn more.
