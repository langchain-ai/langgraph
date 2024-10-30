# LangGraph Server

The LangGraph Platform API consists of a few core data models: [Graphs](#graphs), [Assistants](#assistants), [Threads](#threads), [Runs](#runs), and [Cron Jobs](#cron-jobs).

## Key Features

The LangGraph platform incorporates best practices for agent deployment, so you can focus on building your agent logic.

* **Streaming endpoints**: Endpoints that expose [multiple different streaming modes](streaming.md). We've made these work even for long-running agents that may go minutes between steam event.
* **Background runs**:  We've exposed endpoints for running your graph in the background. This includes endpoints for polling for the status of a run.
* **Support for long runs**: In addition to adding endpoints for background runs, we've also designed our endpoints and infrastructure to support long-running agents.
* **Task queue**: We've added a task queue to make sure we don't drop any requests if they arrive in a bursty nature.
* **Horizontally scalable infrastructure**: LangGraph Platform is designed to be horizontally scalable, allowing you to scale up and down your usage as needed.
* **Double texting support**: Many times users might interact with your graph in unintended ways. For instance, a user may send one message and before the graph has finished running send a second message. We call this ["double texting"](double_texting.md) and have added four different ways to handle this.
* **Optimized checkpointer**: LangGraph Platform comes with a built-in [checkpointer](persistence.md#checkpoints) optimized for LangGraph applications.
* **Human-in-the-loop endpoints**: We've exposed all endpoints needed to support [human-in-the-loop](human_in_the_loop.md) features.
* **Memory**: In addition to thread-level persistence (covered above by [checkpointers](persistence.md#checkpoints)), LangGraph Platform also comes with a built-in [memory store](persistence.md#memory-store).
* **Cron jobs**: Built-in support for scheduling tasks, enabling you to automate regular actions like data clean-up or batch processing within your applications.
* **Webhooks**: Allows your application to send real-time notifications and data updates to external systems, making it easy to integrate with third-party services and trigger actions based on specific events.
* **Monitoring**: LangGraph Server integrates seamlessly with the [LangSmith](https://docs.smith.langchain.com/) monitoring platform, providing real-time insights into your application's performance and health.

## Graphs

In order to use LangGraph Platform, you need to specify the graph(s) you want to deploy.
You can specify multiple graphs to deploy at the same time.
You do not need to specify [checkpointers](persistence.md#checkpoints) or [memory stores](persistence.md#memory-store) when compiling your graphs - LangGraph platform will add those in automatically.

Please see [application structure](./application_structure.md) for more information on how to structure your LangGraph application for deployment.

## Assistants

An [Assistant](assistants.md) refers to a [graph](#graphs) plus specific [configuration](low_level.md#configuration) settings for that graph.

You can think of an assistant as a saved configuration of an [agent](agentic_concepts.md).

When building agents, it is fairly common to make rapid changes that *do not* alter the graph logic. For example, simply changing prompts or the LLM selection can have significant impacts on the behavior of the agents. Assistants offer an easy way to make and save these types of changes to agent configuration.

## Threads

A thread contains the accumulated state of a sequence of runs. If a run is executed on a thread, then the [state](low_level.md#state) of the underlying graph of the assistant will be persisted to the thread. 

A thread's current and historical state can be retrieved. To persist state, a thread must be created prior to executing a run.

The state of a thread at a particular point in time is called a `checkpoint`.

For more on threads and checkpoints, see this section of the [LangGraph conceptual guide](low_level.md#persistence).

The LangGraph Cloud API provides several endpoints for creating and managing threads and thread state. See the [API reference](../reference/api/api_ref.html#tag/threadscreate) for more details.

## Runs

A run is an invocation of an assistant. Each run may have its own input, configuration, and metadata, which may affect execution and output of the underlying graph. A run can optionally be executed on a thread.

The LangGraph Cloud API provides several endpoints for creating and managing runs. See the [API reference](../reference/api/api_ref.html#tag/runscreate) for more details.

## Cron Jobs

There are many situations in which it is useful to run an assistant on a schedule. 

For example, say that you're building an assistant that runs daily and sends an email summary
of the day's news. You could use a cron job to run the assistant every day at 8:00 PM.

LangGraph Cloud supports cron jobs, which run on a user-defined schedule. The user specifies a schedule, an assistant, and some input. After that, on the specified schedule, LangGraph Cloud will:

- Create a new thread with the specified assistant
- Send the specified input to that thread

Note that this sends the same input to the thread every time. See the [how-to guide](../cloud/how-tos/cron_jobs.md) for creating cron jobs.

The LangGraph Cloud API provides several endpoints for creating and managing cron jobs. See the [API reference](../reference/api/api_ref.html#tag/runscreate/POST/threads/{thread_id}/runs/crons) for more details.


## Related

* Read the conceptual guide about [application structure](./application_structure.md) to learn how to structure your LangGraph application for deployment.
* [How-to guides for the LangGraph Platform](../how-tos/index.md) 
