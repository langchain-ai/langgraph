# API Concepts

This page describes the high-level concepts of the LangGraph Cloud API. The conceptual guide of LangGraph (Python library) is [here](../../concepts/index.md).

## Data Models

The LangGraph Cloud API consists of a few core data models: [Assistants](#assistants), [Threads](#threads), [Runs](#runs), and [Cron Jobs](#cron-jobs).

### Assistants

An assistant is a configured instance of a [`CompiledGraph`][compiledgraph]. It abstracts the cognitive architecture of the graph and contains instance specific configuration and metadata. Multiple assistants can reference the same graph but can contain different configuration and metadata, which may differentiate the behavior of the assistants. An assistant (i.e. the graph) is invoked as part of a run.

The LangGraph Cloud API provides several endpoints for creating and managing assistants. See the [API reference](../reference/api/api_ref.html#tag/assistantscreate) for more details.

#### Configuring Assistants

You can save custom assistants from the same graph to set different default prompts, models, and other configurations without changing a line of code in your graph. This allows you the ability to quickly test out different configurations without having to rewrite your graph every time, and also give users the flexibility to select different configurations when using your LangGraph application. See <a href="https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/configuration_cloud/">this</a> how-to for information on how to configure a deployed graph. 

### Threads

A thread contains the accumulated state of a group of runs. If a run is executed on a thread, then the [state][state] of the underlying graph of the assistant will be persisted to the thread. A thread's current and historical state can be retrieved. To persist state, a thread must be created prior to executing a run.

The state of a thread at a particular point in time is called a checkpoint.

For more on threads and checkpoints, see this section of the [LangGraph conceptual guide](../../concepts/persistence.md).

The LangGraph Cloud API provides several endpoints for creating and managing threads and thread state. See the [API reference](../reference/api/api_ref.html#tag/threadscreate) for more details.

### Runs

A run is an invocation of an assistant. Each run may have its own input, configuration, and metadata, which may affect execution and output of the underlying graph. A run can optionally be executed on a thread.

The LangGraph Cloud API provides several endpoints for creating and managing runs. See the [API reference](../reference/api/api_ref.html#tag/runscreate) for more details.

### Cron Jobs

It's often useful to run graphs on some schedule. LangGraph Cloud supports cron jobs, which run on a user defined schedule. The user specifies a schedule, an assistant, and some input. After than, on the specified schedule LangGraph cloud will:

- Create a new thread with the specified assistant
- Send the specified input to that thread

Note that this sends the same input to the thread every time. See the [how-to guide](../how-tos/cloud_examples/cron_jobs.ipynb) for creating cron jobs.

The LangGraph Cloud API provides several endpoints for creating and managing cron jobs. See the [API reference](../reference/api/api_ref.html#tag/runscreate/POST/threads/{thread_id}/runs/crons) for more details.

## Features

The LangGraph Cloud API offers several features to support complex agent architectures.

### Streaming

Streaming is critical for making LLM applications feel responsive to end users. When creating a streaming run, the streaming mode determines what data is streamed back to the API client. The LangGraph Cloud API supports five streaming modes.

- `values`: Stream the full state of the graph after each [super-step](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs) is executed. See the [how-to guide](../how-tos/stream_values.md) for streaming values.
- `messages`: Stream complete messages (at the end of node execution) as well as tokens for any messages generated inside a node. This mode is primarily meant for powering chat applications. This is only an option if your graph contains a `messages` key. See the [how-to guide](../how-tos/stream_messages.md) for streaming messages.
- `updates`: Streams updates to the state of the graph after each node is executed. See the [how-to guide](../how-tos/stream_updates.md) for streaming updates.
- `events`: Stream all events (including the state of the graph) that occur during graph execution. See the [how-to guide](../how-tos/stream_events.md) for streaming events. This can be used to do token-by-token streaming for LLMs.
- `debug`: Stream debug events throughout graph execution. See the [how-to guide](../how-tos/stream_debug.md) for streaming debug events.

You can also specify multiple streaming modes at the same time. See the [how-to guide](../how-tos/stream_multiple.md) for configuring multiple streaming modes at the same time.

See the [API reference](../reference/api/api_ref.html#tag/runscreate/POST/threads/{thread_id}/runs/stream) for how to create streaming runs.

Streaming modes `values`, `updates`, and `debug` are very similar to modes available in the LangGraph library - for a deeper conceptual explanation of those, you can see the LangGraph library documentation [here](../../concepts/low_level.md#streaming).

Streaming mode `events` is the same as using `.astream_events` in the LangGraph library - for a deeper conceptual explanation of this, you can see the LangGraph library documentation [here](../../concepts/low_level.md#streaming).

#### `mode="messages"`
Streaming mode `messages` is a new streaming mode, currently only available in the API. What does this mode enable?

This mode is focused on streaming back messages. It currently assumes that you have a `messages` key in your graph that is a list of messages. Assuming we have a simple react agent deployed, what does this stream look like?

All events emitted have two attributes:

- `event`: This is the name of the event
- `data`: This is data associated with the event

Let's run it on a question that should trigger a tool call:

```python
thread = await client.threads.create()
input = {"messages": [{"role": "user", "content": "what's the weather in sf?"}]}

events = []
async for event in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",  # This may need to change depending on the graph you deployed
    input=input,
    stream_mode="messages",
):
    print(event.event)
```
```shell
metadata
messages/complete
messages/metadata
messages/partial
...
messages/partial
messages/complete
messages/complete
messages/metadata
messages/partial
...
messages/partial
messages/complete
end
```

We first get some `metadata` - this is metadata about the run. 

```python
StreamPart(event='metadata', data={'run_id': '1ef657cf-ae55-6f65-97d4-f4ed1dbdabc6'})
```

We then get a `messages/complete` event - this a fully formed message getting emitted. In this case,
this was the just the input message we sent in. 

```python
StreamPart(event='messages/complete', data=[{'content': 'hi!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '833c09a3-bb19-46c9-81d9-1e5954ec5f92', 'example': False}])
```

We then get a `messages/metadata` - this is just letting us know that a new message is starting.

```python
StreamPart(event='messages/metadata', data={'run-985c0f14-9f43-40d4-a505-4637fc58e333': {'metadata': {'created_by': 'system', 'run_id': '1ef657de-7594-66df-8eb2-31518e4a1ee2', 'graph_id': 'agent', 'thread_id': 'c178eab5-e293-423c-8e7d-1d113ffe7cd9', 'model_name': 'openai', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o', 'ls_model_type': 'chat', 'ls_temperature': 0.0}}})
```

We then get a BUNCH of `messages/partial` events - these are the individual tokens from the LLM! In the case below, we can see the START of a tool call.

```python
StreamPart(event='messages/partial', data=[{'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_w8Hr8dHGuZCPgRfd5FqRBArs', 'function': {'arguments': '', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-985c0f14-9f43-40d4-a505-4637fc58e333', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [{'name': 'tavily_search_results_json', 'args': '', 'id': 'call_w8Hr8dHGuZCPgRfd5FqRBArs', 'error': None}], 'usage_metadata': None}])
```

After that, we get a `messages/complete` event - this is the AIMessage finishing. It's now a complete tool call:

```python
StreamPart(event='messages/complete', data=[{'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_w8Hr8dHGuZCPgRfd5FqRBArs', 'function': {'arguments': '{"query":"current weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-985c0f14-9f43-40d4-a505-4637fc58e333', 'example': False, 'tool_calls': [{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_w8Hr8dHGuZCPgRfd5FqRBArs'}], 'invalid_tool_calls': [], 'usage_metadata': None}])
```

After that, we get ANOTHER `messages/complete` event. This is a tool message - our agent has called a tool, gotten a response, and now inserting it into the state in the form of a tool message.

```python
StreamPart(event='messages/complete', data=[{'content': '[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'San Francisco\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 37.78, \'lon\': -122.42, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1724877689, \'localtime\': \'2024-08-28 13:41\'}, \'current\': {\'last_updated_epoch\': 1724877000, \'last_updated\': \'2024-08-28 13:30\', \'temp_c\': 23.3, \'temp_f\': 73.9, \'is_day\': 1, \'condition\': {\'text\': \'Partly cloudy\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/day/116.png\', \'code\': 1003}, \'wind_mph\': 15.0, \'wind_kph\': 24.1, \'wind_degree\': 310, \'wind_dir\': \'NW\', \'pressure_mb\': 1014.0, \'pressure_in\': 29.93, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 57, \'cloud\': 25, \'feelslike_c\': 25.0, \'feelslike_f\': 77.1, \'windchill_c\': 20.9, \'windchill_f\': 69.6, \'heatindex_c\': 23.3, \'heatindex_f\': 74.0, \'dewpoint_c\': 12.9, \'dewpoint_f\': 55.2, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 6.0, \'gust_mph\': 19.5, \'gust_kph\': 31.3}}"}]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'tavily_search_results_json', 'id': '0112eba5-7660-4375-9f24-c7a1d6777b97', 'tool_call_id': 'call_w8Hr8dHGuZCPgRfd5FqRBArs'}])
```

After that, we see the agent doing another LLM call and streaming back a response. We then get an `end` event:

```python
StreamPart(event='end', data=None)
```

And that's it! This is more focused streaming mode specifically focused on streaming back messages. See this [how-to guide](../how-tos/stream_messages.md) for more information.


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
