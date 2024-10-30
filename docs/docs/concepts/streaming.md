# Streaming

LangGraph is built with first class support for streaming. There are several different ways to stream back outputs from a graph run

## Streaming graph outputs (`.stream` and `.astream`)

`.stream` and `.astream` are sync and async methods for streaming back outputs from a graph run.
There are several different modes you can specify when calling these methods (e.g. `graph.stream(..., mode="...")):

- [`"values"`](../how-tos/stream-values.ipynb): This streams the full value of the state after each step of the graph.
- [`"updates"`](../how-tos/stream-updates.ipynb): This streams the updates to the state after each step of the graph. If multiple updates are made in the same step (e.g. multiple nodes are run) then those updates are streamed separately.
- [`"custom"`](../how-tos/streaming-content.ipynb): This streams custom data from inside your graph nodes.
- [`"messages"`](../how-tos/streaming-tokens.ipynb): This streams LLM tokens and metadata for the graph node where LLM is invoked.
- `"debug"`: This streams as much information as possible throughout the execution of the graph.

You can also specify multiple streaming modes at the same time by passing them as a list. When you do this, the streamed outputs will be tuples `(stream_mode, data)`. For example:

```python
graph.stream(..., stream_mode=["updates", "messages"])
```

```
...
('messages', (AIMessageChunk(content='Hi'), {'langgraph_step': 3, 'langgraph_node': 'agent', ...}))
...
('updates', {'agent': {'messages': [AIMessage(content="Hi, how can I help you?")]}})
```

The below visualization shows the difference between the `values` and `updates` modes:

![values vs updates](../static/values_vs_updates.png)


## Streaming LLM tokens and events (`.astream_events`)

In addition, you can use the [`astream_events`](../how-tos/streaming-events-from-within-tools.ipynb) method to stream back events that happen _inside_ nodes. This is useful for [streaming tokens of LLM calls](../how-tos/streaming-tokens.ipynb).

This is a standard method on all [LangChain objects](https://python.langchain.com/docs/concepts/#runnable-interface). This means that as the graph is executed, certain events are emitted along the way and can be seen if you run the graph using `.astream_events`. 

All events have (among other things) `event`, `name`, and `data` fields. What do these mean?

- `event`: This is the type of event that is being emitted. You can find a detailed table of all callback events and triggers [here](https://python.langchain.com/docs/concepts/#callback-events).
- `name`: This is the name of event.
- `data`: This is the data associated with the event.

What types of things cause events to be emitted?

* each node (runnable) emits `on_chain_start` when it starts execution, `on_chain_stream` during the node execution and `on_chain_end` when the node finishes. Node events will have the node name in the event's `name` field
* the graph will emit `on_chain_start` in the beginning of the graph execution, `on_chain_stream` after each node execution and `on_chain_end` when the graph finishes. Graph events will have the `LangGraph` in the event's `name` field
* Any writes to state channels (i.e. anytime you update the value of one of your state keys) will emit `on_chain_start` and `on_chain_end` events

Additionally, any events that are created inside your nodes (LLM events, tool events, manually emitted events, etc.) will also be visible in the output of `.astream_events`.

To make this more concrete and to see what this looks like, let's see what events are returned when we run a simple graph:

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

model = ChatOpenAI(model="gpt-4o-mini")


def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages": response}

workflow = StateGraph(MessagesState)
workflow.add_node(call_model)
workflow.add_edge(START, "call_model")
workflow.add_edge("call_model", END)
app = workflow.compile()

inputs = [{"role": "user", "content": "hi!"}]
async for event in app.astream_events({"messages": inputs}, version="v1"):
    kind = event["event"]
    print(f"{kind}: {event['name']}")
```
```shell
on_chain_start: LangGraph
on_chain_start: __start__
on_chain_end: __start__
on_chain_start: call_model
on_chat_model_start: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_end: ChatOpenAI
on_chain_start: ChannelWrite<call_model,messages>
on_chain_end: ChannelWrite<call_model,messages>
on_chain_stream: call_model
on_chain_end: call_model
on_chain_stream: LangGraph
on_chain_end: LangGraph
```

We start with the overall graph start (`on_chain_start: LangGraph`). We then write to the `__start__` node (this is special node to handle input).
We then start the `call_model` node (`on_chain_start: call_model`). We then start the chat model invocation (`on_chat_model_start: ChatOpenAI`),
stream back token by token (`on_chat_model_stream: ChatOpenAI`) and then finish the chat model (`on_chat_model_end: ChatOpenAI`). From there, 
we write the results back to the channel (`ChannelWrite<call_model,messages>`) and then finish the `call_model` node and then the graph as a whole.

This should hopefully give you a good sense of what events are emitted in a simple graph. But what data do these events contain?
Each type of event contains data in a different format. Let's look at what `on_chat_model_stream` events look like. This is an important type of event
since it is needed for streaming tokens from an LLM response.

These events look like:

```shell
{'event': 'on_chat_model_stream',
 'name': 'ChatOpenAI',
 'run_id': '3fdbf494-acce-402e-9b50-4eab46403859',
 'tags': ['seq:step:1'],
 'metadata': {'langgraph_step': 1,
  'langgraph_node': 'call_model',
  'langgraph_triggers': ['start:call_model'],
  'langgraph_task_idx': 0,
  'checkpoint_id': '1ef657a0-0f9d-61b8-bffe-0c39e4f9ad6c',
  'checkpoint_ns': 'call_model',
  'ls_provider': 'openai',
  'ls_model_name': 'gpt-4o-mini',
  'ls_model_type': 'chat',
  'ls_temperature': 0.7},
 'data': {'chunk': AIMessageChunk(content='Hello', id='run-3fdbf494-acce-402e-9b50-4eab46403859')},
 'parent_ids': []}
```
We can see that we have the event type and name (which we knew from before).

We also have a bunch of stuff in metadata. Noticeably, `'langgraph_node': 'call_model',` is some really helpful information
which tells us which node this model was invoked inside of.

Finally, `data` is a really important field. This contains the actual data for this event! Which in this case
is an AIMessageChunk. This contains the `content` for the message, as well as an `id`.
This is the ID of the overall AIMessage (not just this chunk) and is super helpful - it helps
us track which chunks are part of the same message (so we can show them together in the UI).

This information contains all that is needed for creating a UI for streaming LLM tokens. You can see a 
guide for that [here](../how-tos/streaming-tokens.ipynb).


!!! warning "ASYNC IN PYTHON<=3.10"
    You may fail to see events being emitted from inside a node when using `.astream_events` in Python <= 3.10. If you're using a Langchain RunnableLambda, a RunnableGenerator, or Tool asynchronously inside your node, you will have to propagate callbacks to these objects manually. This is because LangChain cannot automatically propagate callbacks to child objects in this case. Please see examples [here](../how-tos/streaming-content.ipynb) and [here](../how-tos/streaming-events-from-within-tools.ipynb).


## LangGraph Platform

Streaming is critical for making LLM applications feel responsive to end users. When creating a streaming run, the streaming mode determines what data is streamed back to the API client. LangGraph Platform supports five streaming modes:

- `values`: Stream the full state of the graph after each [super-step](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs) is executed. See the [how-to guide](../cloud/how-tos/stream_values.md) for streaming values.
- `messages`: Stream complete messages (at the end of node execution) as well as tokens for any messages generated inside a node. This mode is primarily meant for powering chat applications. This is only an option if your graph contains a `messages` key. See the [how-to guide](../cloud/how-tos/stream_messages.md) for streaming messages.
- `updates`: Streams updates to the state of the graph after each node is executed. See the [how-to guide](../cloud/how-tos/stream_updates.md) for streaming updates.
- `events`: Stream all events (including the state of the graph) that occur during graph execution. See the [how-to guide](../cloud/how-tos/stream_events.md) for streaming events. This can be used to do token-by-token streaming for LLMs.
- `debug`: Stream debug events throughout graph execution. See the [how-to guide](../cloud/how-tos/stream_debug.md) for streaming debug events.

You can also specify multiple streaming modes at the same time. See the [how-to guide](../cloud/how-tos/stream_multiple.md) for configuring multiple streaming modes at the same time.

See the [API reference](../reference/api/api_ref.html#tag/runscreate/POST/threads/{thread_id}/runs/stream) for how to create streaming runs.

Streaming modes `values`, `updates`, and `debug` are very similar to modes available in the LangGraph library - for a deeper conceptual explanation of those, you can see the [previous section](#streaming-graph-outputs-stream-and-astream).

Streaming mode `events` is the same as using `.astream_events` in the LangGraph library - for a deeper conceptual explanation of this, you can see the LangGraph library documentation [here](low_level.md#streaming).

### `stream_mode="messages"`

Streaming mode `messages` is for streaming back messages from the LLM. Assuming we have a simple [ReAct](./agentic_concepts.md#react-implementation)-style agent deployed, what does this stream look like?

All events emitted have two attributes:

- `event`: This is the name of the event
- `data`: This is data associated with the event

!!! note
    Streaming mode `messages` is different from the one in the LangGraph library:
    - LangGraph Server streams event objects with messages in the `data` field, while LangGraph library streams tuples (`AIMessageChunk`, metadata).
    - In LangGraph Server, metadata is streamed only once per message (`messages/metadata`), before the individual tokens are streamed (`messages/partial`), while in LangGraph library it's streamed with every `AIMessageChunk` (for each LLM token).
    - LangGraph Server also streams additional events (`metadata`, `messages/complete`, see below for more details).

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

We then get a `messages/metadata` - this is letting us know that a new message is starting and provides additional information about the LLM as well as the node where the LLM is invoked.

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

And that's it! This is more focused streaming mode specifically focused on streaming back messages. See this [how-to guide](../cloud/how-tos/stream_messages.md) for more information.