# How to convert LangGraph calls to LangGraph Cloud calls

So you're used to interacting with your graph locally, but now you've deployed it with LangGraph cloud. How do you change all the places in your codebase where you call LangGraph directly to call LangGraph Cloud? This notebook contains side-by-side comparisons so you can easily transition from calling LangGraph to calling LangGraph Cloud.

## Setup

We'll be using a simple ReAct agent for this how-to guide. You will also need to set up a project with `agent.py` and `langgraph.json` files. See [quick start](https://langchain-ai.github.io/langgraph/cloud/quick_start/#develop) for setting this up.


```
%%capture --no-stderr
%pip install -U langgraph langchain-openai
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```


```python
# this is all that's needed for the agent.py
from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

model = ChatOpenAI(model_name="gpt-4o", temperature=0)
graph = create_react_agent(model, tools)
```

Now we'll set up the langgraph client. The client assumes the LangGraph Cloud server is running on `localhost:8123`


```python
from langgraph_sdk import get_client

client = get_client()
```

## Invoking the graph

Below examples show how to mirror `.invoke() / .ainvoke()` methods of LangGraph's `CompiledGraph` runnable, i.e. create a blocking graph execution

### With LangGraph


```python
inputs = {"messages": [("human", "what's the weather in sf")]}
invoke_output = await graph.ainvoke(inputs)
```


```python
for m in invoke_output["messages"]:
    m.pretty_print()
```

### With LangGraph Cloud


```python
# NOTE: We're not specifying the thread here -- this allows us to create a thread just for this run
wait_output = await client.runs.wait(None, "agent", input=inputs)
```


```python
# we'll use this for pretty message formatting
from langchain_core.messages import convert_to_messages
```


```python
for m in convert_to_messages(wait_output["messages"]):
    m.pretty_print()
```

## Streaming

Below examples show how to mirror `.stream() / .astream()` methods for streaming partial graph execution results.  
Note: LangGraph's `stream_mode=values/updates/debug` behave nearly identically in LangGraph Cloud (with the exception of additional streamed chunks with `metadata` / `end` events types)

### With LangGraph


```python
inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in graph.astream(inputs, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

### With LangGraph Cloud


```python
inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in client.runs.stream(
    None, "agent", input=inputs, stream_mode="values"
):
    if chunk.event == "values":
        messages = convert_to_messages(chunk.data["messages"])
        messages[-1].pretty_print()
```

## Persistence

In LangGraph, you need to provide a `checkpointer` object when compiling your graph to persist state across interactions with your graph (i.e. threads). In LangGraph Cloud, you don't need to create a checkpointer -- the server already implements one for you. You can also directly manage the threads from a client.

### With LangGraph


```python
from langgraph.checkpoint.memory import MemorySaver
```


```python
checkpointer = MemorySaver()
graph_with_memory = create_react_agent(model, tools, checkpointer=checkpointer)
```


```python
inputs = {"messages": [("human", "what's the weather in nyc")]}
invoke_output = await graph_with_memory.ainvoke(
    inputs, config={"configurable": {"thread_id": "1"}}
)
invoke_output["messages"][-1].pretty_print()
```


```python
inputs = {"messages": [("human", "what's it known for?")]}
invoke_output = await graph_with_memory.ainvoke(
    inputs, config={"configurable": {"thread_id": "1"}}
)
invoke_output["messages"][-1].pretty_print()
```


```python
inputs = {"messages": [("human", "what's it known for?")]}
invoke_output = await graph_with_memory.ainvoke(
    inputs, config={"configurable": {"thread_id": "2"}}
)
invoke_output["messages"][-1].pretty_print()
```


```python
# get the state of the thread
checkpointer.get({"configurable": {"thread_id": "2"}})
```






### With LangGraph Cloud

Let's now reproduce the same using LangGraph Cloud. Note that instead of using a checkpointer we just create a new thread on the backend and pass the ID to the API


```python
thread = await client.threads.create()
```


```python
inputs = {"messages": [("human", "what's the weather in nyc")]}
wait_output = await client.runs.wait(thread["thread_id"], "agent", input=inputs)
convert_to_messages(wait_output["messages"])[-1].pretty_print()
```


```python
inputs = {"messages": [("human", "what's it known for?")]}
wait_output = await client.runs.wait(thread["thread_id"], "agent", input=inputs)
convert_to_messages(wait_output["messages"])[-1].pretty_print()
```


```python
thread = await client.threads.create()
```


```python
inputs = {"messages": [("human", "what's it known for?")]}
wait_output = await client.runs.wait(thread["thread_id"], "agent", input=inputs)
convert_to_messages(wait_output["messages"])[-1].pretty_print()
```


```python
# get the state of the thread
await client.threads.get_state(thread["thread_id"])
```






## Breakpoints

### With LangGraph


```python
inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in graph_with_memory.astream(
    inputs,
    stream_mode="values",
    interrupt_before=["tools"],
    config={"configurable": {"thread_id": "3"}},
):
    chunk["messages"][-1].pretty_print()
```


```python
async for chunk in graph_with_memory.astream(
    None,
    stream_mode="values",
    interrupt_before=["tools"],
    config={"configurable": {"thread_id": "3"}},
):
    chunk["messages"][-1].pretty_print()
```

### With LangGraph Cloud

Similar to the persistence example, we need to create a thread so we can persist state and continue from the breakpoint.


```python
thread = await client.threads.create()

async for chunk in client.runs.stream(
    thread["thread_id"],
    "agent",
    input=inputs,
    stream_mode="values",
    interrupt_before=["tools"],
):
    if chunk.event == "values":
        messages = convert_to_messages(chunk.data["messages"])
        messages[-1].pretty_print()
```


```python
async for chunk in client.runs.stream(
    thread["thread_id"],
    "agent",
    input=None,
    stream_mode="values",
    interrupt_before=["tools"],
):
    if chunk.event == "values":
        messages = convert_to_messages(chunk.data["messages"])
        messages[-1].pretty_print()
```

## Steaming events

For streaming events, in LangGraph you need to use `.astream` method on the `CompiledGraph`. In LangGraph Cloud this is done via passing `stream_mode="events"`

### With LangGraph


```python
from langchain_core.messages import AIMessageChunk

inputs = {"messages": [("human", "what's the weather in sf")]}
first = True
async for msg, metadata in graph.astream(inputs, stream_mode="messages"):
    if msg.content:
        print(msg.content, end="|", flush=True)

    if isinstance(msg, AIMessageChunk):
        if first:
            gathered = msg
            first = False
        else:
            gathered = gathered + msg

        if msg.tool_call_chunks:
            print(gathered.tool_calls)
```

### With LangGraph Cloud


```python
inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in client.runs.stream(
    None, "agent", input=inputs, stream_mode="events"
):
    if chunk.event == "events" and chunk.data["event"] == "on_chat_model_stream":
        print(chunk.data["data"]["chunk"])
```
