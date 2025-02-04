# How to use MongoDB checkpointer for persistence

<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide assumes familiarity with the following:
        <ul>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/persistence/">
                    Persistence
                </a>
            </li>       
            <li>
                <a href="https://www.mongodb.com/">
                    MongoDB
                </a>
            </li>        
        </ul>
    </p>
</div> 

When creating LangGraph agents, you can also set them up so that they persist their state. This allows you to do things like interact with an agent multiple times and have it remember previous interactions. 

This reference implementation shows how to use MongoDB as the backend for persisting checkpoint state using the `langgraph-checkpoint-mongodb` library.

For demonstration purposes we add persistence to a [prebuilt ReAct agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/).

In general, you can add a checkpointer to any custom graph that you build like this:

```python
from langgraph.graph import StateGraph

builder = StateGraph(...)
# ... define the graph
checkpointer = # mongodb checkpointer (see examples below)
graph = builder.compile(checkpointer=checkpointer)
...
```

## Setup

To use the MongoDB checkpointer, you will need a MongoDB cluster. Follow [this guide](https://www.mongodb.com/docs/guides/atlas/cluster/) to create a cluster if you don't already have one.

Next, let's install the required packages and set our API keys


```
%%capture --no-stderr
%pip install -U pymongo langgraph langgraph-checkpoint-mongodb
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Define model and tools for the graph


```python
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
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```

## MongoDB checkpointer usage

### With a connection string

This creates a connection to MongoDB directly using the connection string of your cluster. This is ideal for use in scripts, one-off operations and short-lived applications.


```python
from langgraph.checkpoint.mongodb import MongoDBSaver

MONGODB_URI = "localhost:27017"  # replace this with your connection string

with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    response = graph.invoke(
        {"messages": [("human", "what's the weather in sf")]}, config
    )
```


```python
response
```






### Using the MongoDB client

This creates a connection to MongoDB using the MongoDB client. This is ideal for long-running applications since it allows you to reuse the client instance for multiple database operations without needing to reinitialize the connection each time.


```python
from pymongo import MongoClient

mongodb_client = MongoClient(MONGODB_URI)

checkpointer = MongoDBSaver(mongodb_client)
graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "2"}}
response = graph.invoke({"messages": [("user", "What's the weather in sf?")]}, config)
```


```python
response
```







```python
# Retrieve the latest checkpoint for the given thread ID
# To retrieve a specific checkpoint, pass the checkpoint_id in the config
checkpointer.get_tuple(config)
```







```python
# Remember to close the connection after you're done
mongodb_client.close()
```

### Using an async connection

This creates a short-lived asynchronous connection to MongoDB. 

Async connections allow non-blocking database operations. This means other parts of your application can continue running while waiting for database operations to complete. It's particularly useful in high-concurrency scenarios or when dealing with I/O-bound operations.


```python
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

async with AsyncMongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "3"}}
    response = await graph.ainvoke(
        {"messages": [("user", "What's the weather in sf?")]}, config
    )
```


```python
response
```






### Using the async MongoDB client

This routes connections to MongoDB through an asynchronous MongoDB client.


```python
from pymongo import AsyncMongoClient

async_mongodb_client = AsyncMongoClient(MONGODB_URI)

checkpointer = AsyncMongoDBSaver(async_mongodb_client)
graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "4"}}
response = await graph.ainvoke(
    {"messages": [("user", "What's the weather in sf?")]}, config
)
```


```python
response
```







```python
# Retrieve the latest checkpoint for the given thread ID
# To retrieve a specific checkpoint, pass the checkpoint_id in the config
latest_checkpoint = await checkpointer.aget_tuple(config)
print(latest_checkpoint)
```


```python
# Remember to close the connection after you're done
await async_mongodb_client.close()
```
