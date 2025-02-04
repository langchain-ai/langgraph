# How to use Postgres checkpointer for persistence

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
                <a href="https://www.postgresql.org/about/">
                    Postgresql
                </a>
            </li>        
        </ul>
    </p>
</div> 

When creating LangGraph agents, you can also set them up so that they persist their state. This allows you to do things like interact with an agent multiple times and have it remember previous interactions.

This how-to guide shows how to use `Postgres` as the backend for persisting checkpoint state using the [`langgraph-checkpoint-postgres`](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres) library.

For demonstration purposes we add persistence to the [pre-built create react agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent). 

In general, you can add a checkpointer to any custom graph that you build like this:

```python
from langgraph.graph import StateGraph

builder = StateGraph(....)
# ... define the graph
checkpointer = # postgres checkpointer (see examples below)
graph = builder.compile(checkpointer=checkpointer)
...
```

!!! info "Setup"
    You need to run `.setup()` once on your checkpointer to initialize the database before you can use it.

## Setup

You will need access to a postgres instance. There are many resources online that can help
you set up a postgres instance.

Next, let's install the required packages and set our API keys


```
%%capture --no-stderr
%pip install -U psycopg psycopg-pool langgraph langgraph-checkpoint-postgres
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
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


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

## Use sync connection

This sets up a synchronous connection to the database. 

Synchronous connections execute operations in a blocking manner, meaning each operation waits for completion before moving to the next one. The `DB_URI` is the database connection URI, with the protocol used for connecting to a PostgreSQL database, authentication, and host where database is running. The connection_kwargs dictionary defines additional parameters for the database connection.


```python
DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
```


```python
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
```

### With a connection pool

This manages a pool of reusable database connections: 
- Advantages: Efficient resource utilization, improved performance for frequent connections
- Best for: Applications with many short-lived database operations



```python
from psycopg_pool import ConnectionPool

with ConnectionPool(
    # Example configuration
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
) as pool:
    checkpointer = PostgresSaver(pool)

    # NOTE: you need to call .setup() the first time you're using your checkpointer
    checkpointer.setup()

    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)
    checkpoint = checkpointer.get(config)
```


```python
res
```







```python
checkpoint
```






### With a connection

This creates a single, dedicated connection to the database:
- Advantages: Simple to use, suitable for longer transactions
- Best for: Applications with fewer, longer-lived database operations


```python
from psycopg import Connection


with Connection.connect(DB_URI, **connection_kwargs) as conn:
    checkpointer = PostgresSaver(conn)
    # NOTE: you need to call .setup() the first time you're using your checkpointer
    # checkpointer.setup()
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "2"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    checkpoint_tuple = checkpointer.get_tuple(config)
```


```python
checkpoint_tuple
```






### With a connection string

This creates a connection based on a connection string:
- Advantages: Simplicity, encapsulates connection details
- Best for: Quick setup or when connection details are provided as a string


```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "3"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    checkpoint_tuples = list(checkpointer.list(config))
```


```python
checkpoint_tuples
```






## Use async connection

This sets up an asynchronous connection to the database. 

Async connections allow non-blocking database operations. This means other parts of your application can continue running while waiting for database operations to complete. It's particularly useful in high-concurrency scenarios or when dealing with I/O-bound operations.

### With a connection pool


```python
from psycopg_pool import AsyncConnectionPool

async with AsyncConnectionPool(
    # Example configuration
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
) as pool:
    checkpointer = AsyncPostgresSaver(pool)

    # NOTE: you need to call .setup() the first time you're using your checkpointer
    await checkpointer.setup()

    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "4"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )

    checkpoint = await checkpointer.aget(config)
```


```python
checkpoint
```






### With a connection


```python
from psycopg import AsyncConnection

async with await AsyncConnection.connect(DB_URI, **connection_kwargs) as conn:
    checkpointer = AsyncPostgresSaver(conn)
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "5"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )
    checkpoint_tuple = await checkpointer.aget_tuple(config)
```


```python
checkpoint_tuple
```






### With a connection string


```python
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "6"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )
    checkpoint_tuples = [c async for c in checkpointer.alist(config)]
```


```python
checkpoint_tuples
```





