# How to add cross-thread persistence (functional API)

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Functional API](../../concepts/functional_api/.md)
    - [Persistence](../../concepts/persistence/.md)
    - [Memory](../../concepts/memory/.md)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

LangGraph allows you to persist data across **different [threads](../../concepts/persistence/#threads.md)**. For instance, you can store information about users (their names or preferences) in a shared (cross-thread) memory and reuse them in the new threads (e.g., new conversations).

When using the [functional API](../../concepts/functional_api/.md), you can set it up to store and retrieve memories by using the [Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) interface:

1. Create an instance of a `Store`

    ```python
    from langgraph.store.memory import InMemoryStore, BaseStore
    
    store = InMemoryStore()
    ```

2. Pass the `store` instance to the `entrypoint()` decorator and expose `store` parameter in the function signature:

    ```python
    from langgraph.func import entrypoint

    @entrypoint(store=store)
    def workflow(inputs: dict, store: BaseStore):
        my_task(inputs).result()
        ...
    ```
    
In this guide, we will show how to construct and use a workflow that has a shared memory implemented using the [Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) interface.

!!! note Note

    Support for the [`Store`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) API that is used in this guide was added in LangGraph `v0.2.32`.

    Support for __index__ and __query__ arguments of the [`Store`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) API that is used in this guide was added in LangGraph `v0.2.54`.

!!! tip "Note"

    If you need to add cross-thread persistence to a `StateGraph`, check out this [how-to guide](../cross-thread-persistence.md).

## Setup

First, let's install the required packages and set our API keys


```
%%capture --no-stderr
%pip install -U langchain_anthropic langchain_openai langgraph
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
_set_env("OPENAI_API_KEY")
```

!!! tip "Set up [LangSmith](https://smith.langchain.com) for LangGraph development"

    Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started [here](https://docs.smith.langchain.com)

## Example: simple chatbot with long-term memory

### Define store

In this example we will create a workflow that will be able to retrieve information about a user's preferences. We will do so by defining an `InMemoryStore` - an object that can store data in memory and query that data.

When storing objects using the `Store` interface you define two things:

* the namespace for the object, a tuple (similar to directories)
* the object key (similar to filenames)

In our example, we'll be using `("memories", <user_id>)` as namespace and random UUID as key for each new memory.

Importantly, to determine the user, we will be passing `user_id` via the config keyword argument of the node function.

Let's first define our store!


```python
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

in_memory_store = InMemoryStore(
    index={
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "dims": 1536,
    }
)
```

### Create workflow


```python
import uuid

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore


model = ChatAnthropic(model="claude-3-5-sonnet-latest")


@task
def call_model(messages: list[BaseMessage], memory_store: BaseStore, user_id: str):
    namespace = ("memories", user_id)
    last_message = messages[-1]
    memories = memory_store.search(namespace, query=str(last_message.content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # Store new memories if the user asks the model to remember
    if "remember" in last_message.content.lower():
        memory = "User name is Bob"
        memory_store.put(namespace, str(uuid.uuid4()), {"data": memory})

    response = model.invoke([{"role": "system", "content": system_msg}] + messages)
    return response


# NOTE: we're passing the store object here when creating a workflow via entrypoint()
@entrypoint(checkpointer=MemorySaver(), store=in_memory_store)
def workflow(
    inputs: list[BaseMessage],
    *,
    previous: list[BaseMessage],
    config: RunnableConfig,
    store: BaseStore,
):
    user_id = config["configurable"]["user_id"]
    previous = previous or []
    inputs = add_messages(previous, inputs)
    response = call_model(inputs, store, user_id).result()
    return entrypoint.final(value=response, save=add_messages(inputs, response))
```

!!! note Note

    If you're using LangGraph Cloud or LangGraph Studio, you __don't need__ to pass store to the entrypoint decorator, since it's done automatically.

### Run the workflow!

Now let's specify a user ID in the config and tell the model our name:


```python
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
input_message = {"role": "user", "content": "Hi! Remember: my name is Bob"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```


```python
config = {"configurable": {"thread_id": "2", "user_id": "1"}}
input_message = {"role": "user", "content": "what is my name?"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```

We can now inspect our in-memory store and verify that we have in fact saved the memories for the user:


```python
for memory in in_memory_store.search(("memories", "1")):
    print(memory.value)
```

Let's now run the workflow for another user to verify that the memories about the first user are self contained:


```python
config = {"configurable": {"thread_id": "3", "user_id": "2"}}
input_message = {"role": "user", "content": "what is my name?"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```
