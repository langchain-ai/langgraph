# How to add semantic search to your agent's memory

This guide shows how to enable semantic search in your agent's memory store. This lets search for items in the store by semantic similarity.

!!! tip Prerequisites
    This guide assumes familiarity with the [memory in LangGraph](https://langchain-ai.github.io/langgraph/concepts/memory/).

First, install this guide's prerequisites.


```
%%capture --no-stderr
%pip install -U langgraph langchain-openai langchain
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

Next, create the store with an [index configuration](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.IndexConfig). By default, stores are configured without semantic/vector search. You can opt in to indexing items when creating the store by providing an [IndexConfig](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.IndexConfig) to the store's constructor. If your store class does not implement this interface, or if you do not pass in an index configuration, semantic search is disabled, and all `index` arguments passed to `put` or `aput` will have no effect. Below is an example.


```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

# Create store with semantic search enabled
embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)
```

Now let's store some memories:


```python
# Store some memories
store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I prefer Italian food"})
store.put(("user_123", "memories"), "3", {"text": "I don't like spicy food"})
store.put(("user_123", "memories"), "3", {"text": "I am studying econometrics"})
store.put(("user_123", "memories"), "3", {"text": "I am a plumber"})
```

Search memories using natural language:


```python
# Find memories about food preferences
memories = store.search(("user_123", "memories"), query="I like food?", limit=5)

for memory in memories:
    print(f'Memory: {memory.value["text"]} (similarity: {memory.score})')
```

## Using in your agent

Add semantic search to any node by injecting the store.


```python
from typing import Optional

from langchain.chat_models import init_chat_model
from langgraph.store.base import BaseStore

from langgraph.graph import START, MessagesState, StateGraph

llm = init_chat_model("openai:gpt-4o-mini")


def chat(state, *, store: BaseStore):
    # Search based on user's last message
    items = store.search(
        ("user_123", "memories"), query=state["messages"][-1].content, limit=2
    )
    memories = "\n".join(item.value["text"] for item in items)
    memories = f"## Memories of user\n{memories}" if memories else ""
    response = llm.invoke(
        [
            {"role": "system", "content": f"You are a helpful assistant.\n{memories}"},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node(chat)
builder.add_edge(START, "chat")
graph = builder.compile(store=store)

for message, metadata in graph.stream(
    input={"messages": [{"role": "user", "content": "I'm hungry"}]},
    stream_mode="messages",
):
    print(message.content, end="")
```

## Using in `create_react_agent`

Add semantic search to your tool calling agent by injecting the store in the `prompt` function. You can also use the store in a tool to let your agent manually store or search for memories.


```python
import uuid
from typing import Optional

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from typing_extensions import Annotated

from langgraph.prebuilt import create_react_agent


def prepare_messages(state, *, store: BaseStore):
    # Search based on user's last message
    items = store.search(
        ("user_123", "memories"), query=state["messages"][-1].content, limit=2
    )
    memories = "\n".join(item.value["text"] for item in items)
    memories = f"## Memories of user\n{memories}" if memories else ""
    return [
        {"role": "system", "content": f"You are a helpful assistant.\n{memories}"}
    ] + state["messages"]


# You can also use the store directly within a tool!
def upsert_memory(
    content: str,
    *,
    memory_id: Optional[uuid.UUID] = None,
    store: Annotated[BaseStore, InjectedStore],
):
    """Upsert a memory in the database."""
    # The LLM can use this tool to store a new memory
    mem_id = memory_id or uuid.uuid4()
    store.put(
        ("user_123", "memories"),
        key=str(mem_id),
        value={"text": content},
    )
    return f"Stored memory {mem_id}"


agent = create_react_agent(
    init_chat_model("openai:gpt-4o-mini"),
    tools=[upsert_memory],
    # The 'prompt' function is run to prepare the messages for the LLM. It is called
    # right before each LLM call
    prompt=prepare_messages,
    store=store,
)
```


```python
for message, metadata in agent.stream(
    input={"messages": [{"role": "user", "content": "I'm hungry"}]},
    stream_mode="messages",
):
    print(message.content, end="")
```

## Advanced Usage

#### Multi-vector indexing

Store and search different aspects of memories separately to improve recall or omit certain fields from being indexed.


```python
# Configure store to embed both memory content and emotional context
store = InMemoryStore(
    index={"embed": embeddings, "dims": 1536, "fields": ["memory", "emotional_context"]}
)
# Store memories with different content/emotion pairs
store.put(
    ("user_123", "memories"),
    "mem1",
    {
        "memory": "Had pizza with friends at Mario's",
        "emotional_context": "felt happy and connected",
        "this_isnt_indexed": "I prefer ravioli though",
    },
)
store.put(
    ("user_123", "memories"),
    "mem2",
    {
        "memory": "Ate alone at home",
        "emotional_context": "felt a bit lonely",
        "this_isnt_indexed": "I like pie",
    },
)

# Search focusing on emotional state - matches mem2
results = store.search(
    ("user_123", "memories"), query="times they felt isolated", limit=1
)
print("Expect mem 2")
for r in results:
    print(f"Item: {r.key}; Score ({r.score})")
    print(f"Memory: {r.value['memory']}")
    print(f"Emotion: {r.value['emotional_context']}\n")

# Search focusing on social eating - matches mem1
print("Expect mem1")
results = store.search(("user_123", "memories"), query="fun pizza", limit=1)
for r in results:
    print(f"Item: {r.key}; Score ({r.score})")
    print(f"Memory: {r.value['memory']}")
    print(f"Emotion: {r.value['emotional_context']}\n")

print("Expect random lower score (ravioli not indexed)")
results = store.search(("user_123", "memories"), query="ravioli", limit=1)
for r in results:
    print(f"Item: {r.key}; Score ({r.score})")
    print(f"Memory: {r.value['memory']}")
    print(f"Emotion: {r.value['emotional_context']}\n")
```

#### Override fields at storage time
You can override which fields to embed when storing a specific memory using `put(..., index=[...fields])`, regardless of the store's default configuration.


```python
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["memory"],
    }  # Default to embed memory field
)

# Store one memory with default indexing
store.put(
    ("user_123", "memories"),
    "mem1",
    {"memory": "I love spicy food", "context": "At a Thai restaurant"},
)

# Store another overriding which fields to embed
store.put(
    ("user_123", "memories"),
    "mem2",
    {"memory": "The restaurant was too loud", "context": "Dinner at an Italian place"},
    index=["context"],  # Override: only embed the context
)

# Search about food - matches mem1 (using default field)
print("Expect mem1")
results = store.search(
    ("user_123", "memories"), query="what food do they like", limit=1
)
for r in results:
    print(f"Item: {r.key}; Score ({r.score})")
    print(f"Memory: {r.value['memory']}")
    print(f"Context: {r.value['context']}\n")

# Search about restaurant atmosphere - matches mem2 (using overridden field)
print("Expect mem2")
results = store.search(
    ("user_123", "memories"), query="restaurant environment", limit=1
)
for r in results:
    print(f"Item: {r.key}; Score ({r.score})")
    print(f"Memory: {r.value['memory']}")
    print(f"Context: {r.value['context']}\n")
```

#### Disable Indexing for Specific Memories

Some memories shouldn't be searchable by content. You can disable indexing for these while still storing them using 
`put(..., index=False)`. Example:


```python
store = InMemoryStore(index={"embed": embeddings, "dims": 1536, "fields": ["memory"]})

# Store a normal indexed memory
store.put(
    ("user_123", "memories"),
    "mem1",
    {"memory": "I love chocolate ice cream", "type": "preference"},
)

# Store a system memory without indexing
store.put(
    ("user_123", "memories"),
    "mem2",
    {"memory": "User completed onboarding", "type": "system"},
    index=False,  # Disable indexing entirely
)

# Search about food preferences - finds mem1
print("Expect mem1")
results = store.search(("user_123", "memories"), query="what food preferences", limit=1)
for r in results:
    print(f"Item: {r.key}; Score ({r.score})")
    print(f"Memory: {r.value['memory']}")
    print(f"Type: {r.value['type']}\n")

# Search about onboarding - won't find mem2 (not indexed)
print("Expect low score (mem2 not indexed)")
results = store.search(("user_123", "memories"), query="onboarding status", limit=1)
for r in results:
    print(f"Item: {r.key}; Score ({r.score})")
    print(f"Memory: {r.value['memory']}")
    print(f"Type: {r.value['type']}\n")
```
