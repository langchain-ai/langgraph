# How to add semantic search to your LangGraph deployment

This guide explains how to add semantic search to your LangGraph deployment's cross-thread [store](../../concepts/persistence.md#memory-store), so that your agent can search for memories and other documents by semantic similarity.

## Prerequisites

- A LangGraph deployment (see [how to deploy](setup_pyproject.md))
- API keys for your embedding provider (in this case, OpenAI)
- `langchain >= 0.3.8` (if you specify using the string format below)

## Steps

1. Update your `langgraph.json` configuration file to include the store configuration:

```json
{
    ...
    "store": {
        "index": {
            "embed": "openai:text-embedding-3-small",
            "dims": 1536,
            "fields": ["$"]
        }
    }
}
```

This configuration:

- Uses OpenAI's text-embedding-3-small model for generating embeddings
- Sets the embedding dimension to 1536 (matching the model's output)
- Indexes all fields in your stored data (`["$"]` means index everything, or specify specific fields like `["text", "metadata.title"]`)

2. To use the string embedding format above, make sure your dependencies include `langchain >= 0.3.8`:

```toml
# In pyproject.toml
[project]
dependencies = [
    "langchain>=0.3.8"
]
```

Or if using requirements.txt:

```
langchain>=0.3.8
```

## Usage

Once configured, you can use semantic search in your LangGraph nodes. The store requires a namespace tuple to organize memories:

```python
def search_memory(state: State, *, store: BaseStore):
    # Search the store using semantic similarity
    # The namespace tuple helps organize different types of memories
    # e.g., ("user_facts", "preferences") or ("conversation", "summaries")
    results = store.search(
        namespace=("memory", "facts"),  # Organize memories by type
        query="your search query",
        limit=3  # number of results to return
    )
    return results
```

## Custom Embeddings

If you want to use custom embeddings, you can pass a path to a custom embedding function:

```json
{
    ...
    "store": {
        "index": {
            "embed": "path/to/embedding_function.py:embed",
            "dims": 1536,
            "fields": ["$"]
        }
    }
}
```

The deployment will look for the function in the specified path. The function must be async and accept a list of strings:

```python
# path/to/embedding_function.py
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def aembed_texts(texts: list[str]) -> list[list[float]]:
    """Custom embedding function that must:
    1. Be async
    2. Accept a list of strings
    3. Return a list of float arrays (embeddings)
    """
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in response.data]
```

## Querying via the API

You can also query the store using the LangGraph SDK. Since the SDK uses async operations:

```python
from langgraph_sdk import get_client

async def search_store():
    client = get_client()
    results = await client.store.search_items(
        ("memory", "facts"),
        query="your search query",
        limit=3  # number of results to return
    )
    return results

# Use in an async context
results = await search_store()
```
