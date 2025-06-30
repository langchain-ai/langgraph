# LangGraph Memgraph Checkpoint

Implementation of a LangGraph CheckpointSaver and Store that uses [Memgraph](https://memgraph.com/).

## Overview

This package provides two main parts:

1. **Checkpoint Implementation**: `MemgraphSaver` and `AsyncMemgraphSaver` classes that implement the LangGraph checkpoint interface.
2. **Store Implementation**: `MemgraphStore` and `AsyncMemgraphStore` classes that provide a low-level key-value store interface.

The checkpoint implementation is built on top of the store implementation, providing a higher-level interface for managing checkpoints in LangGraph applications.

## Dependencies

The package pulls in `neo4j>=5.14` automatically. No native drivers or Memgraph-specific wheels are required.

## Setup

You will need to have a Memgraph instance running. The easiest way to do this is with Docker:

```bash
docker run -it --rm -p 7687:7687 -p 7444:7444 memgraph/memgraph-mage
```

The default user/password is `memgraph`/`memgraph`.

## Usage

> [!IMPORTANT]
> When using Memgraph implementations for the first time, make sure to call the `.setup()` method to create the required indexes and constraints.

### Checkpoint Usage

The checkpoint implementation provides a higher-level interface for managing LangGraph checkpoints:

```python
from langgraph.channels.topic import Topic
from langgraph.checkpoint.memgraph import MemgraphSaver
from langgraph.checkpoint.base import empty_checkpoint, create_checkpoint

# Configuration for identifying the thread/conversation
config = {
    "configurable": {
        "thread_id": "conversation-123",
        "checkpoint_id": "checkpoint-1",
        "checkpoint_ns": "",  # Root namespace
    }
}

DB_URI = "bolt://memgraph:memgraph@localhost:7687"

with MemgraphSaver.from_conn_string(DB_URI) as checkpointer:
    # call .setup() the first time you're using the checkpointer
    checkpointer.setup()

    # Create an empty initial checkpoint
    initial_checkpoint = empty_checkpoint()

    # Store the initial checkpoint with metadata
    config = checkpointer.put(
        config, 
        initial_checkpoint, 
        metadata={"source": "input", "step": 1},
        writes={}
    )

    # Create a new checkpoint based on the previous one
    next_checkpoint = create_checkpoint(
        initial_checkpoint,
        channels={"messages": Topic(value=["Hello, world!"])},
        step=1
    )

    # Store the new checkpoint
    config = checkpointer.put(
        config, 
        next_checkpoint, 
        metadata={"source": "processing", "step": 2},
        writes={}
    )

    # Load the latest checkpoint
    loaded_checkpoint_tuple = checkpointer.get_tuple(config)
    print(f"Loaded checkpoint ID: {loaded_checkpoint_tuple.checkpoint['id']}")
    print(f"Channel values: {loaded_checkpoint_tuple.checkpoint['channel_values']}")
    print(f"Metadata: {loaded_checkpoint_tuple.metadata}")

    # List all checkpoints for this thread
    thread_config = {"configurable": {"thread_id": "conversation-123"}}
    checkpoints = list(checkpointer.list(thread_config))
    print(f"Found {len(checkpoints)} checkpoints.")

    # The checkpoints are returned in reverse chronological order (newest first)
    for i, checkpoint_tuple in enumerate(checkpoints):
        print(f"Checkpoint {i+1}:")
        print(f"  ID: {checkpoint_tuple.checkpoint['id']}")
        print(f"  Step: {checkpoint_tuple.metadata.get('step')}")
        print(f"  Source: {checkpoint_tuple.metadata.get('source')}")
```

#### Searching Checkpoints

You can search for checkpoints using metadata filters:

```python
from langgraph.checkpoint.memgraph import MemgraphSaver

DB_URI = "bolt://memgraph:memgraph@localhost:7687"

with MemgraphSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    # Search by metadata
    input_checkpoints = list(checkpointer.list(
        config=None,  # None means search across all configs
        filter={"source": "input"}
    ))
    print(f"Found {len(input_checkpoints)} input checkpoints")

    # Search by step number
    step_2_checkpoints = list(checkpointer.list(
        config=None,
        filter={"step": 2}
    ))
    print(f"Found {len(step_2_checkpoints)} checkpoints at step 2")

    # Search by multiple metadata fields
    complex_filter = {
        "source": "processing",
        "step": {"$gt": 1}  # MongoDB-style operators supported
    }
    filtered_checkpoints = list(checkpointer.list(
        config=None,
        filter=complex_filter
    ))
    print(f"Found {len(filtered_checkpoints)} processing checkpoints after step 1")

    # Search by thread ID
    thread_checkpoints = list(checkpointer.list(
        config={"configurable": {"thread_id": "conversation-123"}}
    ))
    print(f"Found {len(thread_checkpoints)} checkpoints for thread conversation-123")
```

#### Working with Checkpoint Tasks

You can store and retrieve tasks associated with checkpoints:

```python
from langgraph.checkpoint.memgraph import MemgraphSaver
from langgraph.checkpoint.base import empty_checkpoint, create_checkpoint
from langgraph.checkpoint.serde.types import TASKS

DB_URI = "bolt://memgraph:memgraph@localhost:7687"

with MemgraphSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    # Create a config for a thread
    config = {
        "configurable": {
            "thread_id": "thread-tasks",
            "checkpoint_ns": "",
        }
    }

    # Create and store an initial checkpoint
    checkpoint_0 = empty_checkpoint()
    config = checkpointer.put(config, checkpoint_0, {}, {})

    # Store pending tasks linked to the checkpoint
    checkpointer.put_writes(
        config, 
        [(TASKS, "task-1"), (TASKS, "task-2")], 
        task_id="batch-1"
    )
    checkpointer.put_writes(
        config, 
        [(TASKS, "task-3")], 
        task_id="batch-2"
    )

    # Create a new checkpoint - tasks will be automatically associated with it
    checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
    config = checkpointer.put(config, checkpoint_1, {}, {})

    # Retrieve the checkpoint with tasks
    tuple_1 = checkpointer.get_tuple(config)
    print(f"Tasks in checkpoint: {tuple_1.checkpoint['channel_values'].get(TASKS)}")
```

### Store Usage

The store provides a lower-level key-value interface:

```python
from langgraph.store.memgraph import MemgraphStore

DB_URI = "bolt://memgraph:memgraph@localhost:7687"

with MemgraphStore.from_conn_string(DB_URI) as store:
    # call .setup() the first time you're using the store
    store.setup()

    # Define a namespace and key
    namespace = ("test", "documents")
    key = "doc1"
    value = {"title": "Test Document", "content": "Hello, World!"}

    # Store a value
    store.put(namespace, key, value)

    # Retrieve the value
    item = store.get(namespace, key)
    print(item.value)  # {"title": "Test Document", "content": "Hello, World!"}
    print(f"Key: {item.key}, Namespace: {item.namespace}")

    # Update a value
    updated_value = {"title": "Updated Document", "content": "Hello, Updated!"}
    store.put(namespace, key, updated_value)
    updated_item = store.get(namespace, key)
    print(f"Updated at: {updated_item.updated_at}")

    # Delete the value
    store.delete(namespace, key)
    deleted_item = store.get(namespace, key)
    print(f"After deletion: {deleted_item}")  # None
```

#### Searching and Filtering

You can search for items across namespaces with optional filtering:

```python
from langgraph.store.memgraph import MemgraphStore

DB_URI = "bolt://memgraph:memgraph@localhost:7687"

with MemgraphStore.from_conn_string(DB_URI) as store:
    store.setup()

    # Store some test data
    store.put(("test", "docs"), "doc1", 
              {"title": "First Doc", "author": "Alice", "tags": ["important"]})
    store.put(("test", "docs"), "doc2", 
              {"title": "Second Doc", "author": "Bob", "tags": ["draft"]})
    store.put(("test", "images"), "img1", 
              {"title": "Image 1", "author": "Alice", "tags": ["final"]})

    # Search all items in the "test" namespace
    all_items = store.search(["test"])
    print(f"Found {len(all_items)} total items")

    # Search with namespace prefix
    docs_items = store.search(["test", "docs"])
    print(f"Found {len(docs_items)} documents")

    # Search with filter
    alice_items = store.search(["test"], filter={"author": "Alice"})
    print(f"Found {len(alice_items)} items by Alice")

    # Search with pagination
    paginated_items = store.search(["test"], limit=2)
    print(f"Page 1: {[item.key for item in paginated_items]}")

    offset_items = store.search(["test"], offset=2)
    print(f"Page 2: {[item.key for item in offset_items]}")
```

#### Listing Namespaces

You can list and filter namespaces:

```python
from langgraph.store.memgraph import MemgraphStore

DB_URI = "bolt://memgraph:memgraph@localhost:7687"

with MemgraphStore.from_conn_string(DB_URI) as store:
    store.setup()

    # Create data in various namespaces
    namespaces = [
        ("test", "documents", "public"),
        ("test", "documents", "private"),
        ("test", "images", "public"),
        ("test", "images", "private"),
        ("prod", "documents", "public"),
    ]
    for namespace in namespaces:
        store.put(namespace, "dummy", {"content": "dummy"})

    # List all namespaces
    all_namespaces = store.list_namespaces()
    print(f"All namespaces: {all_namespaces}")

    # Filter by prefix
    test_namespaces = store.list_namespaces(prefix=("test",))
    print(f"Test namespaces: {test_namespaces}")

    # Filter by suffix
    public_namespaces = store.list_namespaces(suffix=("public",))
    print(f"Public namespaces: {public_namespaces}")

    # Limit depth
    depth_2_namespaces = store.list_namespaces(max_depth=2)
    print(f"Depth 2 namespaces: {depth_2_namespaces}")

    # Pagination
    paginated_namespaces = store.list_namespaces(limit=3)
    print(f"First 3 namespaces: {paginated_namespaces}")
```

#### Batch Operations

For better performance, you can execute multiple operations in a single batch:

```python
from langgraph.store.base import GetOp, PutOp, SearchOp, ListNamespacesOp
from langgraph.store.memgraph import MemgraphStore

DB_URI = "bolt://memgraph:memgraph@localhost:7687"

with MemgraphStore.from_conn_string(DB_URI) as store:
    store.setup()

    # Setup test data
    store.put(("test", "foo"), "key1", {"data": "value1"})
    store.put(("test", "bar"), "key2", {"data": "value2"})

    # Create a batch of operations
    ops = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "updated"}),
        SearchOp(namespace_prefix=("test",), filter={"data": "value1"}, limit=10),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10),
        GetOp(namespace=("test",), key="key3"),  # Non-existent key
    ]

    # Execute the batch
    results = store.batch(ops)

    # Process results
    print(f"Get result: {results[0].value}")  # First operation result
    print(f"Put result: {results[1]}")  # None for put operations
    print(f"Search result count: {len(results[2])}")
    print(f"List namespaces count: {len(results[3])}")
    print(f"Non-existent key result: {results[4]}")  # None
```

### Async Usage

#### Async Checkpoint

The async checkpoint implementation provides the same functionality as the sync version but with async/await support:

```python
import asyncio
from langgraph.checkpoint.memgraph.aio import AsyncMemgraphSaver
from langgraph.checkpoint.base import empty_checkpoint, create_checkpoint

# Configuration for identifying the thread/conversation
run_config = {
    "configurable": {
        "thread_id": "async-conversation-123",
        "checkpoint_id": "checkpoint-1",
        "checkpoint_ns": "",  # Root namespace
    }
}

DB_URI = "bolt://memgraph:memgraph@localhost:7687"

async def main():
    async with AsyncMemgraphSaver.from_conn_string(DB_URI) as checkpointer:
        # call .setup() the first time you're using the checkpointer
        await checkpointer.setup()

        # Create an empty initial checkpoint
        initial_checkpoint = empty_checkpoint()

        # Store the initial checkpoint with metadata
        config = await checkpointer.aput(
            run_config, 
            initial_checkpoint, 
            metadata={"source": "input", "step": 1},
            writes={}
        )

        # Create a new checkpoint based on the previous one
        next_checkpoint = create_checkpoint(
            initial_checkpoint,
            channel_values={"messages": ["Hello, async world!"]},
            version=1
        )

        # Store the new checkpoint
        config = await checkpointer.aput(
            config, 
            next_checkpoint, 
            metadata={"source": "processing", "step": 2},
            writes={}
        )

        # Load the latest checkpoint
        loaded_checkpoint_tuple = await checkpointer.aget_tuple(config)
        print(f"Loaded checkpoint ID: {loaded_checkpoint_tuple.checkpoint['id']}")
        print(f"Channel values: {loaded_checkpoint_tuple.checkpoint['channel_values']}")
        print(f"Metadata: {loaded_checkpoint_tuple.metadata}")

        # List all checkpoints for this thread
        thread_config = {"configurable": {"thread_id": "async-conversation-123"}}
        checkpoints = [c async for c in checkpointer.alist(thread_config)]
        print(f"Found {len(checkpoints)} checkpoints.")

        # Search by metadata
        input_checkpoints = [c async for c in checkpointer.alist(
            config=None,  # None means search across all configs
            filter={"source": "input"}
        )]
        print(f"Found {len(input_checkpoints)} input checkpoints")

        # Store and retrieve tasks
        from langgraph.checkpoint.serde.types import TASKS

        # Store pending tasks linked to the checkpoint
        await checkpointer.aput_writes(
            config, 
            [(TASKS, "async-task-1"), (TASKS, "async-task-2")], 
            task_id="async-batch-1"
        )

        # Create a new checkpoint - tasks will be automatically associated with it
        final_checkpoint = create_checkpoint(next_checkpoint, {}, 2)
        config = await checkpointer.aput(config, final_checkpoint, {}, {})

        # Retrieve the checkpoint with tasks
        final_tuple = await checkpointer.aget_tuple(config)
        print(f"Tasks in checkpoint: {final_tuple.checkpoint['channel_values'].get(TASKS)}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Async Store

The async store implementation provides the same functionality as the sync version but with async/await support:

```python
import asyncio
from langgraph.store.memgraph import AsyncMemgraphStore

DB_URI = "bolt://memgraph:memgraph@localhost:7687"

async def main():
    async with AsyncMemgraphStore.from_conn_string(DB_URI) as store:
        # call .setup() the first time you're using the store
        await store.setup()

        # Define a namespace and key
        namespace = ("test", "documents")
        key = "doc1"
        value = {"title": "Test Document", "content": "Hello, Async World!"}

        # Store a value
        await store.aput(namespace, key, value)

        # Retrieve the value
        item = await store.aget(namespace, key)
        print(f"Value: {item.value}")
        print(f"Key: {item.key}, Namespace: {item.namespace}")

        # Update a value
        updated_value = {"title": "Updated Document", "content": "Hello, Updated Async!"}
        await store.aput(namespace, key, updated_value)
        updated_item = await store.aget(namespace, key)
        print(f"Updated at: {updated_item.updated_at}")

        # Delete the value
        await store.adelete(namespace, key)
        deleted_item = await store.aget(namespace, key)
        print(f"After deletion: {deleted_item}")  # None

        # Store some test data for search examples
        await store.aput(("test", "docs"), "doc1", 
                  {"title": "First Doc", "author": "Alice", "tags": ["important"]})
        await store.aput(("test", "docs"), "doc2", 
                  {"title": "Second Doc", "author": "Bob", "tags": ["draft"]})
        await store.aput(("test", "images"), "img1", 
                  {"title": "Image 1", "author": "Alice", "tags": ["final"]})

        # Search all items in the "test" namespace
        all_items = await store.asearch(["test"])
        print(f"Found {len(all_items)} total items")

        # Search with filter
        alice_items = await store.asearch(["test"], filter={"author": "Alice"})
        print(f"Found {len(alice_items)} items by Alice")

        # Batch operations
        from langgraph.store.base import GetOp, PutOp, SearchOp, ListNamespacesOp

        ops = [
            GetOp(namespace=("test", "docs"), key="doc1"),
            SearchOp(namespace_prefix=("test",), filter={"author": "Alice"}, limit=10),
            ListNamespacesOp(match_conditions=None, max_depth=None, limit=10),
        ]

        results = await store.abatch(ops)
        print(f"Batch results: {len(results)} operations completed")
        print(f"First doc title: {results[0].value['title']}")
        print(f"Alice items count: {len(results[1])}")
        print(f"Namespaces count: {len(results[2])}")

if __name__ == "__main__":
    asyncio.run(main())
```

##### Async Vector Search

Vector search works the same way with async:

```python
import asyncio
from langgraph.store.memgraph import AsyncMemgraphStore, MemgraphIndexConfig
from langchain_core.embeddings import Embeddings

# Simple embeddings model for demonstration
class SimpleEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(ord(c)) for c in text[:10].ljust(10)] for text in texts]

    def embed_query(self, text):
        return [float(ord(c)) for c in text[:10].ljust(10)]

async def main():
    # Configure vector indexing
    index_config = MemgraphIndexConfig(
        dimension=10,
        capacity=1000,
        embed=SimpleEmbeddings(),
        metric="cos",
        fields=["text"],
    )

    async with AsyncMemgraphStore.from_conn_string(
        "bolt://memgraph:memgraph@localhost:7687",
        index=index_config
    ) as store:
        await store.setup()

        # Insert documents with automatic embedding
        docs = [
            ("doc1", {"text": "short text"}),
            ("doc2", {"text": "longer text document"}),
            ("doc3", {"text": "longest text document here"}),
        ]
        for key, value in docs:
            await store.aput(("test",), key, value, index=["text"])

        # Search by semantic similarity
        results = await store.asearch(("test",), query="long text")
        print(f"Search results: {[r.key for r in results]}")

        # Search with filters
        filtered_results = await store.asearch(
            ("test",), 
            query="document", 
            filter={"text": {"$regex": ".*longest.*"}}
        )
        print(f"Filtered results: {[r.key for r in filtered_results]}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Features

#### Vector Search

You can use Memgraph for vector search with automatic embedding:

```python
from langgraph.store.memgraph import MemgraphStore, MemgraphIndexConfig
from langchain_core.embeddings import Embeddings

# Define a simple embeddings model (use your preferred embeddings model)
class SimpleEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # Simple embedding function for demonstration
        return [[float(ord(c)) for c in text[:10].ljust(10)] for text in texts]

    def embed_query(self, text):
        # Simple embedding function for demonstration
        return [float(ord(c)) for c in text[:10].ljust(10)]

# Configure vector indexing
index_config = MemgraphIndexConfig(
    dimension=10,  # Dimension of your embeddings
    capacity=1000,  # Maximum number of vectors to index
    embed=SimpleEmbeddings(),  # Embeddings model
    metric="cos",  # Similarity metric: "cos" (cosine), "ip" (inner product), or "l2sq" (L2 squared)
    fields=["text"],  # Default fields to embed
)

# Create store with vector indexing
with MemgraphStore.from_conn_string(
    "bolt://memgraph:memgraph@localhost:7687",
    index=index_config
) as store:
    store.setup()
    
    # Insert documents with automatic embedding
    docs = [
        ("doc1", {"text": "short text"}),
        ("doc2", {"text": "longer text document"}),
        ("doc3", {"text": "longest text document here"}),
    ]
    for key, value in docs:
        store.put(("test",), key, value, index=["text"])  # Specify fields to index
    
    # Search by semantic similarity
    results = store.search(("test",), query="long text")
    print(f"Search results: {[r.key for r in results]}")
    print(f"Top result score: {results[0].score}")
    
    # Search with filters
    filtered_results = store.search(
        ("test",), 
        query="document", 
        filter={"text": {"$regex": ".*longest.*"}}
    )
    print(f"Filtered results: {[r.key for r in filtered_results]}")
    
    # Update a document (automatically updates the vector)
    store.put(("test",), "doc1", {"text": "new text about documents"}, index=["text"])
    
    # Skip indexing for a document
    store.put(("test",), "doc4", {"text": "unindexed document"}, index=False)
```

##### Advanced Vector Search Configuration

For more control over vector indexing:

```python
from langgraph.store.memgraph import MemgraphStore, MemgraphIndexConfig

# Configure indexing for specific fields
index_config = MemgraphIndexConfig(
    dimension=1536,  # Common dimension for many embedding models
    capacity=10000,
    metric="cos",
    fields=["description", "content.text"],  # Nested fields supported with dot notation
)

# Create store with indexing
with MemgraphStore.from_conn_string(
    "bolt://memgraph:memgraph@localhost:7687",
    index=index_config
) as store:
    store.setup()
    
    # Store documents with nested fields
    store.put(
        ("products",), 
        "product1", 
        {
            "name": "Smartphone",
            "description": "High-end smartphone with advanced features",
            "content": {
                "text": "This smartphone features a 6.7-inch display, 5G connectivity, and all-day battery life."
            }
        }
    )
    
    # Search using specific fields for this operation
    store.search(
        ("products",),
        query="phone with long battery",
        fields=["description", "content.text"]  # Override default fields for this search
    )
```

#### TTL (Time-To-Live) Configuration

You can configure automatic deletion of old data:

```python
from langgraph.store.memgraph import MemgraphStore
from langgraph.store.base import TTLConfig
import time

# Configure TTL
ttl_config = TTLConfig(
    default_ttl=3600,  # Delete data after 1 hour
    refresh_on_read=True  # Refresh TTL when item is read
)

# Create store with TTL
with MemgraphStore.from_conn_string(
    "bolt://memgraph:memgraph@localhost:7687",
    ttl=ttl_config
) as store:

    # Start TTL sweeper (runs in background)
    store.start_ttl_sweeper(sweep_interval_minutes=5)
    
    try:
        # Store an item with TTL
        namespace = ("cache",)
        store.put(namespace, key="item1", value={"data": "temporary"}, ttl=10/60)  # 10 seconds
    
        # Item is available immediately
        item = store.get(namespace, key="item1")
        print(f"Item exists: {item is not None}")  # True
    
        # Wait for TTL to expire
        time.sleep(11)  # Wait 11 seconds
    
        # Item should be gone
        expired_item = store.get(namespace, key="item1")
        print(f"Item after expiration: {expired_item}")  # None
    
        # Demonstrate TTL refresh
        store.put(namespace, key="item2", value={"data": "refreshable"}, ttl=10/60)  # 10 seconds
    
        # Wait half the TTL time
        time.sleep(5)
    
        # Read the item with refresh_ttl=True to extend its lifetime
        refreshed_item = store.get(namespace, key="item2", refresh_ttl=True)
        print(f"Item refreshed: {refreshed_item is not None}")  # True
    
        # Wait another 5 seconds (would have expired without refresh)
        time.sleep(5)
    
        # Item should still exist because TTL was refreshed
        still_exists = store.get(namespace, key="item2", refresh_ttl=False)
        print(f"Item still exists: {still_exists is not None}")  # True
    
        # Wait for the refreshed TTL to expire
        time.sleep(6)
    
        # Now it should be gone
        finally_expired = store.get(namespace, key="item2")
        print(f"Item finally expired: {finally_expired is None}")  # True
    finally:
        # Always stop the TTL sweeper when done
        store.stop_ttl_sweeper()
```

##### TTL with Search Operations

TTL can also be refreshed during search operations:

```python
from langgraph.store.memgraph import MemgraphStore
from langgraph.store.base import TTLConfig

# Configure TTL with refresh on read
ttl_config = TTLConfig(
    default_ttl=3600,
    refresh_on_read=True
)

with MemgraphStore.from_conn_string(
    "bolt://memgraph:memgraph@localhost:7687",
    ttl=ttl_config
) as store:
    store.setup()
    store.start_ttl_sweeper()
    
    try:
        # Store items with TTL
        store.put(("docs",), "doc1", {"title": "Temporary Doc"}, ttl=60/60)  # 60 seconds
    
        # Search with refresh_ttl=True to extend TTL of matching items
        results = store.search(("docs",), refresh_ttl=True)
    
        print(f"Found {len(results)} items, TTL refreshed")
    finally:
        store.stop_ttl_sweeper()
```

#### Bring-your-own Neo4j driver

If you need explicit control over the connection (e.g., connection pooling, SSL settings), you can create the driver yourself and pass it in:

```python
from neo4j import GraphDatabase
from langgraph.checkpoint.memgraph import MemgraphSaver

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("memgraph", "memgraph"),
    max_connection_lifetime=180,  # custom parameter
)
try:
    checkpointer = MemgraphSaver(driver)
    checkpointer.setup()
    # ... use checkpointer as needed
finally:
    driver.close()
```

The async version works similarly with `AsyncGraphDatabase.driver`.
