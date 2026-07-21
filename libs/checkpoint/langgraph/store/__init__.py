"""LangGraph persistent key-value store.

Stores provide long-term, cross-thread memory for agents. Unlike checkpointers,
which save per-thread graph state, stores hold arbitrary key-value data that is
shared across all threads and survives beyond a single conversation.

Core abstractions:

- `BaseStore`: abstract base class for all store implementations.
- `InMemoryStore`: in-memory implementation suitable for development and testing.
- `Item`: a stored value with namespace, key, and timestamp metadata.
- `SearchItem`: an `Item` with an additional relevance score from semantic search.

Operations are expressed as data types and batched via `BaseStore.batch` /
`BaseStore.abatch`, or issued individually through the convenience helpers
`get`, `put`, `delete`, `search`, and `list_namespaces`.

Example::

    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    store.put(("users", "alice"), "preferences", {"theme": "dark"})
    item = store.get(("users", "alice"), "preferences")
    print(item.value)  # {"theme": "dark"}

For production, use a persistent backend such as `PostgresStore` (from
`langgraph-checkpoint-postgres`) or `SqliteStore` (from
`langgraph-checkpoint-sqlite`).
"""

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    MatchCondition,
    NamespaceMatchType,
    NamespacePath,
    NotProvided,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
)
from langgraph.store.base.embed import (
    AEmbeddingsFunc,
    EmbeddingsFunc,
    EmbeddingsLambda,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)
from langgraph.store.memory import InMemoryStore

__all__ = [
    # Core abstractions
    "BaseStore",
    "Item",
    "SearchItem",
    # Operation types
    "GetOp",
    "ListNamespacesOp",
    "Op",
    "PutOp",
    "SearchOp",
    # Configuration
    "IndexConfig",
    "TTLConfig",
    # Namespace types
    "MatchCondition",
    "NamespaceMatchType",
    "NamespacePath",
    # Exceptions and sentinels
    "InvalidNamespaceError",
    "NotProvided",
    # Type aliases
    "Result",
    # Embedding helpers
    "AEmbeddingsFunc",
    "EmbeddingsFunc",
    "EmbeddingsLambda",
    "ensure_embeddings",
    "get_text_at_path",
    "tokenize_path",
    # Implementations
    "InMemoryStore",
]
