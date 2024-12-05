"""Base classes and types for persistent key-value stores.

Stores provide long-term memory that persists across threads and conversations.
Supports hierarchical namespaces, key-value storage, and optional vector search.

Core types:
    - BaseStore: Store interface with sync/async operations
    - Item: Stored key-value pairs with metadata
    - Op: Get/Put/Search/List operations
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable, Literal, NamedTuple, Optional, TypedDict, Union, cast

from langchain_core.embeddings import Embeddings

from langgraph.store.base.embed import (
    AEmbeddingsFunc,
    EmbeddingsFunc,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)


class Item:
    """Represents a stored item with metadata.

    Args:
        value (dict[str, Any]): The stored data as a dictionary. Keys are filterable.
        key (str): Unique identifier within the namespace.
        namespace (tuple[str, ...]): Hierarchical path defining the collection in which this document resides.
            Represented as a tuple of strings, allowing for nested categorization.
            For example: ("documents", 'user123')
        created_at (datetime): Timestamp of item creation.
        updated_at (datetime): Timestamp of last update.
    """

    __slots__ = ("value", "key", "namespace", "created_at", "updated_at")

    def __init__(
        self,
        *,
        value: dict[str, Any],
        key: str,
        namespace: tuple[str, ...],
        created_at: datetime,
        updated_at: datetime,
    ):
        self.value = value
        self.key = key
        # The casting from json-like types is for if this object is
        # deserialized.
        self.namespace = tuple(namespace)
        self.created_at = (
            datetime.fromisoformat(cast(str, created_at))
            if isinstance(created_at, str)
            else created_at
        )
        self.updated_at = (
            datetime.fromisoformat(cast(str, created_at))
            if isinstance(updated_at, str)
            else updated_at
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Item):
            return False
        return (
            self.value == other.value
            and self.key == other.key
            and self.namespace == other.namespace
            and self.created_at == other.created_at
            and self.updated_at == other.updated_at
        )

    def __hash__(self) -> int:
        return hash((self.namespace, self.key))

    def dict(self) -> dict:
        return {
            "namespace": list(self.namespace),
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        return f"Item({', '.join(f'{k}={v!r}' for k, v in self.dict().items())})"


class SearchItem(Item):
    """Represents an item returned from a search operation with additional metadata."""

    __slots__ = ("score",)

    def __init__(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        created_at: datetime,
        updated_at: datetime,
        score: Optional[float] = None,
    ) -> None:
        """Initialize a result item.

        Args:
            namespace: Hierarchical path to the item.
            key: Unique identifier within the namespace.
            value: The stored value.
            created_at: When the item was first created.
            updated_at: When the item was last updated.
            score: Relevance/similarity score if from a ranked operation.
        """
        super().__init__(
            value=value,
            key=key,
            namespace=namespace,
            created_at=created_at,
            updated_at=updated_at,
        )
        self.score = score

    def dict(self) -> dict:
        result = super().dict()
        result["score"] = self.score
        return result


class GetOp(NamedTuple):
    """Operation to retrieve a specific item by its namespace and key.

    This operation allows precise retrieval of stored items using their full path
    (namespace) and unique identifier (key) combination.

    ???+ example "Examples"

        Basic item retrieval:
        ```python
        GetOp(namespace=("users", "profiles"), key="user123")
        GetOp(namespace=("cache", "embeddings"), key="doc456")
        ```
    """

    namespace: tuple[str, ...]
    """Hierarchical path that uniquely identifies the item's location.

    ???+ example "Examples"

        ```python
        ("users",)  # Root level users namespace
        ("users", "profiles")  # Profiles within users namespace
        ```
    """

    key: str
    """Unique identifier for the item within its specific namespace.

    ???+ example "Examples"

        ```python
        "user123"  # For a user profile
        "doc456"  # For a document
        ```
    """


class SearchOp(NamedTuple):
    """Operation to search for items within a specified namespace hierarchy.

    This operation supports both structured filtering and natural language search
    within a given namespace prefix. It provides pagination through limit and offset
    parameters.

    Note:
        Natural language search support depends on your store implementation.

    ???+ example "Examples"
        Search with filters and pagination:
        ```python
        SearchOp(
            namespace_prefix=("documents",),
            filter={"type": "report", "status": "active"},
            limit=5,
            offset=10
        )
        ```

        Natural language search:
        ```python
        SearchOp(
            namespace_prefix=("users", "content"),
            query="technical documentation about APIs",
            limit=20
        )
        ```
    """

    namespace_prefix: tuple[str, ...]
    """Hierarchical path prefix defining the search scope.

    ???+ example "Examples"

        ```python
        ()  # Search entire store
        ("documents",)  # Search all documents
        ("users", "content")  # Search within user content
        ```
    """

    filter: Optional[dict[str, Any]] = None
    """Key-value pairs for filtering results based on exact matches or comparison operators.

    The filter supports both exact matches and operator-based comparisons.

    Supported Operators:
        - $eq: Equal to (same as direct value comparison)
        - $ne: Not equal to
        - $gt: Greater than
        - $gte: Greater than or equal to
        - $lt: Less than
        - $lte: Less than or equal to

    ???+ example "Examples"
        Simple exact match:

        ```python
        {"status": "active"}
        ```

        Comparison operators:

        ```python
        {"score": {"$gt": 4.99}}  # Score greater than 4.99
        ```

        Multiple conditions:

        ```python
        {
            "score": {"$gte": 3.0},
            "color": "red"
        }
        ```
    """

    limit: int = 10
    """Maximum number of items to return in the search results."""

    offset: int = 0
    """Number of matching items to skip for pagination."""

    query: Optional[str] = None
    """Natural language search query for semantic search capabilities.

    ???+ example "Examples"
        - "technical documentation about REST APIs"
        - "machine learning papers from 2023"
    """


# Type representing a namespace path that can include wildcards
NamespacePath = tuple[Union[str, Literal["*"]], ...]
"""A tuple representing a namespace path that can include wildcards.

???+ example "Examples"
    ```python
    ("users",)  # Exact users namespace
    ("documents", "*")  # Any sub-namespace under documents
    ("cache", "*", "v1")  # Any cache category with v1 version
    ```
"""

# Type for specifying how to match namespaces
NamespaceMatchType = Literal["prefix", "suffix"]
"""Specifies how to match namespace paths.

Values:
    "prefix": Match from the start of the namespace
    "suffix": Match from the end of the namespace
"""


class MatchCondition(NamedTuple):
    """Represents a pattern for matching namespaces in the store.

    This class combines a match type (prefix or suffix) with a namespace path
    pattern that can include wildcards to flexibly match different namespace
    hierarchies.

    ???+ example "Examples"
        Prefix matching:
        ```python
        MatchCondition(match_type="prefix", path=("users", "profiles"))
        ```

        Suffix matching with wildcard:
        ```python
        MatchCondition(match_type="suffix", path=("cache", "*"))
        ```

        Simple suffix matching:
        ```python
        MatchCondition(match_type="suffix", path=("v1",))
        ```
    """

    match_type: NamespaceMatchType
    """Type of namespace matching to perform."""

    path: NamespacePath
    """Namespace path pattern that can include wildcards."""


class ListNamespacesOp(NamedTuple):
    """Operation to list and filter namespaces in the store.

    This operation allows exploring the organization of data, finding specific
    collections, and navigating the namespace hierarchy.

    ???+ example "Examples"

        List all namespaces under the "documents" path:
        ```python
        ListNamespacesOp(
            match_conditions=(MatchCondition(match_type="prefix", path=("documents",)),),
            max_depth=2
        )
        ```

        List all namespaces that end with "v1":
        ```python
        ListNamespacesOp(
            match_conditions=(MatchCondition(match_type="suffix", path=("v1",)),),
            limit=50
        )
        ```

    """

    match_conditions: Optional[tuple[MatchCondition, ...]] = None
    """Optional conditions for filtering namespaces.

    ???+ example "Examples"
        All user namespaces:
        ```python
        (MatchCondition(match_type="prefix", path=("users",)),)
        ```

        All namespaces that start with "docs" and end with "draft":
        ```python
        (
            MatchCondition(match_type="prefix", path=("docs",)),
            MatchCondition(match_type="suffix", path=("draft",))
        ) 
        ```
    """

    max_depth: Optional[int] = None
    """Maximum depth of namespace hierarchy to return.

    Note:
        Namespaces deeper than this level will be truncated.
    """

    limit: int = 100
    """Maximum number of namespaces to return."""

    offset: int = 0
    """Number of namespaces to skip for pagination."""


class PutOp(NamedTuple):
    """Operation to store, update, or delete an item in the store.

    This class represents a single operation to modify the store's contents,
    whether adding new items, updating existing ones, or removing them.
    """

    namespace: tuple[str, ...]
    """Hierarchical path that identifies the location of the item.

    The namespace acts as a folder-like structure to organize items.
    Each element in the tuple represents one level in the hierarchy.

    ???+ example "Examples"
        Root level documents
        ```python
        ("documents",)
        ```
        
        User-specific documents
        ```python
        ("documents", "user123")
        ```
        
        Nested cache structure
        ```python
        ("cache", "embeddings", "v1")
        ```
    """

    key: str
    """Unique identifier for the item within its namespace.

    The key must be unique within the specific namespace to avoid conflicts.
    Together with the namespace, it forms a complete path to the item.

    Example:
        If namespace is ("documents", "user123") and key is "report1",
        the full path would effectively be "documents/user123/report1"
    """

    value: Optional[dict[str, Any]]
    """The data to store, or None to mark the item for deletion.

    The value must be a dictionary with string keys and JSON-serializable values.
    Setting this to None signals that the item should be deleted.

    Example:
        {
            "field1": "string value",
            "field2": 123,
            "nested": {"can": "contain", "any": "serializable data"}
        }
    """

    index: Optional[Union[Literal[False], list[str]]] = None  # type: ignore[assignment]
    """Controls how the item's fields are indexed for search operations.

    Indexing configuration determines how the item can be found through search:
        - None (default): Uses the store's default indexing configuration (if provided)
        - False: Disables indexing for this item
        - list[str]: Specifies which json path fields to index for search

    The item remains accessible through direct get() operations regardless of indexing.
    When indexed, fields can be searched using natural language queries through
    vector similarity search (if supported by the store implementation).

    Path Syntax:
        - Simple field access: "field"
        - Nested fields: "parent.child.grandchild"
        - Array indexing:
          - Specific index: "array[0]"
          - Last element: "array[-1]"
          - All elements (each individually): "array[*]"

    ???+ example "Examples"
        - None - Use store defaults (whole item)
        - list[str] - List of fields to index
        
        ```python
        [
            "metadata.title",                    # Nested field access
            "context[*].content",                # Index content from all context as separate vectors
            "authors[0].name",                   # First author's name
            "revisions[-1].changes",             # Most recent revision's changes
            "sections[*].paragraphs[*].text",    # All text from all paragraphs in all sections
            "metadata.tags[*]",                  # All tags in metadata
        ]
        ```
    """


Op = Union[GetOp, SearchOp, PutOp, ListNamespacesOp]
Result = Union[Item, list[Item], list[SearchItem], list[tuple[str, ...]], None]


class InvalidNamespaceError(ValueError):
    """Provided namespace is invalid."""


class IndexConfig(TypedDict, total=False):
    """Configuration for indexing documents for semantic search in the store.

    If not provided to the store, the store will not support vector search.
    In that case, all `index` arguments to put() and `aput()` operations will be ignored.
    """

    dims: int
    """Number of dimensions in the embedding vectors.
    
    Common embedding models have the following dimensions:
        - openai:text-embedding-3-large: 3072
        - openai:text-embedding-3-small: 1536
        - openai:text-embedding-ada-002: 1536
        - cohere:embed-english-v3.0: 1024
        - cohere:embed-english-light-v3.0: 384
        - cohere:embed-multilingual-v3.0: 1024
        - cohere:embed-multilingual-light-v3.0: 384
    """

    embed: Union[Embeddings, EmbeddingsFunc, AEmbeddingsFunc]
    """Optional function to generate embeddings from text.
    
    Can be specified in three ways:
        1. A LangChain Embeddings instance
        2. A synchronous embedding function (EmbeddingsFunc)
        3. An asynchronous embedding function (AEmbeddingsFunc)
    
    ???+ example "Examples"
        Using LangChain's initialization with InMemoryStore:
        ```python
        from langchain.embeddings import init_embeddings
        from langgraph.store.memory import InMemoryStore
        
        store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": init_embeddings("openai:text-embedding-3-small")
            }
        )
        ```
        
        Using a custom embedding function with InMemoryStore:
        ```python
        from openai import OpenAI
        from langgraph.store.memory import InMemoryStore
        
        client = OpenAI()
        
        def embed_texts(texts: list[str]) -> list[list[float]]:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [e.embedding for e in response.data]
            
        store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": embed_texts
            }
        )
        ```
        
        Using an asynchronous embedding function with InMemoryStore:
        ```python
        from openai import AsyncOpenAI
        from langgraph.store.memory import InMemoryStore
        
        client = AsyncOpenAI()
        
        async def aembed_texts(texts: list[str]) -> list[list[float]]:
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [e.embedding for e in response.data]
            
        store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": aembed_texts
            }
        )
        ```
    """

    fields: Optional[list[str]]
    """Fields to extract text from for embedding generation.
    
    Controls which parts of stored items are embedded for semantic search. Follows JSON path syntax:

        - ["$"]: Embeds the entire JSON object as one vector  (default)
        - ["field1", "field2"]: Embeds specific top-level fields
        - ["parent.child"]: Embeds nested fields using dot notation
        - ["array[*].field"]: Embeds field from each array element separately
    
    Note:
        You can always override this behavior when storing an item using the
        `index` parameter in the `put` or `aput` operations.
    
    ???+ example "Examples"
        ```python
        # Embed entire document (default)
        fields=["$"]
        
        # Embed specific fields
        fields=["text", "summary"]
        
        # Embed nested fields
        fields=["metadata.title", "content.body"]
        
        # Embed from arrays
        fields=["messages[*].content"]  # Each message content separately
        fields=["context[0].text"]      # First context item's text
        ```
    
    Note:
        - Fields missing from a document are skipped
        - Array notation creates separate embeddings for each element
        - Complex nested paths are supported (e.g., "a.b[*].c.d")
    """


class BaseStore(ABC):
    """Abstract base class for persistent key-value stores.

    Stores enable persistence and memory that can be shared across threads,
    scoped to user IDs, assistant IDs, or other arbitrary namespaces.
    Some implementations may support semantic search capabilities through
    an optional `index` configuration.

    Note:
        Semantic search capabilities vary by implementation and are typically
        disabled by default. Stores that support this feature can be configured
        by providing an `index` configuration at creation time. Without this
        configuration, semantic search is disabled and any `index` arguments
        to storage operations will have no effect.
    """

    __slots__ = ("__weakref__",)

    @abstractmethod
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations synchronously in a single batch.

        Args:
            ops: An iterable of operations to execute.

        Returns:
            A list of results, where each result corresponds to an operation in the input.
            The order of results matches the order of input operations.
        """

    @abstractmethod
    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations asynchronously in a single batch.

        Args:
            ops: An iterable of operations to execute.

        Returns:
            A list of results, where each result corresponds to an operation in the input.
            The order of results matches the order of input operations.
        """

    def get(self, namespace: tuple[str, ...], key: str) -> Optional[Item]:
        """Retrieve a single item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.

        Returns:
            The retrieved item or None if not found.
        """
        return self.batch([GetOp(namespace, key)])[0]

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        """Search for items within a namespace prefix.

        Args:
            namespace_prefix: Hierarchical path prefix to search within.
            query: Optional query for natural language search.
            filter: Key-value pairs to filter results.
            limit: Maximum number of items to return.
            offset: Number of items to skip before returning results.

        Returns:
            List of items matching the search criteria.

        ???+ example "Examples"
            Basic filtering:
            ```python
            # Search for documents with specific metadata
            results = store.search(
                ("docs",),
                filter={"type": "article", "status": "published"}
            )
            ```

            Natural language search (requires vector store implementation):
            ```python
            # Initialize store with embedding configuration
            store = YourStore( # e.g., InMemoryStore, AsyncPostgresStore
                index={
                    "dims": 1536,  # embedding dimensions
                    "embed": your_embedding_function,  # function to create embeddings
                    "fields": ["text"]  # fields to embed. Defaults to ["$"]
                }
            )

            # Search for semantically similar documents
            results = store.search(
                ("docs",),
                query="machine learning applications in healthcare",
                filter={"type": "research_paper"},
                limit=5
            )
            ```

            Note: Natural language search support depends on your store implementation
            and requires proper embedding configuration.
        """
        return self.batch([SearchOp(namespace_prefix, filter, limit, offset, query)])[0]

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
    ) -> None:
        """Store or update an item in the store.

        Args:
            namespace: Hierarchical path for the item, represented as a tuple of strings.
                Example: ("documents", "user123")
            key: Unique identifier within the namespace. Together with namespace forms
                the complete path to the item.
            value: Dictionary containing the item's data. Must contain string keys
                and JSON-serializable values.
            index: Controls how the item's fields are indexed for search:

                - None (default): Use `fields` you configured when creating the store (if any)
                    If you do not initialize the store with indexing capabilities,
                    the `index` parameter will be ignored
                - False: Disable indexing for this item
                - list[str]: List of field paths to index, supporting:
                    - Nested fields: "metadata.title"
                    - Array access: "chapters[*].content" (each indexed separately)
                    - Specific indices: "authors[0].name"

        Note:
            Indexing support depends on your store implementation.
            If you do not initialize the store with indexing capabilities,
            the `index` parameter will be ignored.

        ???+ example "Examples"
            Store item. Indexing depends on how you configure the store.
            ```python
            store.put(("docs",), "report", {"memory": "Will likes ai"})
            ```

            Do not index item for semantic search. Still accessible through get()
            and search() operations but won't have a vector representation.
            ```python
            store.put(("docs",), "report", {"memory": "Will likes ai"}, index=False)
            ```

            Index specific fields for search.
            ```python
            store.put(("docs",), "report", {"memory": "Will likes ai"}, index=["memory"])
            ```
        """
        _validate_namespace(namespace)
        self.batch([PutOp(namespace, key, value, index=index)])

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.
        """
        self.batch([PutOp(namespace, key, None)])

    def list_namespaces(
        self,
        *,
        prefix: Optional[NamespacePath] = None,
        suffix: Optional[NamespacePath] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List and filter namespaces in the store.

        Used to explore the organization of data,
        find specific collections, or navigate the namespace hierarchy.

        Args:
            prefix (Optional[Tuple[str, ...]]): Filter namespaces that start with this path.
            suffix (Optional[Tuple[str, ...]]): Filter namespaces that end with this path.
            max_depth (Optional[int]): Return namespaces up to this depth in the hierarchy.
                Namespaces deeper than this level will be truncated.
            limit (int): Maximum number of namespaces to return (default 100).
            offset (int): Number of namespaces to skip for pagination (default 0).

        Returns:
            List[Tuple[str, ...]]: A list of namespace tuples that match the criteria.
            Each tuple represents a full namespace path up to `max_depth`.

        ???+ example "Examples":
            Setting max_depth=3. Given the namespaces:
            ```python
            # Example if you have the following namespaces:
            # ("a", "b", "c")
            # ("a", "b", "d", "e")
            # ("a", "b", "d", "i")
            # ("a", "b", "f")
            # ("a", "c", "f")
            store.list_namespaces(prefix=("a", "b"), max_depth=3)
            # [("a", "b", "c"), ("a", "b", "d"), ("a", "b", "f")]
            ```
        """
        match_conditions = []
        if prefix:
            match_conditions.append(MatchCondition(match_type="prefix", path=prefix))
        if suffix:
            match_conditions.append(MatchCondition(match_type="suffix", path=suffix))

        op = ListNamespacesOp(
            match_conditions=tuple(match_conditions),
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )
        return self.batch([op])[0]

    async def aget(self, namespace: tuple[str, ...], key: str) -> Optional[Item]:
        """Asynchronously retrieve a single item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.

        Returns:
            The retrieved item or None if not found.
        """
        return (await self.abatch([GetOp(namespace, key)]))[0]

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        """Asynchronously search for items within a namespace prefix.

        Args:
            namespace_prefix: Hierarchical path prefix to search within.
            query: Optional query for natural language search.
            filter: Key-value pairs to filter results.
            limit: Maximum number of items to return.
            offset: Number of items to skip before returning results.

        Returns:
            List of items matching the search criteria.

        ???+ example "Examples"
            Basic filtering:
            ```python
            # Search for documents with specific metadata
            results = await store.asearch(
                ("docs",),
                filter={"type": "article", "status": "published"}
            )
            ```

            Natural language search (requires vector store implementation):
            ```python
            # Initialize store with embedding configuration
            store = YourStore( # e.g., InMemoryStore, AsyncPostgresStore
                index={
                    "dims": 1536,  # embedding dimensions
                    "embed": your_embedding_function,  # function to create embeddings
                    "fields": ["text"]  # fields to embed
                }
            )

            # Search for semantically similar documents
            results = await store.asearch(
                ("docs",),
                query="machine learning applications in healthcare",
                filter={"type": "research_paper"},
                limit=5
            )
            ```

            Note: Natural language search support depends on your store implementation
            and requires proper embedding configuration.
        """
        return (
            await self.abatch(
                [SearchOp(namespace_prefix, filter, limit, offset, query)]
            )
        )[0]

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
    ) -> None:
        """Asynchronously store or update an item in the store.

        Args:
            namespace: Hierarchical path for the item, represented as a tuple of strings.
                Example: ("documents", "user123")
            key: Unique identifier within the namespace. Together with namespace forms
                the complete path to the item.
            value: Dictionary containing the item's data. Must contain string keys
                and JSON-serializable values.
            index: Controls how the item's fields are indexed for search:

                - None (default): Use `fields` you configured when creating the store (if any)
                    If you do not initialize the store with indexing capabilities,
                    the `index` parameter will be ignored
                - False: Disable indexing for this item
                - list[str]: List of field paths to index, supporting:
                    - Nested fields: "metadata.title"
                    - Array access: "chapters[*].content" (each indexed separately)
                    - Specific indices: "authors[0].name"

        Note:
            Indexing support depends on your store implementation.
            If you do not initialize the store with indexing capabilities,
            the `index` parameter will be ignored.

        ???+ example "Examples"
            Store item. Indexing depends on how you configure the store.
            ```python
            await store.aput(("docs",), "report", {"memory": "Will likes ai"})
            ```

            Do not index item for semantic search. Still accessible through get()
            and search() operations but won't have a vector representation.
            ```python
            await store.aput(("docs",), "report", {"memory": "Will likes ai"}, index=False)
            ```

            Index specific fields for search (if store configured to index items):
            ```python
            await store.aput(
                ("docs",),
                "report",
                {
                    "memory": "Will likes ai",
                    "context": [{"content": "..."}, {"content": "..."}]
                },
                index=["memory", "context[*].content"]
            )
            ```
        """
        _validate_namespace(namespace)
        await self.abatch([PutOp(namespace, key, value, index=index)])

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Asynchronously delete an item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.
        """
        await self.abatch([PutOp(namespace, key, None)])

    async def alist_namespaces(
        self,
        *,
        prefix: Optional[NamespacePath] = None,
        suffix: Optional[NamespacePath] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List and filter namespaces in the store asynchronously.

        Used to explore the organization of data,
        find specific collections, or navigate the namespace hierarchy.

        Args:
            prefix (Optional[Tuple[str, ...]]): Filter namespaces that start with this path.
            suffix (Optional[Tuple[str, ...]]): Filter namespaces that end with this path.
            max_depth (Optional[int]): Return namespaces up to this depth in the hierarchy.
                Namespaces deeper than this level will be truncated to this depth.
            limit (int): Maximum number of namespaces to return (default 100).
            offset (int): Number of namespaces to skip for pagination (default 0).

        Returns:
            List[Tuple[str, ...]]: A list of namespace tuples that match the criteria.
            Each tuple represents a full namespace path up to `max_depth`.

        ???+ example "Examples"
            Setting max_depth=3 with existing namespaces:
            ```python
            # Given the following namespaces:
            # ("a", "b", "c")
            # ("a", "b", "d", "e")
            # ("a", "b", "d", "i")
            # ("a", "b", "f")
            # ("a", "c", "f")

            await store.alist_namespaces(prefix=("a", "b"), max_depth=3)
            # Returns: [("a", "b", "c"), ("a", "b", "d"), ("a", "b", "f")]
            ```
        """
        match_conditions = []
        if prefix:
            match_conditions.append(MatchCondition(match_type="prefix", path=prefix))
        if suffix:
            match_conditions.append(MatchCondition(match_type="suffix", path=suffix))

        op = ListNamespacesOp(
            match_conditions=tuple(match_conditions),
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )
        return (await self.abatch([op]))[0]


def _validate_namespace(namespace: tuple[str, ...]) -> None:
    if not namespace:
        raise InvalidNamespaceError("Namespace cannot be empty.")
    for label in namespace:
        if not isinstance(label, str):
            raise InvalidNamespaceError(
                f"Invalid namespace label '{label}' found in {namespace}. Namespace labels"
                f" must be strings, but got {type(label).__name__}."
            )
        if "." in label:
            raise InvalidNamespaceError(
                f"Invalid namespace label '{label}' found in {namespace}. Namespace labels cannot contain periods ('.')."
            )
        elif not label:
            raise InvalidNamespaceError(
                f"Namespace labels cannot be empty strings. Got {label} in {namespace}"
            )
    if namespace[0] == "langgraph":
        raise InvalidNamespaceError(
            f'Root label for namespace cannot be "langgraph". Got: {namespace}'
        )


__all__ = [
    "BaseStore",
    "Item",
    "Op",
    "PutOp",
    "GetOp",
    "SearchOp",
    "ListNamespacesOp",
    "MatchCondition",
    "NamespacePath",
    "NamespaceMatchType",
    "Embeddings",
    "ensure_embeddings",
    "tokenize_path",
    "get_text_at_path",
]
