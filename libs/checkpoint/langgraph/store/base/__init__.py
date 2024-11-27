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

from langgraph.store.base._embed import (
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
            "value": self.value,
            "key": self.key,
            "namespace": list(self.namespace),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ResponseMetadata(TypedDict, total=False):
    """Additional metadata about the response/result."""

    score: float
    """Relevance/similarity score if from a ranked operation."""


class SearchItem(Item):
    """Represents a result item with additional response metadata."""

    __slots__ = "response_metadata"

    def __init__(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        created_at: datetime,
        updated_at: datetime,
        response_metadata: Optional[ResponseMetadata] = None,
    ) -> None:
        """Initialize a result item.

        Args:
            namespace: Hierarchical path to the item.
            key: Unique identifier within the namespace.
            value: The stored value.
            created_at: When the item was first created.
            updated_at: When the item was last updated.
            response_metadata: Optional metadata about the response/result.
        """
        super().__init__(
            value=value,
            key=key,
            namespace=namespace,
            created_at=created_at,
            updated_at=updated_at,
        )
        self.response_metadata = response_metadata or {}

    def dict(self) -> dict:
        result = super().dict()
        result["response_metadata"] = self.response_metadata
        return result


class GetOp(NamedTuple):
    """Operation to retrieve an item by namespace and key."""

    namespace: tuple[str, ...]
    """Hierarchical path for the item."""
    key: str
    """Unique identifier within the namespace."""


class SearchOp(NamedTuple):
    """Operation to search for items within a namespace prefix."""

    namespace_prefix: tuple[str, ...]
    """Hierarchical path prefix to search within."""
    filter: Optional[dict[str, Any]] = None
    """Key-value pairs to filter results."""
    limit: int = 10
    """Maximum number of items to return."""
    offset: int = 0
    """Number of items to skip before returning results."""
    query: Optional[str] = None
    """The search query for natural language search."""


class PutOp(NamedTuple):
    """Operation to store, update, or delete an item."""

    namespace: tuple[str, ...]
    """Hierarchical path for the item.
    
    Represented as a tuple of strings, allowing for nested categorization.
    For example: ("documents", "user123")
    """

    key: str
    """Unique identifier for the document.
    
    Should be distinct within its namespace.
    """

    value: Optional[dict[str, Any]]
    """Data to be stored, or None to delete the item.
    
    Schema:
    - Should be a dictionary where:
      - Keys are strings representing field names
      - Values can be of any serializable type
    - If None, it indicates that the item should be deleted
    """
    index: Optional[bool] = None  # type: ignore[assignment]
    """Whether to index the item (if supported by the store).
    
    Defaults to True if the store supports indexing. This will embed the document
    so it can be queried using search.
    """


NameSpacePath = tuple[Union[str, Literal["*"]], ...]

NamespaceMatchType = Literal["prefix", "suffix"]


class MatchCondition(NamedTuple):
    """Represents a single match condition."""

    match_type: NamespaceMatchType
    path: NameSpacePath


class ListNamespacesOp(NamedTuple):
    """Operation to list namespaces with optional match conditions."""

    match_conditions: Optional[tuple[MatchCondition, ...]] = None
    """A tuple of match conditions to apply to namespaces."""

    max_depth: Optional[int] = None
    """Return namespaces up to this depth in the hierarchy."""

    limit: int = 100
    """Maximum number of namespaces to return."""

    offset: int = 0
    """Number of namespaces to skip before returning results."""


Op = Union[GetOp, SearchOp, PutOp, ListNamespacesOp]
Result = Union[Item, list[Item], list[SearchItem], list[tuple[str, ...]], None]


class InvalidNamespaceError(ValueError):
    """Provided namespace is invalid."""


class IndexConfig(TypedDict, total=False):
    """Configuration for indexing documents for semantic search in the store."""

    dims: int
    """Number of dimensions in the embedding vectors.
    
    Common embedding models have the following dimensions:
        - OpenAI text-embedding-3-large: 256, 1024, or 3072
        - OpenAI text-embedding-3-small: 512 or 1536
        - OpenAI text-embedding-ada-002: 1536
        - Cohere embed-english-v3.0: 1024
        - Cohere embed-english-light-v3.0: 384
        - Cohere embed-multilingual-v3.0: 1024
        - Cohere embed-multilingual-light-v3.0: 384
    """

    embed: Union[Embeddings, EmbeddingsFunc, AEmbeddingsFunc]
    """Optional function to generate embeddings from text."""

    text_fields: Optional[list[str]]
    """Fields to extract text from for embedding generation.
    
    Defaults to ["__root__"], which embeds the json object as a whole.
    """


class BaseStore(ABC):
    """Abstract base class for persistent key-value stores.

    Stores enable persistence and memory that can be shared across threads,
    scoped to user IDs, assistant IDs, or other arbitrary namespaces.
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
        """
        return self.batch([SearchOp(namespace_prefix, filter, limit, offset, query)])[0]

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[bool] = None,
    ) -> None:
        """Store or update an item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.
            value: Dictionary containing the item's data.
            index: Whether to index the item (if supported by the store).
                Defaults to True if the store supports indexing.
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
        prefix: Optional[NameSpacePath] = None,
        suffix: Optional[NameSpacePath] = None,
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
                Namespaces deeper than this level will be truncated to this depth.
            limit (int): Maximum number of namespaces to return (default 100).
            offset (int): Number of namespaces to skip for pagination (default 0).

        Returns:
            List[Tuple[str, ...]]: A list of namespace tuples that match the criteria.
            Each tuple represents a full namespace path up to `max_depth`.

        Examples:

            Setting max_depth=3. Given the namespaces:
                # ("a", "b", "c")
                # ("a", "b", "d", "e")
                # ("a", "b", "d", "i")
                # ("a", "b", "f")
                # ("a", "c", "f")
                store.list_namespaces(prefix=("a", "b"), max_depth=3)
                # [("a", "b", "c"), ("a", "b", "d"), ("a", "b", "f")]
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
        index: Optional[bool] = None,
    ) -> None:
        """Asynchronously store or update an item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.
            value: Dictionary containing the item's data.
            index: Whether to index the item (if supported by the store).
                Defaults to True if the store supports indexing.
        """
        _validate_namespace(namespace)
        await self.abatch([PutOp(namespace, key, value, index)])

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
        prefix: Optional[NameSpacePath] = None,
        suffix: Optional[NameSpacePath] = None,
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

        Examples:

            Setting max_depth=3. Given the namespaces:
                # ("a", "b", "c")
                # ("a", "b", "d", "e")
                # ("a", "b", "d", "i")
                # ("a", "b", "f")
                # ("a", "c", "f")
                await store.alist_namespaces(prefix=("a", "b"), max_depth=3)
                # [("a", "b", "c"), ("a", "b", "d"), ("a", "b", "f")]
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
    "NameSpacePath",
    "NamespaceMatchType",
    "Embeddings",
    "ensure_embeddings",
    "tokenize_path",
    "get_text_at_path",
]
