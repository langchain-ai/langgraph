"""Base classes and types for persistent key-value stores.

Stores enable persistence and memory that can be shared across threads,
scoped to user IDs, assistant IDs, or other arbitrary namespaces.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Literal, NamedTuple, Optional, Union


@dataclass
class Item:
    """Represents a stored item with metadata."""

    value: dict[str, Any]
    """The stored data as a dictionary.
    
    Keys are filterable.
    """

    scores: dict[str, float]
    """Relevance scores for the item.
    
    Keys can include built-in scores like 'recency' and 'relevance',
    as well as any key present in the 'value' dictionary. This allows
    for multi-dimensional scoring of items.
    """

    id: str
    """Unique identifier within the namespace."""

    namespace: tuple[str, ...]
    """Hierarchical path defining the collection in which this document resides.
    
    Represented as a tuple of strings, allowing for nested categorization.
    For example: ("documents", 'user123')
    """

    created_at: datetime
    """Timestamp of item creation."""

    updated_at: datetime
    """Timestamp of last update."""

    last_accessed_at: datetime
    """Timestamp of last access."""


class GetOp(NamedTuple):
    """Operation to retrieve an item by namespace and ID."""

    namespace: tuple[str, ...]
    """Hierarchical path for the item."""
    id: str
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


class PutOp(NamedTuple):
    """Operation to store, update, or delete an item."""

    namespace: tuple[str, ...]
    """Hierarchical path for the item.
    
    Represented as a tuple of strings, allowing for nested categorization.
    For example: ("documents", "user123")
    """

    id: str
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


Op = Union[GetOp, SearchOp, PutOp]
Result = Union[Item, list[Item], None]


class BaseStore(ABC):
    """Abstract base class for key-value stores."""

    __slots__ = ("__weakref__",)

    @abstractmethod
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations synchronously."""

    @abstractmethod
    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations asynchronously."""

    def batch_namespaces(
        self, ops: Iterable[ListNamespacesOp]
    ) -> list[tuple[str, ...]]:
        """Execute a batch of namespace operations synchronously."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement batch_namespaces"
        )

    async def abatch_namespace(
        self, ops: Iterable[ListNamespacesOp]
    ) -> list[tuple[str, ...]]:
        """Execute a batch of namespace operations asynchronously."""
        return asyncio.get_event_loop().run_in_executor(
            None, self.batch_namespaces, ops
        )

    def get(self, namespace: tuple[str, ...], id: str) -> Optional[Item]:
        """Retrieve a single item.

        Args:
            namespace: Hierarchical path for the item.
            id: Unique identifier within the namespace.

        Returns:
            The retrieved item or None if not found.
        """
        return self.batch([GetOp(namespace, id)])[0]

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Item]:
        """Search for items within a namespace prefix.

        Args:
            namespace_prefix: Hierarchical path prefix to search within.
            filter: Key-value pairs to filter results.
            limit: Maximum number of items to return.
            offset: Number of items to skip before returning results.

        Returns:
            List of items matching the search criteria.
        """
        return self.batch([SearchOp(namespace_prefix, filter, limit, offset)])[0]

    def put(self, namespace: tuple[str, ...], id: str, value: dict[str, Any]) -> None:
        """Store or update an item.

        Args:
            namespace: Hierarchical path for the item.
            id: Unique identifier within the namespace.
            value: Dictionary containing the item's data.
        """
        self.batch([PutOp(namespace, id, value)])

    def delete(self, namespace: tuple[str, ...], id: str) -> None:
        """Delete an item.

        Args:
            namespace: Hierarchical path for the item.
            id: Unique identifier within the namespace.
        """
        self.batch([PutOp(namespace, id, None)])

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
        return self.batch_namespaces([op])[0]

    async def aget(self, namespace: tuple[str, ...], id: str) -> Optional[Item]:
        """Asynchronously retrieve a single item.

        Args:
            namespace: Hierarchical path for the item.
            id: Unique identifier within the namespace.

        Returns:
            The retrieved item or None if not found.
        """
        return (await self.abatch([GetOp(namespace, id)]))[0]

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Item]:
        """Asynchronously search for items within a namespace prefix.

        Args:
            namespace_prefix: Hierarchical path prefix to search within.
            filter: Key-value pairs to filter results.
            limit: Maximum number of items to return.
            offset: Number of items to skip before returning results.

        Returns:
            List of items matching the search criteria.
        """
        return (await self.abatch([SearchOp(namespace_prefix, filter, limit, offset)]))[
            0
        ]

    async def aput(
        self, namespace: tuple[str, ...], id: str, value: dict[str, Any]
    ) -> None:
        """Asynchronously store or update an item.

        Args:
            namespace: Hierarchical path for the item.
            id: Unique identifier within the namespace.
            value: Dictionary containing the item's data.
        """
        await self.abatch([PutOp(namespace, id, value)])

    async def adelete(self, namespace: tuple[str, ...], id: str) -> None:
        """Asynchronously delete an item.

        Args:
            namespace: Hierarchical path for the item.
            id: Unique identifier within the namespace.
        """
        await self.abatch([PutOp(namespace, id, None)])

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
        return (await self.abatch_namespace([op]))[0]
