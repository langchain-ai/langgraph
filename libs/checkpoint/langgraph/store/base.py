"""Base classes and types for persistent key-value stores.

Stores enable persistence and memory that can be shared across threads,
scoped to user IDs, assistant IDs, or other arbitrary namespaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, NamedTuple, Optional, Union


@dataclass
class Item:
    """Represents a stored item with metadata."""

    value: dict[str, Any]
    """The stored data."""
    scores: dict[str, float]
    """Relevance scores for the item."""
    id: str
    """Unique identifier within the namespace."""
    namespace: tuple[str, ...]
    """Hierarchical path for organizing items."""
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
    """Operation to store or update an item."""

    namespace: tuple[str, ...]
    """Hierarchical path for the item."""
    id: str
    """Unique identifier within the namespace."""
    value: Optional[dict[str, Any]]
    """Data to be stored, or None to delete."""


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
