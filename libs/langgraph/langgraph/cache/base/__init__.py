from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Generic, TypeVar

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

ValueT = TypeVar("ValueT")
Namespace = tuple[str, ...]
FullKey = tuple[Namespace, str]


class BaseCache(ABC, Generic[ValueT]):
    """Base class for a cache."""

    serde: SerializerProtocol = JsonPlusSerializer(pickle_fallback=True)

    def __init__(self, *, serde: SerializerProtocol | None = None) -> None:
        """Initialize the cache with a serializer."""
        self.serde = serde or self.serde

    @abstractmethod
    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""

    @abstractmethod
    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""

    @abstractmethod
    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""

    @abstractmethod
    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""

    @abstractmethod
    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""

    @abstractmethod
    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
