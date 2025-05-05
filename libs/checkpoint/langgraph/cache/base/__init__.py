from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Generic, Sequence, TypeVar

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

ValueT = TypeVar("ValueT")


class BaseCache(ABC, Generic[ValueT]):
    """Base class for a cache."""

    serde: SerializerProtocol = JsonPlusSerializer(pickle_fallback=True)

    def __init__(self, *, serde: SerializerProtocol | None = None) -> None:
        """Initialize the cache with a serializer."""
        self.serde = serde or self.serde

    @abstractmethod
    def get(self, keys: Sequence[str]) -> dict[str, ValueT]:
        """Get the cached values for the given keys."""

    @abstractmethod
    async def aget(self, keys: Sequence[str]) -> dict[str, ValueT]:
        """Asynchronously get the cached values for the given keys."""

    @abstractmethod
    def set(self, mapping: Mapping[str, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""

    @abstractmethod
    async def aset(self, mapping: Mapping[str, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""

    @abstractmethod
    def delete(self, keys: Sequence[str]) -> None:
        """Delete the cached values for the given keys."""

    @abstractmethod
    async def adelete(self, keys: Sequence[str]) -> None:
        """Asynchronously delete the cached values for the given keys."""
