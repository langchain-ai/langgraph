import asyncio
import dbm

import ormsgpack

from langgraph.cache.base import BaseCache
from langgraph.checkpoint.serde.base import SerializerProtocol


class FileCache(BaseCache):
    """File-based cache using dbm."""

    def __init__(
        self,
        *,
        path: str,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the cache with a file path."""
        super().__init__(serde=serde)
        self._db = dbm.open(path, "c")

    def get(self, keys: list[str]) -> dict[str, bytes]:
        """Get the cached values for the given keys."""
        return {
            key: self.serde.loads_typed(ormsgpack.unpackb(self._db[key]))
            for key in keys
            if key in self._db
        }

    def set(self, mapping: dict[str, tuple[bytes, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""
        for key, (value, _) in mapping.items():
            # File-based caches do not support TTLs, so we ignore them.
            self._db[key] = ormsgpack.packb(self.serde.dumps_typed(value))

    def delete(self, keys: list[str]) -> None:
        """Delete the cached values for the given keys."""
        for key in keys:
            self._db.pop(key, None)

    async def aget(self, keys: list[str]) -> dict[str, bytes]:
        """Asynchronously get the cached values for the given keys."""
        return await asyncio.to_thread(self.get, keys)

    async def aset(self, mapping: dict[str, tuple[bytes, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""
        await asyncio.to_thread(self.set, mapping)

    async def adelete(self, keys: list[str]) -> None:
        """Asynchronously delete the cached values for the given keys."""
        await asyncio.to_thread(self.delete, keys)
