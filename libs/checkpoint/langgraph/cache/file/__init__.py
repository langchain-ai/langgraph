import asyncio
import datetime
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
        now = datetime.datetime.now(datetime.timezone.utc).timestamp()
        values: dict[str, bytes] = {}
        for key in keys:
            if val := self._db.get(key):
                expiry, *data = ormsgpack.unpackb(val)
                if expiry is not None and now > expiry:
                    self._db.pop(key, None)
                    continue
                values[key] = self.serde.loads_typed(data)
        return values

    async def aget(self, keys: list[str]) -> dict[str, bytes]:
        """Asynchronously get the cached values for the given keys."""
        return await asyncio.to_thread(self.get, keys)

    def set(self, mapping: dict[str, tuple[bytes, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""
        now = datetime.datetime.now(datetime.timezone.utc)
        for key, (value, ttl) in mapping.items():
            if ttl is not None:
                delta = datetime.timedelta(seconds=ttl)
                expiry: float | None = (now + delta).timestamp()
            else:
                expiry = None
            self._db[key] = ormsgpack.packb((expiry, *self.serde.dumps_typed(value)))

    async def aset(self, mapping: dict[str, tuple[bytes, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""
        await asyncio.to_thread(self.set, mapping)

    def delete(self, keys: list[str]) -> None:
        """Delete the cached values for the given keys."""
        for key in keys:
            self._db.pop(key, None)

    async def adelete(self, keys: list[str]) -> None:
        """Asynchronously delete the cached values for the given keys."""
        await asyncio.to_thread(self.delete, keys)
