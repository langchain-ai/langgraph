from __future__ import annotations

import datetime
import threading
from collections.abc import Sequence
from typing import Generic

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol


class InMemoryCache(BaseCache[ValueT], Generic[ValueT]):
    def __init__(self, *, serde: SerializerProtocol | None = None):
        super().__init__(serde=serde)
        self._cache: dict[Namespace, dict[str, tuple[str, bytes, int | None]]] = {}
        self._lock = threading.RLock()

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""
        with self._lock:
            if not keys:
                return {}
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            values: dict[FullKey, ValueT] = {}
            for ns_tuple, key in keys:
                ns = Namespace(ns_tuple)
                if ns in self._cache and key in self._cache[ns]:
                    enc, val, expiry = self._cache[ns][key]
                    if expiry is None or now < expiry:
                        values[(ns, key)] = self.serde.loads_typed((enc, val))
                    else:
                        del self._cache[ns][key]
            return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""
        return self.get(keys)

    def set(self, keys: dict[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys."""
        with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            for (ns, key), (value, ttl) in keys.items():
                if ttl is not None:
                    delta = datetime.timedelta(seconds=ttl)
                    expiry: float | None = (now + delta).timestamp()
                else:
                    expiry = None
                if ns not in self._cache:
                    self._cache[ns] = {}
                self._cache[ns][key] = (
                    *self.serde.dumps_typed(value),
                    expiry,
                )

    async def aset(self, keys: dict[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys."""
        self.set(keys)

    def delete(self, keys: Sequence[Namespace]) -> None:
        """Delete the cached values for the given namespaces."""
        with self._lock:
            for ns in keys:
                if ns in self._cache:
                    del self._cache[ns]

    async def adelete(self, keys: Sequence[Namespace]) -> None:
        """Asynchronously delete the cached values for the given namespaces."""
        self.delete(keys)
