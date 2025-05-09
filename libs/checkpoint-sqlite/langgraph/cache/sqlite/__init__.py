from __future__ import annotations

import asyncio
import datetime
import sqlite3
import threading
from collections.abc import Mapping, Sequence

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol


class SqliteCache(BaseCache[ValueT]):
    """File-based cache using SQLite."""

    def __init__(
        self,
        *,
        path: str,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the cache with a file path."""
        super().__init__(serde=serde)
        # SQLite backing store
        self._conn = sqlite3.connect(
            path,
            check_same_thread=False,
        )
        # Serialize access to the shared connection across threads
        self._lock = threading.RLock()
        # Better concurrency & atomicity
        self._conn.execute("PRAGMA journal_mode=WAL;")
        # Schema: key -> (expiry, encoding, value)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                ns TEXT,
                key TEXT,
                expiry REAL,
                encoding TEXT NOT NULL,
                val BLOB NOT NULL,
                PRIMARY KEY (ns, key)
            )"""
        )
        self._conn.commit()

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""
        with self._lock, self._conn:
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            if not keys:
                return {}
            placeholders = ",".join("(?, ?)" for _ in keys)
            params: list[str] = []
            for ns_tuple, key in keys:
                params.extend((",".join(ns_tuple), key))
            cursor = self._conn.execute(
                f"SELECT ns, key, expiry, encoding, val FROM cache WHERE (ns, key) IN ({placeholders})",
                tuple(params),
            )
            values: dict[FullKey, ValueT] = {}
            rows = cursor.fetchall()
            for ns, key, expiry, encoding, raw in rows:
                if expiry is not None and now > expiry:
                    # purge expired entry
                    self._conn.execute(
                        "DELETE FROM cache WHERE (ns, key) = (?, ?)", (ns, key)
                    )
                    continue
                values[(tuple(ns.split(",")), key)] = self.serde.loads_typed(
                    (encoding, raw)
                )
            return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""
        return await asyncio.to_thread(self.get, keys)

    def set(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""
        with self._lock, self._conn:
            now = datetime.datetime.now(datetime.timezone.utc)
            for key, (value, ttl) in mapping.items():
                if ttl is not None:
                    delta = datetime.timedelta(seconds=ttl)
                    expiry: float | None = (now + delta).timestamp()
                else:
                    expiry = None
                encoding, raw = self.serde.dumps_typed(value)
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache (ns, key, expiry, encoding, val) VALUES (?, ?, ?, ?, ?)",
                    (",".join(key[0]), key[1], expiry, encoding, raw),
                )

    async def aset(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""
        await asyncio.to_thread(self.set, mapping)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        with self._lock, self._conn:
            if namespaces is None:
                self._conn.execute("DELETE FROM cache")
            else:
                placeholders = ",".join("?" for _ in namespaces)
                self._conn.execute(
                    f"DELETE FROM cache WHERE (ns) IN ({placeholders})",
                    tuple(",".join(key) for key in namespaces),
                )

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        await asyncio.to_thread(self.clear, namespaces)

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
