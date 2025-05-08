from __future__ import annotations

import asyncio
import datetime
import sqlite3
import threading
from collections.abc import Mapping, Sequence

from langgraph.cache.base import BaseCache, FullKey, Namespace
from langgraph.checkpoint.serde.base import SerializerProtocol


class FileCache(BaseCache):
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

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, bytes]:
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
            values: dict[FullKey, bytes] = {}
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

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, bytes]:
        """Asynchronously get the cached values for the given keys."""
        return await asyncio.to_thread(self.get, keys)

    def set(self, mapping: Mapping[FullKey, tuple[bytes, int | None]]) -> None:
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

    async def aset(self, mapping: Mapping[FullKey, tuple[bytes, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""
        await asyncio.to_thread(self.set, mapping)

    def delete(self, keys: Sequence[Namespace]) -> None:
        """Delete the cached values for the given namespaces."""
        if not keys:
            return
        with self._lock, self._conn:
            placeholders = ",".join("?" for _ in keys)
            self._conn.execute(
                f"DELETE FROM cache WHERE (ns) IN ({placeholders})",
                tuple(",".join(key) for key in keys),
            )

    async def adelete(self, keys: Sequence[Namespace]) -> None:
        """Asynchronously delete the cached values for the given namespaces."""
        await asyncio.to_thread(self.delete, keys)

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
