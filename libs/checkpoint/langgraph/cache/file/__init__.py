from __future__ import annotations

import asyncio
import datetime
import sqlite3
import threading
from collections.abc import Mapping, Sequence

from langgraph.cache.base import BaseCache
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
                key TEXT PRIMARY KEY,
                expiry REAL,
                encoding TEXT NOT NULL,
                val BLOB NOT NULL
            )"""
        )
        self._conn.commit()

    def get(self, keys: Sequence[str]) -> dict[str, bytes]:
        """Get the cached values for the given keys."""
        with self._lock, self._conn:
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            if not keys:
                return {}
            placeholders = ",".join("?" for _ in keys)
            cursor = self._conn.execute(
                f"SELECT key, expiry, encoding, val FROM cache WHERE key IN ({placeholders})",
                tuple(keys),
            )
            values: dict[str, bytes] = {}
            rows = cursor.fetchall()
            for key, expiry, encoding, raw in rows:
                if expiry is not None and now > expiry:
                    # purge expired entry
                    self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    continue
                values[key] = self.serde.loads_typed((encoding, raw))
            return values

    async def aget(self, keys: Sequence[str]) -> dict[str, bytes]:
        """Asynchronously get the cached values for the given keys."""
        return await asyncio.to_thread(self.get, keys)

    def set(self, mapping: Mapping[str, tuple[bytes, int | None]]) -> None:
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
                    "INSERT OR REPLACE INTO cache (key, expiry, encoding, val) VALUES (?, ?, ?, ?)",
                    (key, expiry, encoding, raw),
                )

    async def aset(self, mapping: Mapping[str, tuple[bytes, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""
        await asyncio.to_thread(self.set, mapping)

    def delete(self, keys: Sequence[str]) -> None:
        """Delete the cached values for the given keys."""
        if not keys:
            return
        with self._lock, self._conn:
            placeholders = ",".join("?" for _ in keys)
            self._conn.execute(
                f"DELETE FROM cache WHERE key IN ({placeholders})", tuple(keys)
            )

    async def adelete(self, keys: Sequence[str]) -> None:
        """Asynchronously delete the cached values for the given keys."""
        await asyncio.to_thread(self.delete, keys)

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
