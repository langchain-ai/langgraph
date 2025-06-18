from langgraph.cache.base import BaseCache, ValueT


class BasePostgresCache(BaseCache[ValueT]):
    """Base class for Postgres cache implementations."""

    MIGRATIONS = [
        """CREATE TABLE IF NOT EXISTS cache_migrations (
    v INTEGER PRIMARY KEY
);""",
        """CREATE TABLE IF NOT EXISTS cache (
    ns TEXT,
    key TEXT,
    expiry TIMESTAMP,
    encoding TEXT NOT NULL,
    val BYTEA NOT NULL,
    PRIMARY KEY (ns, key)
);""",
    ]

    # cache 操作 SQL
    SELECT_SQL = """
    SELECT c.ns, c.key, c.expiry, c.encoding, c.val
      FROM cache c
      JOIN unnest(%s::text[], %s::text[]) AS t(ns, key)
        ON c.ns = t.ns AND c.key = t.key
    """
    UPSERT_SQL = """
        INSERT INTO cache (ns, key, expiry, encoding, val)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (ns, key) DO UPDATE SET
            expiry = EXCLUDED.expiry,
            encoding = EXCLUDED.encoding,
            val = EXCLUDED.val
    """
    CLEAR_ALL_SQL = "DELETE FROM cache"
    CLEAR_NS_SQL = "DELETE FROM cache WHERE ns = ANY(%s)"
    DELETE_EXPIRED_SQL = "DELETE FROM cache WHERE ns = %s AND key = %s"
