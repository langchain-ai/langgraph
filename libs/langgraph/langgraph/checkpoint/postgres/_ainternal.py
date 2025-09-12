"""Shared async utility functions for the Postgres checkpoint & storage classes."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Union

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

Conn = Union[AsyncConnection[DictRow], AsyncConnectionPool[AsyncConnection[DictRow]]]


@asynccontextmanager
async def get_connection(
    conn: Conn,
) -> AsyncIterator[AsyncConnection[DictRow]]:
    if isinstance(conn, AsyncConnection):
        yield conn
    elif isinstance(conn, AsyncConnectionPool):
        async with conn.connection() as conn:
            yield conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
