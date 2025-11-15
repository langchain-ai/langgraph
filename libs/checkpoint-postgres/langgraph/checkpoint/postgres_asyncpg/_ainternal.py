"""Shared async utility functions for the Postgres checkpoint & storage classes."""

from __future__ import annotations
from collections.abc import AsyncIterator

from contextlib import asynccontextmanager

import asyncpg
from asyncpg.pool import Pool

Conn = asyncpg.Connection | Pool


@asynccontextmanager
async def get_connection(
    conn: Conn,
) -> AsyncIterator[asyncpg.Connection]:
    if isinstance(conn, asyncpg.Connection):
        yield conn
    elif isinstance(conn, Pool):
        async with conn.acquire() as connection:
            yield connection
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
