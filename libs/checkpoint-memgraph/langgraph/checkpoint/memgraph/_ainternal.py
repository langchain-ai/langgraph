"""Async helper utilities for Memgraph checkpoint & store back‑ends."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Union

from neo4j import AsyncDriver, AsyncSession

Conn = Union[AsyncDriver, AsyncSession]


@asynccontextmanager
async def get_connection(conn: Conn) -> AsyncIterator[AsyncSession]:
    """Yield an `AsyncSession` from either an AsyncDriver or an existing AsyncSession."""
    if isinstance(conn, AsyncSession):
        yield conn
    elif isinstance(conn, AsyncDriver):
        async with conn.session() as sess:
            yield sess
    else:
        raise TypeError(f"Invalid Memgraph async connection type: {type(conn)}")
