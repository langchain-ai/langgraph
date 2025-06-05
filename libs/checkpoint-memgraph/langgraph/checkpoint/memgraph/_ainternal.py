"""Shared utility functions for Memgraph-based checkpoint & storage classes (async)."""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Union

from neo4j import AsyncGraphDatabase, AsyncSession, AsyncDriver


class AsyncMemgraphConn:
    """
    Represents an async Memgraph connection which can be either a `neo4j.AsyncDriver`
    or an established session. Typically, we'll store just the driver.
    """

    def __init__(self, uri: str, user: str = None, password: str = None):
        if user is not None and password is not None:
            self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        else:
            self.driver = AsyncGraphDatabase.driver(uri)
        self._closed = False

    async def close(self) -> None:
        if not self._closed:
            await self.driver.close()
            self._closed = True


Conn = Union[AsyncDriver, AsyncMemgraphConn]


@asynccontextmanager
async def aget_session(conn: Conn) -> AsyncIterator[AsyncSession]:
    """
    If conn is an AsyncMemgraphConn, open a new session. If it's a raw AsyncDriver, do the same.
    """
    if isinstance(conn, AsyncMemgraphConn):
        async with conn.driver.session() as session:
            yield session
    elif isinstance(conn, AsyncDriver):
        async with conn.session() as session:
            yield session
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")