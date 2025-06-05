"""Shared utility functions for Memgraph-based checkpoint & storage classes (sync)."""

from contextlib import contextmanager
from typing import Iterator, Union

from neo4j import GraphDatabase, Session, Driver


class MemgraphConn:
    """
    Represents a Memgraph connection which can be either a `neo4j.Driver`
    or an established session. Typically, we'll store just the driver.
    """

    def __init__(self, uri: str, user: str = None, password: str = None):
        if user is not None and password is not None:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        else:
            self.driver = GraphDatabase.driver(uri)
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self.driver.close()
            self._closed = True


Conn = Union[Driver, MemgraphConn]


@contextmanager
def get_session(conn: Conn) -> Iterator[Session]:
    """
    If conn is a MemgraphConn, open a new session. If it's a raw Driver, do the same.
    """
    if isinstance(conn, MemgraphConn):
        with conn.driver.session() as session:
            yield session
    elif isinstance(conn, Driver):
        with conn.session() as session:
            yield session
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")

