"""Sync helper utilities for Memgraph checkpoint and store back‑ends."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Union

from neo4j import Driver, Session

Conn = Union[Driver, Session]


@contextmanager
def get_connection(conn: Conn) -> Iterator[Session]:
    """Yield a `neo4j.Session` from either a Driver or an existing Session."""
    if isinstance(conn, Session):
        yield conn
    elif isinstance(conn, Driver):
        with conn.session() as sess:
            yield sess
    else:
        raise TypeError(f"Invalid Memgraph connection type: {type(conn)}")
