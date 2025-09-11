"""
PyTest configuration for checkpoint‑memgraph.

Ensures that the library root (two levels up) is on the import path so that
`import langgraph.checkpoint.memgraph` resolves when the package has not been
installed into the active environment.
"""

from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import unquote, urlparse

import pytest
from neo4j import GraphDatabase, Session

from tests.embed_test_utils import CharacterEmbeddings

DEFAULT_MEMGRAPH_URI = "bolt://memgraph:memgraph@localhost:7687"


@pytest.fixture(scope="function")
def conn() -> Iterator[Session]:
    parsed = urlparse(DEFAULT_MEMGRAPH_URI)
    uri = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 7687}"
    auth = (unquote(parsed.username or ""), unquote(parsed.password or ""))
    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session() as session:
        yield session
    driver.close()


@pytest.fixture(scope="function", autouse=True)
def clear_test_db(conn: Session) -> None:
    """Delete all nodes and relationships before each test."""
    conn.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    return CharacterEmbeddings(dims=500)


VECTOR_TYPES = ["vector", "halfvec"]
