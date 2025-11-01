"""Shared utility functions for the Kusto checkpoint classes."""

from collections.abc import Iterator
from contextlib import contextmanager

from azure.kusto.data import KustoClient
from azure.kusto.ingest import StreamingIngestClient

# Type aliases for connection types
KustoQueryConn = KustoClient
KustoIngestConn = StreamingIngestClient


@contextmanager
def get_query_client(client: KustoQueryConn) -> Iterator[KustoQueryConn]:
    """Get a Kusto query client context manager.

    Args:
        client: The Kusto client to use for queries.

    Yields:
        The Kusto client.
    """
    try:
        yield client
    finally:
        # Kusto clients don't require explicit cleanup in normal operation
        pass


@contextmanager
def get_ingest_client(client: StreamingIngestClient) -> Iterator[StreamingIngestClient]:
    """Get a Kusto ingest client context manager.

    Args:
        client: The Kusto ingest client to use.

    Yields:
        The Kusto ingest client.
    """
    try:
        yield client
    finally:
        # Kusto clients don't require explicit cleanup in normal operation
        pass
