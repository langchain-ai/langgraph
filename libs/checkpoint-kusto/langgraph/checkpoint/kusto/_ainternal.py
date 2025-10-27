"""Shared utility functions for async Kusto checkpoint classes."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from azure.kusto.data.aio import KustoClient as AsyncKustoClient
from azure.kusto.ingest import (
    ManagedStreamingIngestClient as AsyncStreamingIngestClient,
)

# Type aliases for async connection types
AsyncKustoQueryConn = AsyncKustoClient
AsyncKustoIngestConn = AsyncStreamingIngestClient


@asynccontextmanager
async def get_query_client(
    client: AsyncKustoQueryConn,
) -> AsyncIterator[AsyncKustoQueryConn]:
    """Get an async Kusto query client context manager.
    
    Args:
        client: The async Kusto client to use for queries.
        
    Yields:
        The async Kusto client.
    """
    try:
        yield client
    finally:
        # Cleanup if needed
        pass


@asynccontextmanager
async def get_ingest_client(
    client: AsyncStreamingIngestClient,
) -> AsyncIterator[AsyncStreamingIngestClient]:
    """Get an async Kusto ingest client context manager.
    
    Args:
        client: The async Kusto ingest client to use.
        
    Yields:
        The async Kusto ingest client.
    """
    try:
        yield client
    finally:
        # Cleanup if needed
        pass
