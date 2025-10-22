"""Async implementation of Kusto checkpointer."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import orjson
from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.kusto.data import KustoConnectionStringBuilder
from azure.kusto.data.aio import KustoClient as AsyncKustoClient
from azure.kusto.ingest import IngestionProperties, DataFormat
from azure.kusto.ingest.aio import (
    StreamingIngestClient as AsyncStreamingIngestClient,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from langgraph.checkpoint.kusto import _ainternal
from langgraph.checkpoint.kusto.base import BaseKustoSaver

logger = logging.getLogger(__name__)


class AsyncKustoSaver(BaseKustoSaver):
    """Asynchronous checkpointer that stores checkpoints in Azure Data Explorer (Kusto).
    
    This implementation uses Kusto's async clients for both querying and streaming ingestion.
    Streaming ingestion provides low latency (<1 second) data availability.
    
    Example:
        ```python
        async with AsyncKustoSaver.from_connection_string(
            cluster_uri="https://cluster.region.kusto.windows.net",
            database="mydb",
        ) as checkpointer:
            await checkpointer.setup()
            config = {"configurable": {"thread_id": "thread-1"}}
            checkpoint = await checkpointer.aget_tuple(config)
        ```
    
    Attributes:
        query_client: Async Kusto client for querying.
        ingest_client: Async Kusto client for streaming ingestion.
        database: Name of the Kusto database.
        batch_size: Number of records to batch before flushing.
        lock: Async lock for thread-safe operations.
    """

    lock: asyncio.Lock
    query_client: AsyncKustoClient
    ingest_client: AsyncStreamingIngestClient
    database: str
    batch_size: int
    flush_interval: float
    _write_buffer: list[dict[str, Any]]
    _checkpoint_buffer: list[dict[str, Any]]

    def __init__(
        self,
        query_client: AsyncKustoClient,
        ingest_client: AsyncStreamingIngestClient,
        database: str,
        *,
        batch_size: int = 100,
        flush_interval: float = 30.0,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the async Kusto checkpointer.
        
        Args:
            query_client: Async Kusto client for queries.
            ingest_client: Async Kusto client for streaming ingestion.
            database: Name of the Kusto database.
            batch_size: Number of records to batch before auto-flush.
            flush_interval: Seconds between automatic flushes.
            serde: Custom serializer (optional).
        """
        super().__init__(serde=serde)
        self.query_client = query_client
        self.ingest_client = ingest_client
        self.database = database
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        
        # Initialize buffers for batching
        # Note: Blobs now stored in Checkpoints.channel_values (dynamic column)
        self._write_buffer = []
        self._checkpoint_buffer = []
        
        logger.info(
            "Initialized AsyncKustoSaver with streaming ingestion",
            extra={
                "database": database,
                "batch_size": batch_size,
            },
        )

    @classmethod
    @asynccontextmanager
    async def from_connection_string(
        cls,
        cluster_uri: str,
        database: str,
        *,
        credential: Any | None = None,
        batch_size: int = 100,
        flush_interval: float = 30.0,
        serde: SerializerProtocol | None = None,
    ) -> AsyncIterator[AsyncKustoSaver]:
        """Create an AsyncKustoSaver from connection parameters.
        
        Args:
            cluster_uri: Kusto cluster URI (e.g., "https://cluster.region.kusto.windows.net").
            database: Database name.
            credential: Azure credential (defaults to DefaultAzureCredential).
            batch_size: Number of records to batch before flush.
            flush_interval: Seconds between automatic flushes.
            serde: Custom serializer.
            
        Yields:
            Configured AsyncKustoSaver instance.
            
        Example:
            ```python
            async with AsyncKustoSaver.from_connection_string(
                cluster_uri="https://mycluster.eastus.kusto.windows.net",
                database="langgraph",
            ) as saver:
                await saver.setup()
                # Use saver...
            ```
        """
        if credential is None:
            credential = AsyncDefaultAzureCredential()
        
        # Build connection string for query client
        kcsb = KustoConnectionStringBuilder.with_azure_token_credential(
            cluster_uri, credential
        )
        
        # Create query client
        query_client = AsyncKustoClient(kcsb)
        
        # Create streaming ingest client
        ingest_client = AsyncStreamingIngestClient(kcsb)
        
        try:
            saver = cls(
                query_client=query_client,
                ingest_client=ingest_client,
                database=database,
                batch_size=batch_size,
                flush_interval=flush_interval,
                serde=serde,
            )
            yield saver
        finally:
            # Flush any pending writes
            await saver.flush()
            # Close clients
            await query_client.close()
            if hasattr(ingest_client, 'close'):
                await ingest_client.close()

    async def setup(self) -> None:
        """Set up and validate the Kusto database schema.
        
        This method checks that the required tables exist in the database.
        It does NOT create tables - run provision.kql manually first.
        
        Raises:
            ValueError: If required tables are missing.
        """
        logger.info("Validating Kusto schema", extra={"database": self.database})
        
        # Query to check if tables exist
        query = f"""
        .show database {self.database} schema
        | where TableName in ('Checkpoints', 'CheckpointWrites', 'CheckpointBlobs')
        | summarize tables = make_set(TableName)
        """
        
        async with self._query() as client:
            response = await client.execute(self.database, query)
            
            # Parse response to check for all required tables
            required_tables = {"Checkpoints", "CheckpointWrites", "CheckpointBlobs"}
            if response.primary_results:
                for row in response.primary_results[0]:
                    found_tables = set(row["tables"])
                    if not required_tables.issubset(found_tables):
                        missing = required_tables - found_tables
                        raise ValueError(
                            f"Missing required tables: {missing}. "
                            "Please run provision.kql to create the schema."
                        )
            else:
                raise ValueError(
                    "Could not validate schema. Please ensure provision.kql has been run."
                )
        
        logger.info("Schema validation successful")

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from Kusto asynchronously.
        
        If the config contains a checkpoint_id, retrieves that specific checkpoint.
        Otherwise, retrieves the latest checkpoint for the thread.
        
        Args:
            config: Configuration containing thread_id and optional checkpoint_id.
            
        Returns:
            CheckpointTuple if found, None otherwise.
            
        Example:
            ```python
            config = {"configurable": {"thread_id": "thread-1"}}
            checkpoint = await saver.aget_tuple(config)
            if checkpoint:
                print(f"Found checkpoint: {checkpoint.checkpoint['id']}")
            ```
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        
        logger.debug(
            "Fetching checkpoint",
            extra={
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": checkpoint_ns,
            },
        )
        
        # Build query with parameters
        filter_info = self._build_kql_filter(config, None, None)
        params = filter_info["params"]
        
        # Construct query
        # Use materialized view for "latest checkpoint" queries (more efficient than ORDER BY + TAKE 1)
        # Use base table for specific checkpoint_id queries
        if checkpoint_id:
            # Querying specific checkpoint - use base table with filter
            checkpoint_filter = f"| where checkpoint_id == '{checkpoint_id}'"
            limit = ""
            query = self.SELECT_CHECKPOINT_KQL.format(
                checkpoint_id_filter=checkpoint_filter,
                limit_clause=limit,
            )
        else:
            # Querying latest checkpoint - use materialized view with arg_max()
            # This is significantly more efficient as the view pre-computes the latest checkpoint
            checkpoint_filter = ""
            query = self.SELECT_LATEST_CHECKPOINT_KQL.format(
                checkpoint_id_filter=checkpoint_filter,
            )
        
        # Replace parameter placeholders
        for param_name, param_value in params.items():
            query = query.replace(param_name, f"'{param_value}'")
        
        async with self._query() as client:
            response = await client.execute(self.database, query)
            
            if not response.primary_results or not response.primary_results[0]:
                logger.debug("No checkpoint found", extra={"thread_id": thread_id})
                return None
            
            # Get first row
            row = response.primary_results[0][0]
            return await self._load_checkpoint_tuple(row)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from Kusto asynchronously.
        
        Checkpoints are ordered by checkpoint_id descending (newest first).
        
        Args:
            config: Base configuration for filtering.
            filter: Additional metadata filters.
            before: Only return checkpoints before this checkpoint_id.
            limit: Maximum number of checkpoints to return.
            
        Yields:
            CheckpointTuple instances matching the criteria.
            
        Example:
            ```python
            config = {"configurable": {"thread_id": "thread-1"}}
            async for checkpoint in saver.alist(config, limit=10):
                print(checkpoint.checkpoint['id'])
            ```
        """
        filter_info = self._build_kql_filter(config, filter, before)
        params = filter_info["params"]
        
        # Build query
        checkpoint_filter = ""
        if filter_info["filters"]:
            checkpoint_filter = "| where " + " and ".join(filter_info["filters"])
        
        limit_clause = f"| take {limit}" if limit else ""
        
        query = self.SELECT_CHECKPOINT_KQL.format(
            checkpoint_id_filter=checkpoint_filter,
            limit_clause=limit_clause,
        )
        
        # Replace parameters
        for param_name, param_value in params.items():
            query = query.replace(param_name, f"'{param_value}'")
        
        logger.debug("Listing checkpoints", extra={"filter": filter, "limit": limit})
        
        async with self._query() as client:
            response = await client.execute(self.database, query)
            
            if not response.primary_results:
                return
            
            for row in response.primary_results[0]:
                yield await self._load_checkpoint_tuple(row)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to Kusto asynchronously.
        
        Args:
            config: Configuration for this checkpoint.
            checkpoint: The checkpoint data to save.
            metadata: Metadata to associate with the checkpoint.
            new_versions: New channel versions for this checkpoint.
            
        Returns:
            Updated configuration with the checkpoint_id.
            
        Example:
            ```python
            config = await saver.aput(
                config={"configurable": {"thread_id": "thread-1"}},
                checkpoint=my_checkpoint,
                metadata={"source": "user"},
                new_versions={"channel1": "1.0"},
            )
            print(f"Saved checkpoint: {config['configurable']['checkpoint_id']}")
            ```
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")
        checkpoint_id = configurable.pop("checkpoint_id", None)
        
        logger.debug(
            "Putting checkpoint",
            extra={
                "thread_id": thread_id,
                "checkpoint_id": checkpoint["id"],
                "checkpoint_ns": checkpoint_ns,
            },
        )
        
        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }
        
        # Separate inline values from blobs
        blob_values = {}
        for k, v in checkpoint["channel_values"].items():
            if v is None or isinstance(v, (str, int, float, bool)):
                pass  # Keep in checkpoint JSON
            else:
                blob_values[k] = copy["channel_values"].pop(k)
        
        # Prepare blob data as dynamic array for the channel_values column
        channel_values_dynamic = []
        if blob_versions := {k: v for k, v in new_versions.items() if k in blob_values}:
            channel_values_dynamic = await asyncio.to_thread(
                self._dump_blobs,
                blob_values,
                blob_versions,
            )
        
        # Prepare checkpoint record with blobs stored in channel_values column
        checkpoint_record = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint["id"],
            "parent_checkpoint_id": checkpoint_id or "",
            "type": "",  # Reserved for future use
            "checkpoint_json": orjson.dumps(copy).decode(),
            "metadata_json": orjson.dumps(
                get_serializable_checkpoint_metadata(config, metadata)
            ).decode(),
            "channel_values": channel_values_dynamic,  # Dynamic column
            "created_at": "now()",
        }
        self._checkpoint_buffer.append(checkpoint_record)
        
        # Auto-flush if batch size reached
        if len(self._checkpoint_buffer) >= self.batch_size:
            await self.flush()
        
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.
        
        Args:
            config: Configuration of the related checkpoint.
            writes: List of (channel, value) tuples to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task in the execution tree.
            
        Example:
            ```python
            await saver.aput_writes(
                config={"configurable": {"thread_id": "thread-1", "checkpoint_id": "abc"}},
                writes=[("channel1", {"data": "value"})],
                task_id="task-1",
            )
            ```
        """
        logger.debug(
            "Putting writes",
            extra={
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_id": config["configurable"]["checkpoint_id"],
                "task_id": task_id,
                "num_writes": len(writes),
            },
        )
        
        records = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"].get("checkpoint_ns", ""),
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        
        self._write_buffer.extend(records)
        
        # Auto-flush if batch size reached
        if len(self._write_buffer) >= self.batch_size:
            await self.flush()

    async def flush(self) -> None:
        """Flush all buffered writes to Kusto.
        
        This method should be called periodically or when you need to ensure
        all pending data is ingested. With streaming ingestion, data typically
        appears in Kusto within 1 second after flushing.
        
        Example:
            ```python
            # After a batch of operations
            await saver.flush()
            ```
        """
        async with self.lock:
            # Flush checkpoints (includes channel_values as dynamic column)
            if self._checkpoint_buffer:
                await self._ingest_records("Checkpoints", self._checkpoint_buffer)
                logger.info(
                    "Flushed checkpoints",
                    extra={"count": len(self._checkpoint_buffer)},
                )
                self._checkpoint_buffer.clear()
            
            # Flush writes
            if self._write_buffer:
                await self._ingest_records("CheckpointWrites", self._write_buffer)
                logger.info(
                    "Flushed writes",
                    extra={"count": len(self._write_buffer)},
                )
                self._write_buffer.clear()

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread.
        
        Note: Kusto deletes are eventually consistent and may take time to propagate.
        Deleting checkpoints also removes associated blobs (stored in channel_values column).
        
        Args:
            thread_id: The thread ID to delete.
            
        Example:
            ```python
            await saver.adelete_thread("thread-1")
            ```
        """
        logger.info("Deleting thread", extra={"thread_id": thread_id})
        
        # Execute delete commands (blobs removed automatically with checkpoints)
        delete_queries = [
            self.DELETE_THREAD_KQL_CHECKPOINTS.format(thread_id=thread_id),
            self.DELETE_THREAD_KQL_WRITES.format(thread_id=thread_id),
        ]
        
        async with self._query() as client:
            for query in delete_queries:
                await client.execute_mgmt(self.database, query)
        
        logger.info("Thread deleted", extra={"thread_id": thread_id})

    async def _ingest_records(
        self, table_name: str, records: list[dict[str, Any]]
    ) -> None:
        """Ingest records into a Kusto table.
        
        Args:
            table_name: Name of the target table.
            records: List of record dictionaries to ingest.
        """
        if not records:
            return
        
        # Convert records to JSON lines format
        json_data = "\n".join(orjson.dumps(r).decode() for r in records)
        
        # Create ingestion properties
        ingestion_props = IngestionProperties(
            database=self.database,
            table=table_name,
            data_format=DataFormat.JSON,
        )
        
        # Ingest data
        async with self._ingest() as client:
            await client.ingest_from_stream(
                stream=json_data,
                ingestion_properties=ingestion_props,
            )

    @asynccontextmanager
    async def _query(self) -> AsyncIterator[AsyncKustoClient]:
        """Get query client context manager."""
        async with _ainternal.get_query_client(self.query_client) as client:
            yield client

    @asynccontextmanager
    async def _ingest(
        self,
    ) -> AsyncIterator[AsyncStreamingIngestClient]:
        """Get ingest client context manager."""
        async with _ainternal.get_ingest_client(self.ingest_client) as client:
            yield client

    async def _load_checkpoint_tuple(self, row: dict[str, Any]) -> CheckpointTuple:
        """Convert a Kusto row into a CheckpointTuple.
        
        Args:
            row: Dictionary representing a row from Kusto.
            
        Returns:
            CheckpointTuple with all data loaded and deserialized.
        """
        # Parse JSON fields
        checkpoint = row["checkpoint"]
        metadata = row["metadata"]
        channel_values = row.get("channel_values", [])
        pending_writes = row.get("pending_writes", [])
        
        # Load blobs
        blob_dict = await asyncio.to_thread(self._load_blobs, channel_values)
        
        # Merge blob values into checkpoint
        full_checkpoint = {
            **checkpoint,
            "channel_values": {
                **(checkpoint.get("channel_values") or {}),
                **blob_dict,
            },
        }
        
        # Load writes
        writes_list = await asyncio.to_thread(self._load_writes, pending_writes)
        
        # Build parent config
        parent_config = None
        if row.get("parent_checkpoint_id"):
            parent_config = {
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["parent_checkpoint_id"],
                }
            }
        
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["checkpoint_id"],
                }
            },
            checkpoint=full_checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=writes_list,
        )

    # Sync wrappers for backwards compatibility
    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Synchronous version of alist. Use from a different thread only."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncKustoSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                )
        except RuntimeError:
            pass
        
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),  # type: ignore
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Synchronous version of aget_tuple. Use from a different thread only."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncKustoSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                )
        except RuntimeError:
            pass
        
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Synchronous version of aput. Use from a different thread only."""
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Synchronous version of aput_writes. Use from a different thread only."""
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        """Synchronous version of adelete_thread. Use from a different thread only."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncKustoSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                )
        except RuntimeError:
            pass
        
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()


__all__ = ["AsyncKustoSaver"]
