from __future__ import annotations

import base64
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any, Optional, cast

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.elasticsearch.base import BaseElasticsearchSaver

MetadataInput = Optional[dict[str, Any]]


class AsyncElasticsearchSaver(BaseElasticsearchSaver):
    """Asynchronous Elasticsearch checkpoint saver.

    This class provides asynchronous operations for storing and retrieving
    checkpoints using Elasticsearch as the storage backend.

    Args:
        es_url: Elasticsearch cluster URL. If not provided, uses ES_URL environment variable.
        api_key: Elasticsearch API key. If not provided, uses ES_API_KEY environment variable.
        index_prefix: Prefix for Elasticsearch indices. Defaults to "langgraph".
        serde: Serializer for encoding/decoding checkpoints. Defaults to JsonPlusSerializer.

    Examples:
        Basic usage:
        >>> saver = AsyncElasticsearchSaver(
        ...     es_url="https://localhost:9200",
        ...     api_key="your-api-key"
        ... )
        >>> # Use with a graph
        >>> graph = builder.compile(checkpointer=saver)

        Using environment variables:
        >>> import os
        >>> os.environ["ES_URL"] = "https://localhost:9200"
        >>> os.environ["ES_API_KEY"] = "your-api-key"
        >>> saver = AsyncElasticsearchSaver()
    """

    async_client: AsyncElasticsearch

    def __init__(
        self,
        es_url: str | None = None,
        api_key: str | None = None,
        index_prefix: str = "langgraph",
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        # Initialize serializers first
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        self.serde = serde or JsonPlusSerializer()
        self.jsonplus_serde = JsonPlusSerializer()

        # Get connection details from environment if not provided
        es_url = es_url or os.environ.get("ES_URL")
        api_key = api_key or os.environ.get("ES_API_KEY")

        if not es_url:
            raise ValueError(
                "ES_URL must be provided either as parameter or environment variable"
            )
        if not api_key:
            raise ValueError(
                "ES_API_KEY must be provided either as parameter or environment variable"
            )

        # Initialize async Elasticsearch client with API key authentication
        self.async_client = AsyncElasticsearch(
            hosts=[es_url],
            api_key=api_key,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3,
        )

        # Set index names
        self.checkpoints_index = f"{index_prefix}_checkpoints"
        self.writes_index = f"{index_prefix}_writes"

    async def _ensure_indices(self) -> None:
        """Ensure that the required indices exist with proper mappings."""
        # Create checkpoints index if it doesn't exist
        if not await self.async_client.indices.exists(index=self.checkpoints_index):
            await self.async_client.indices.create(
                index=self.checkpoints_index, body=self.CHECKPOINTS_INDEX_TEMPLATE
            )

        # Create writes index if it doesn't exist
        if not await self.async_client.indices.exists(index=self.writes_index):
            await self.async_client.indices.create(
                index=self.writes_index, body=self.WRITES_INDEX_TEMPLATE
            )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Asynchronously get a checkpoint tuple from Elasticsearch.

        This method retrieves a checkpoint tuple from Elasticsearch based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint
        with the matching thread ID and checkpoint ID is retrieved. Otherwise, the
        latest checkpoint for the given thread ID is retrieved.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if not found.
        """
        await self._ensure_indices()

        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        try:
            if checkpoint_id := get_checkpoint_id(config):
                # Get specific checkpoint
                doc_id = self._get_document_id(thread_id, checkpoint_ns, checkpoint_id)
                response = await self.async_client.get(
                    index=self.checkpoints_index, id=doc_id
                )
                doc = response["_source"]
            else:
                # Get latest checkpoint for thread
                query = {
                    "bool": {
                        "must": [
                            {"term": {"thread_id": thread_id}},
                            {"term": {"checkpoint_ns": checkpoint_ns}},
                        ]
                    }
                }

                response = await self.async_client.search(
                    index=self.checkpoints_index,
                    body={
                        "query": query,
                        "sort": [{"checkpoint_id": {"order": "desc"}}],
                        "size": 1,
                    },
                )

                if not response["hits"]["hits"]:
                    return None

                doc = response["hits"]["hits"][0]["_source"]
                checkpoint_id = doc["checkpoint_id"]

                # Update config with checkpoint_id for consistency
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                }

            # Deserialize checkpoint
            checkpoint = self._load_checkpoint_data(
                doc["checkpoint_type"], base64.b64decode(doc["checkpoint_data"])
            )

            # Get pending writes for this checkpoint
            writes_query = {
                "bool": {
                    "must": [
                        {"term": {"thread_id": thread_id}},
                        {"term": {"checkpoint_ns": checkpoint_ns}},
                        {"term": {"checkpoint_id": checkpoint_id}},
                    ]
                }
            }

            writes_response = await self.async_client.search(
                index=self.writes_index,
                body={
                    "query": writes_query,
                    "sort": [{"task_id": "asc"}, {"idx": "asc"}],
                    "size": 10000,
                },
            )

            pending_writes = []
            for write_doc in writes_response["hits"]["hits"]:
                write_data = write_doc["_source"]
                pending_writes.append(
                    (
                        write_data["task_id"],
                        write_data["channel"],
                        self.serde.loads_typed(
                            (
                                write_data["value_type"],
                                base64.b64decode(write_data["value_data"]),
                            )
                        ),
                    )
                )

            # Build parent config if exists
            parent_config = None
            if doc.get("parent_checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": doc["parent_checkpoint_id"],
                    }
                }

            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=cast(CheckpointMetadata, doc.get("metadata", {})),
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

        except NotFoundError:
            return None

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints from Elasticsearch.

        This method retrieves a list of checkpoint tuples from Elasticsearch based
        on the provided config. The checkpoints are ordered by checkpoint ID in
        descending order (newest first).

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An async iterator of checkpoint tuples.
        """
        await self._ensure_indices()

        query = self._search_where(config, filter, before)

        search_params = {
            "index": self.checkpoints_index,
            "body": {
                "query": query,
                "sort": [{"checkpoint_id": {"order": "desc"}}],
            },
        }

        if limit:
            search_params["body"]["size"] = limit
        else:
            search_params["body"]["size"] = 10000

        response = await self.async_client.search(**search_params)

        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            thread_id = doc["thread_id"]
            checkpoint_ns = doc["checkpoint_ns"]
            checkpoint_id = doc["checkpoint_id"]

            # Deserialize checkpoint
            checkpoint = self._load_checkpoint_data(
                doc["checkpoint_type"], base64.b64decode(doc["checkpoint_data"])
            )

            # Get pending writes for this checkpoint
            writes_query = {
                "bool": {
                    "must": [
                        {"term": {"thread_id": thread_id}},
                        {"term": {"checkpoint_ns": checkpoint_ns}},
                        {"term": {"checkpoint_id": checkpoint_id}},
                    ]
                }
            }

            writes_response = await self.async_client.search(
                index=self.writes_index,
                body={
                    "query": writes_query,
                    "sort": [{"task_id": "asc"}, {"idx": "asc"}],
                    "size": 10000,
                },
            )

            pending_writes = []
            for write_doc in writes_response["hits"]["hits"]:
                write_data = write_doc["_source"]
                pending_writes.append(
                    (
                        write_data["task_id"],
                        write_data["channel"],
                        self.serde.loads_typed(
                            (
                                write_data["value_type"],
                                base64.b64decode(write_data["value_data"]),
                            )
                        ),
                    )
                )

            # Build parent config if exists
            parent_config = None
            if doc.get("parent_checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": doc["parent_checkpoint_id"],
                    }
                }

            config_for_checkpoint = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

            yield CheckpointTuple(
                config=config_for_checkpoint,
                checkpoint=checkpoint,
                metadata=cast(CheckpointMetadata, doc.get("metadata", {})),
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint in Elasticsearch.

        This method saves a checkpoint to Elasticsearch. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata for the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        await self._ensure_indices()

        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        # Serialize checkpoint data
        checkpoint_type, checkpoint_data = self._dump_checkpoint_data(checkpoint)

        # Prepare document
        doc = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "checkpoint_type": checkpoint_type,
            "checkpoint_data": base64.b64encode(checkpoint_data).decode(),
            "metadata": get_checkpoint_metadata(config, metadata),
            "timestamp": checkpoint["ts"],
            "channel_versions": checkpoint.get("channel_versions", {}),
        }

        # Store in Elasticsearch
        doc_id = self._get_document_id(thread_id, checkpoint_ns, checkpoint_id)
        await self.async_client.index(
            index=self.checkpoints_index, id=doc_id, body=doc, refresh="wait_for"
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronously store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to Elasticsearch.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        if not writes:
            return

        await self._ensure_indices()

        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = str(config["configurable"]["checkpoint_id"])

        # Prepare bulk operations
        bulk_operations = []
        for idx, (channel, value) in enumerate(writes):
            value_type, value_data = self.serde.dumps_typed(value)

            doc_id = f"{thread_id}#{checkpoint_ns}#{checkpoint_id}#{task_id}#{idx}"

            bulk_operations.append(
                {"index": {"_index": self.writes_index, "_id": doc_id}}
            )

            bulk_operations.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "task_path": task_path,
                    "idx": idx,
                    "channel": channel,
                    "value_type": value_type,
                    "value_data": base64.b64encode(value_data).decode(),
                }
            )

        if bulk_operations:
            await self.async_client.bulk(body=bulk_operations, refresh="wait_for")

    async def adelete_thread(self, thread_id: str) -> None:
        """Asynchronously delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID whose checkpoints should be deleted.
        """
        await self._ensure_indices()

        # Delete checkpoints
        await self.async_client.delete_by_query(
            index=self.checkpoints_index,
            body={"query": {"term": {"thread_id": thread_id}}},
            refresh=True,
        )

        # Delete writes
        await self.async_client.delete_by_query(
            index=self.writes_index,
            body={"query": {"term": {"thread_id": thread_id}}},
            refresh=True,
        )

    async def aclose(self) -> None:
        """Close the async Elasticsearch client."""
        await self.async_client.close()

    # Sync methods - not supported in async version
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        raise NotImplementedError(
            "The AsyncElasticsearchSaver does not support sync methods. "
            "Use aget_tuple() instead or consider using ElasticsearchSaver."
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ):
        raise NotImplementedError(
            "The AsyncElasticsearchSaver does not support sync methods. "
            "Use alist() instead or consider using ElasticsearchSaver."
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise NotImplementedError(
            "The AsyncElasticsearchSaver does not support sync methods. "
            "Use aput() instead or consider using ElasticsearchSaver."
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise NotImplementedError(
            "The AsyncElasticsearchSaver does not support sync methods. "
            "Use aput_writes() instead or consider using ElasticsearchSaver."
        )

    def delete_thread(self, thread_id: str) -> None:
        raise NotImplementedError(
            "The AsyncElasticsearchSaver does not support sync methods. "
            "Use adelete_thread() instead or consider using ElasticsearchSaver."
        )
