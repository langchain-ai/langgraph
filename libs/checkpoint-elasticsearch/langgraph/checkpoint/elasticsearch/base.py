from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Optional

from elasticsearch import Elasticsearch
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    Checkpoint,
    SerializerProtocol,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

MetadataInput = Optional[dict[str, Any]]


class BaseElasticsearchSaver(BaseCheckpointSaver[str]):
    """Base Elasticsearch checkpoint saver implementation.

    This class provides the core functionality for storing and retrieving
    checkpoints using Elasticsearch as the storage backend.
    """

    CHECKPOINTS_INDEX_TEMPLATE = {
        "mappings": {
            "properties": {
                "thread_id": {"type": "keyword"},
                "checkpoint_ns": {"type": "keyword"},
                "checkpoint_id": {"type": "keyword"},
                "parent_checkpoint_id": {"type": "keyword"},
                "checkpoint_data": {"type": "binary"},
                "checkpoint_type": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": True},
                "timestamp": {"type": "date"},
                "channel_versions": {"type": "object", "enabled": False},
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "1s",
        },
    }

    WRITES_INDEX_TEMPLATE = {
        "mappings": {
            "properties": {
                "thread_id": {"type": "keyword"},
                "checkpoint_ns": {"type": "keyword"},
                "checkpoint_id": {"type": "keyword"},
                "task_id": {"type": "keyword"},
                "task_path": {"type": "keyword"},
                "idx": {"type": "integer"},
                "channel": {"type": "keyword"},
                "value_type": {"type": "keyword"},
                "value_data": {"type": "binary"},
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "1s",
        },
    }

    client: Elasticsearch
    checkpoints_index: str
    writes_index: str

    def __init__(
        self,
        es_url: str | None = None,
        api_key: str | None = None,
        index_prefix: str = "langgraph",
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
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

        # Initialize Elasticsearch client with API key authentication
        self.client = Elasticsearch(
            hosts=[es_url],
            api_key=api_key,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3,
        )

        # Set index names
        self.checkpoints_index = f"{index_prefix}_checkpoints"
        self.writes_index = f"{index_prefix}_writes"

        # Ensure indices exist
        self._ensure_indices()

    def _ensure_indices(self) -> None:
        """Ensure that the required indices exist with proper mappings."""
        # Create checkpoints index if it doesn't exist
        if not self.client.indices.exists(index=self.checkpoints_index):
            self.client.indices.create(
                index=self.checkpoints_index, body=self.CHECKPOINTS_INDEX_TEMPLATE
            )

        # Create writes index if it doesn't exist
        if not self.client.indices.exists(index=self.writes_index):
            self.client.indices.create(
                index=self.writes_index, body=self.WRITES_INDEX_TEMPLATE
            )

    def _get_document_id(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        """Generate a document ID for a checkpoint."""
        return f"{thread_id}#{checkpoint_ns}#{checkpoint_id}"

    def _load_checkpoint_data(
        self, checkpoint_type: str, checkpoint_data: bytes
    ) -> Checkpoint:
        """Deserialize checkpoint data."""
        return self.serde.loads_typed((checkpoint_type, checkpoint_data))

    def _dump_checkpoint_data(self, checkpoint: Checkpoint) -> tuple[str, bytes]:
        """Serialize checkpoint data."""
        return self.serde.dumps_typed(checkpoint)

    def _load_writes(
        self, writes_docs: list[dict[str, Any]]
    ) -> list[tuple[str, str, Any]]:
        """Load and deserialize writes from Elasticsearch documents."""
        return [
            (
                doc["_source"]["task_id"],
                doc["_source"]["channel"],
                self.serde.loads_typed(
                    (doc["_source"]["value_type"], doc["_source"]["value_data"])
                ),
            )
            for doc in writes_docs
        ]

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prepare writes for bulk indexing."""
        docs = []
        for idx, (channel, value) in enumerate(writes):
            value_type, value_data = self.serde.dumps_typed(value)
            doc = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "value_type": value_type,
                "value_data": value_data,
            }
            docs.append(doc)
        return docs

    def _search_where(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Build Elasticsearch query from config, filter, and before parameters."""
        query = {"bool": {"must": []}}

        # Add config filters
        if config:
            query["bool"]["must"].append(
                {"term": {"thread_id": config["configurable"]["thread_id"]}}
            )

            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            query["bool"]["must"].append({"term": {"checkpoint_ns": checkpoint_ns}})

            if checkpoint_id := get_checkpoint_id(config):
                query["bool"]["must"].append({"term": {"checkpoint_id": checkpoint_id}})

        # Add metadata filters
        if filter:
            for key, value in filter.items():
                query["bool"]["must"].append({"term": {f"metadata.{key}": value}})

        # Add before filter
        if before is not None:
            query["bool"]["must"].append(
                {"range": {"checkpoint_id": {"lt": get_checkpoint_id(before)}}}
            )

        return query

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Generate the next version ID for a channel."""
        import random

        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
