"""Azure CosmosDB implementation of LangGraph checkpointer (sync)."""

from __future__ import annotations

import base64
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosHttpResponseError
from azure.identity import CredentialUnavailableError, DefaultAzureCredential
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

COSMOSDB_KEY_SEPARATOR = "$"


class _CosmosSerializer:
    """Serializer wrapper for CosmosDB that base64-encodes binary data."""

    def __init__(self, serde: SerializerProtocol):
        self.serde = serde

    def dumps_typed(self, obj: Any) -> tuple[str, str]:
        type_, data = self.serde.dumps_typed(obj)
        data_base64 = base64.b64encode(data).decode("utf-8")
        return type_, data_base64

    def loads_typed(self, data: tuple[str, str]) -> Any:
        type_name, serialized_obj = data
        serialized_bytes = base64.b64decode(serialized_obj.encode("utf-8"))
        return self.serde.loads_typed((type_name, serialized_bytes))


def _make_checkpoint_key(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    """Create a checkpoint key for CosmosDB."""
    return COSMOSDB_KEY_SEPARATOR.join(
        ["checkpoint", thread_id, checkpoint_ns, checkpoint_id]
    )


def _make_checkpoint_writes_key(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: int | None,
) -> str:
    """Create a writes key for CosmosDB."""
    if idx is None:
        return COSMOSDB_KEY_SEPARATOR.join(
            ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id]
        )
    return COSMOSDB_KEY_SEPARATOR.join(
        ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx)]
    )


def _parse_checkpoint_key(cosmosdb_key: str) -> dict[str, str]:
    """Parse a checkpoint key."""
    parts = cosmosdb_key.split(COSMOSDB_KEY_SEPARATOR)
    if len(parts) != 4:
        raise ValueError(f"Invalid checkpoint key format: {cosmosdb_key}")
    namespace, thread_id, checkpoint_ns, checkpoint_id = parts
    if namespace != "checkpoint":
        raise ValueError("Expected checkpoint key to start with 'checkpoint'")
    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
    }


def _parse_checkpoint_writes_key(cosmosdb_key: str) -> dict[str, str]:
    """Parse a writes key."""
    parts = cosmosdb_key.split(COSMOSDB_KEY_SEPARATOR)
    if len(parts) != 6:
        raise ValueError(f"Invalid writes key format: {cosmosdb_key}")
    namespace, thread_id, checkpoint_ns, checkpoint_id, task_id, idx = parts
    if namespace != "writes":
        raise ValueError("Expected checkpoint key to start with 'writes'")
    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
        "task_id": task_id,
        "idx": idx,
    }


def _load_writes(
    serde: _CosmosSerializer, task_id_to_data: dict[tuple[str, str], dict]
) -> list[tuple[str, str, Any]]:
    """Load pending writes from CosmosDB data."""
    return [
        (task_id, data["channel"], serde.loads_typed((data["type"], data["value"])))
        for (task_id, _), data in task_id_to_data.items()
    ]


def _parse_checkpoint_data(
    serde: _CosmosSerializer,
    key: str,
    data: dict,
    pending_writes: list[tuple[str, str, Any]] | None = None,
) -> CheckpointTuple | None:
    """Parse checkpoint data from CosmosDB."""
    if not data:
        return None

    parsed_key = _parse_checkpoint_key(key)
    thread_id = parsed_key["thread_id"]
    checkpoint_ns = parsed_key["checkpoint_ns"]
    checkpoint_id = parsed_key["checkpoint_id"]

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }

    checkpoint = serde.loads_typed((data["type"], data["checkpoint"]))
    metadata = serde.loads_typed(data["metadata"])
    parent_checkpoint_id = data.get("parent_checkpoint_id", "")

    parent_config: RunnableConfig | None = (
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        if parent_checkpoint_id
        else None
    )

    return CheckpointTuple(
        config=config,
        checkpoint=checkpoint,
        metadata=metadata,
        parent_config=parent_config,
        pending_writes=pending_writes,
    )


class CosmosDBSaver(BaseCheckpointSaver):
    """CosmosDB synchronous implementation of BaseCheckpointSaver.

    Uses environment variables for connection configuration.

    Args:
        database_name: Name of the CosmosDB database.
        container_name: Name of the CosmosDB container.

    Environment Variables:
        COSMOSDB_ENDPOINT: CosmosDB endpoint URL (required).
        COSMOSDB_KEY: CosmosDB access key (optional, uses DefaultAzureCredential if not provided).

    Example:
        >>> import os
        >>> os.environ["COSMOSDB_ENDPOINT"] = "https://your-account.documents.azure.com:443/"
        >>> os.environ["COSMOSDB_KEY"] = "your_key"  # Optional
        >>>
        >>> checkpointer = CosmosDBSaver(
        ...     database_name="langgraph_db",
        ...     container_name="checkpoints"
        ... )
    """

    container: Any

    def __init__(self, database_name: str, container_name: str):
        super().__init__()

        endpoint = os.getenv("COSMOSDB_ENDPOINT")
        if not endpoint:
            raise ValueError("COSMOSDB_ENDPOINT environment variable is not set")

        key = os.getenv("COSMOSDB_KEY")

        try:
            if key:
                self.client = CosmosClient(endpoint, key)
                self.database = self.client.create_database_if_not_exists(database_name)
                self.container = self.database.create_container_if_not_exists(
                    id=container_name, partition_key=PartitionKey(path="/partition_key")
                )
            else:
                credential = DefaultAzureCredential()
                self.client = CosmosClient(endpoint, credential=credential)
                self.database = self.client.get_database_client(database_name)
                self.container = self.database.get_container_client(container_name)
        except CredentialUnavailableError as e:
            raise RuntimeError(
                "Failed to obtain default credentials. Ensure the environment is "
                "correctly configured for DefaultAzureCredential."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "An unexpected error occurred during CosmosClient initialization."
            ) from e

        self.cosmos_serde = _CosmosSerializer(self.serde)

    @classmethod
    @contextmanager
    def from_conn_info(
        cls,
        *,
        endpoint: str,
        key: str,
        database_name: str,
        container_name: str,
    ) -> Iterator[CosmosDBSaver]:
        """Create a CosmosDBSaver from explicit connection info.

        Args:
            endpoint: The CosmosDB endpoint URL.
            key: The CosmosDB access key.
            database_name: Name of the CosmosDB database.
            container_name: Name of the CosmosDB container.

        Yields:
            CosmosDBSaver: A configured saver instance.
        """
        old_endpoint = os.environ.get("COSMOSDB_ENDPOINT")
        old_key = os.environ.get("COSMOSDB_KEY")

        try:
            os.environ["COSMOSDB_ENDPOINT"] = endpoint
            os.environ["COSMOSDB_KEY"] = key
            saver = cls(database_name, container_name)
            yield saver
        finally:
            if old_endpoint is not None:
                os.environ["COSMOSDB_ENDPOINT"] = old_endpoint
            else:
                os.environ.pop("COSMOSDB_ENDPOINT", None)
            if old_key is not None:
                os.environ["COSMOSDB_KEY"] = old_key
            else:
                os.environ.pop("COSMOSDB_KEY", None)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to CosmosDB.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata for the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        key = _make_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        partition_key = _make_checkpoint_key(thread_id, checkpoint_ns, "")

        type_, serialized_checkpoint = self.cosmos_serde.dumps_typed(checkpoint)
        serialized_metadata = self.cosmos_serde.dumps_typed(metadata)

        data = {
            "partition_key": partition_key,
            "id": key,
            "thread_id": thread_id,
            "checkpoint": serialized_checkpoint,
            "type": type_,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }

        try:
            self.container.create_item(data)
        except CosmosHttpResponseError as e:
            print(f"Unexpected error ({e.status_code}): {e.message}")
            raise

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        is_upsert = all(w[0] in WRITES_IDX_MAP for w in writes)

        for idx, (channel, value) in enumerate(writes):
            key = _make_checkpoint_writes_key(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
            )
            partition_key = _make_checkpoint_writes_key(
                thread_id, checkpoint_ns, checkpoint_id, "", None
            )

            type_, serialized_value = self.cosmos_serde.dumps_typed(value)

            data = {
                "partition_key": partition_key,
                "id": key,
                "thread_id": thread_id,
                "channel": channel,
                "type": type_,
                "value": serialized_value,
            }

            if is_upsert:
                self.container.upsert_item(data)
            else:
                try:
                    self.container.create_item(data)
                except CosmosHttpResponseError as e:
                    if e.status_code != 409:  # Conflict: Item already exists
                        print(f"Unexpected error ({e.status_code}): {e.message}")
                        raise

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple from CosmosDB.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint tuple, or None if not found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        partition_key = _make_checkpoint_key(thread_id, checkpoint_ns, "")
        checkpoint_key = self._get_checkpoint_key(
            self.container, thread_id, checkpoint_ns, checkpoint_id
        )

        if not checkpoint_key:
            return None

        checkpoint_id = _parse_checkpoint_key(checkpoint_key)["checkpoint_id"]

        query = "SELECT * FROM c WHERE c.partition_key=@partition_key AND c.id=@checkpoint_key"
        parameters = [
            {"name": "@partition_key", "value": partition_key},
            {"name": "@checkpoint_key", "value": checkpoint_key},
        ]
        items = list(
            self.container.query_items(
                query=query, parameters=parameters, enable_cross_partition_query=True
            )
        )
        checkpoint_data = items[0] if items else {}

        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )

        return _parse_checkpoint_data(
            self.cosmos_serde,
            checkpoint_key,
            checkpoint_data,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from CosmosDB.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria (currently unused).
            before: List checkpoints created before this configuration (currently unused).
            limit: Maximum number of checkpoints to return (currently unused).

        Yields:
            Matching checkpoint tuples.
        """
        if not config:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        partition_key = _make_checkpoint_key(thread_id, checkpoint_ns, "")

        query = "SELECT * FROM c WHERE c.partition_key=@partition_key"
        parameters = [{"name": "@partition_key", "value": partition_key}]
        items = list(
            self.container.query_items(
                query=query, parameters=parameters, enable_cross_partition_query=True
            )
        )

        for data in items:
            if data and "checkpoint" in data and "metadata" in data:
                key = data["id"]
                checkpoint_id = _parse_checkpoint_key(key)["checkpoint_id"]
                pending_writes = self._load_pending_writes(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                checkpoint_tuple = _parse_checkpoint_data(
                    self.cosmos_serde, key, data, pending_writes=pending_writes
                )
                if checkpoint_tuple is not None:
                    yield checkpoint_tuple

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        """Load pending writes for a checkpoint."""
        partition_key = _make_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "", None
        )

        query = "SELECT * FROM c WHERE c.partition_key=@partition_key"
        parameters = [{"name": "@partition_key", "value": partition_key}]
        writes = list(
            self.container.query_items(
                query=query, parameters=parameters, enable_cross_partition_query=True
            )
        )

        parsed_keys = [_parse_checkpoint_writes_key(write["id"]) for write in writes]
        return _load_writes(
            self.cosmos_serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): write
                for write, parsed_key in sorted(
                    zip(writes, parsed_keys, strict=True), key=lambda x: x[1]["idx"]
                )
            },
        )

    def _get_checkpoint_key(
        self,
        container: Any,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str | None,
    ) -> str | None:
        """Get the checkpoint key, finding the latest if checkpoint_id is None."""
        if checkpoint_id:
            return _make_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        partition_key = _make_checkpoint_key(thread_id, checkpoint_ns, "")

        query = "SELECT * FROM c WHERE c.partition_key=@partition_key"
        parameters = [{"name": "@partition_key", "value": partition_key}]
        all_keys = list(
            container.query_items(
                query=query, parameters=parameters, enable_cross_partition_query=True
            )
        )

        if not all_keys:
            return None

        latest_key = max(
            all_keys, key=lambda k: _parse_checkpoint_key(k["id"])["checkpoint_id"]
        )
        return latest_key["id"]


__all__ = ["CosmosDBSaver"]
