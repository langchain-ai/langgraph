from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib.metadata import version
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pymongo import MongoClient, UpdateOne
from pymongo.database import Database as MongoDatabase
from pymongo.driver_info import DriverInfo

from langgraph.checkpoint.base import (WRITES_IDX_MAP, BaseCheckpointSaver,
                                       ChannelVersions, Checkpoint,
                                       CheckpointMetadata, CheckpointTuple,
                                       get_checkpoint_id)

from .utils import dumps_metadata, loads_metadata


class MongoDBSaver(BaseCheckpointSaver):
    """A checkpointer that stores StateGraph checkpoints in a MongoDB database.

    A compound index as shown below will be added to each of the collections
    backing the saver (checkpoints, pending writes). If the collections pre-exist,
    and have indexes already, nothing will be done during initialization::

        keys=[("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)],
        unique=True,

    Args:
        client (MongoClient): The MongoDB connection.
        db_name (Optional[str]): Database name
        checkpoint_collection_name (Optional[str]): Name of Collection of Checkpoints
        writes_collection_name (Optional[str]): Name of Collection of intermediate writes.

    Examples:

        >>> from langgraph.checkpoint.mongodb import MongoDBSaver
        >>> from langgraph.graph import StateGraph
        >>> from pymongo import MongoClient
        >>>
        >>> builder = StateGraph(int)
        >>> builder.add_node("add_one", lambda x: x + 1)
        >>> builder.set_entry_point("add_one")
        >>> builder.set_finish_point("add_one")
        >>> client = MongoClient("mongodb://localhost:27017")
        >>> memory = MongoDBSaver(client)
        >>> graph = builder.compile(checkpointer=memory)
        >>> config = {"configurable": {"thread_id": "1"}}
        >>> graph.get_state(config)
        >>> result = graph.invoke(3, config)
        >>> graph.get_state(config)
        StateSnapshot(values=4, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef8b22d-df71-6ddc-8001-7c821b5c45fd'}}, metadata={'source': 'loop', 'writes': {'add_one': 4}, 'step': 1, 'parents': {}}, created_at='2024-10-15T18:25:34.088329+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef8b22d-df6f-6eec-8000-20f621dcf3b7'}}, tasks=())
    """

    client: MongoClient
    db: MongoDatabase

    def __init__(
        self,
        client: MongoClient,
        db_name: str = "checkpointing_db",
        checkpoint_collection_name: str = "checkpoints",
        writes_collection_name: str = "checkpoint_writes",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.client = client
        self.db = self.client[db_name]
        self.checkpoint_collection = self.db[checkpoint_collection_name]
        self.writes_collection = self.db[writes_collection_name]

        # Create indexes if not present
        if len(self.checkpoint_collection.list_indexes().to_list()) < 2:
            self.checkpoint_collection.create_index(
                keys=[("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)],
                unique=True,
            )
        if len(self.writes_collection.list_indexes().to_list()) < 2:
            self.writes_collection.create_index(
                keys=[
                    ("thread_id", 1),
                    ("checkpoint_ns", 1),
                    ("checkpoint_id", -1),
                    ("task_id", 1),
                    ("idx", 1),
                ],
                unique=True,
            )

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: Optional[str] = None,
        db_name: str = "checkpointing_db",
        checkpoint_collection_name: str = "checkpoints",
        writes_collection_name: str = "checkpoint_writes",
        **kwargs: Any,
    ) -> Iterator["MongoDBSaver"]:
        """Context manager to create a MongoDB checkpoint saver.

        A compound index as shown below will be added to each of the collections
        backing the saver (checkpoints, pending writes). If the collections pre-exist,
        and have indexes already, nothing will be done during initialization::

        keys=[("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)],
        unique=True

        Args:
            conn_string: MongoDB connection string. See [class:~pymongo.MongoClient].
            db_name: Database name. It will be created if it doesn't exist.
            checkpoint_collection_name: Checkpoint Collection name. Created if it doesn't exist.
            writes_collection_name: Collection name of intermediate writes. Created if it doesn't exist.
        Yields: A new MongoDBSaver.
        """
        client: Optional[MongoClient] = None
        try:
            client = MongoClient(
                conn_string,
                driver=DriverInfo(
                    name="Langgraph", version=version("langgraph-checkpoint-mongodb")
                ),
            )
            yield MongoDBSaver(
                client,
                db_name,
                checkpoint_collection_name,
                writes_collection_name,
                **kwargs,
            )
        finally:
            if client:
                client.close()

    def close(self) -> None:
        """Close the resources used by the MongoDBSaver."""
        self.client.close()

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

         This method retrieves a checkpoint tuple from the MongoDB database based on the
         provided config. If the config contains a "checkpoint_id" key, the checkpoint with
         the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
         for the given thread ID is retrieved.

         Args:
             config (RunnableConfig): The config to use for retrieving the checkpoint.

         Returns:
             Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.

        Examples:

             Basic:
             >>> config = {"configurable": {"thread_id": "1"}}
             >>> checkpoint_tuple = memory.get_tuple(config)
             >>> print(checkpoint_tuple)
             CheckpointTuple(...)

             With checkpoint ID:
             >>> config = {
             ...    "configurable": {
             ...        "thread_id": "1",
             ...        "checkpoint_ns": "",
             ...        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
             ...    }
             ... }
             >>> checkpoint_tuple = memory.get_tuple(config)
             >>> print(checkpoint_tuple)
             CheckpointTuple(...)
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id := get_checkpoint_id(config):
            query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        else:
            query = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}

        result = self.checkpoint_collection.find(
            query, sort=[("checkpoint_id", -1)], limit=1
        )
        for doc in result:
            config_values = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["checkpoint_id"],
            }
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            serialized_writes = self.writes_collection.find(config_values)
            pending_writes = [
                (
                    doc["task_id"],
                    doc["channel"],
                    self.serde.loads_typed((doc["type"], doc["value"])),
                )
                for doc in serialized_writes
            ]
            return CheckpointTuple(
                {"configurable": config_values},
                checkpoint,
                loads_metadata(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes,
            )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the MongoDB database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            filter (Optional[dict[str, Any]]): Additional filtering criteria for metadata. Defaults to None.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.

            Examples:
            >>> from langgraph.checkpoint.mongodb import MongoDBSaver
            >>> with MongoDBSaver.from_conn_string("mongodb://localhost:27017") as memory:
            ... # Run a graph, then list the checkpoints
            >>>     config = {"configurable": {"thread_id": "1"}}
            >>>     checkpoints = list(memory.list(config, limit=2))
            >>> print(checkpoints)
            [CheckpointTuple(...), CheckpointTuple(...)]
        """
        query = {}
        if config is not None:
            if "thread_id" in config["configurable"]:
                query["thread_id"] = config["configurable"]["thread_id"]
            if "checkpoint_ns" in config["configurable"]:
                query["checkpoint_ns"] = config["configurable"]["checkpoint_ns"]

        if filter:
            for key, value in filter.items():
                query[f"metadata.{key}"] = dumps_metadata(value)

        if before is not None:
            query["checkpoint_id"] = {"$lt": before["configurable"]["checkpoint_id"]}

        result = self.checkpoint_collection.find(
            query, limit=0 if limit is None else limit, sort=[("checkpoint_id", -1)]
        )

        for doc in result:
            config_values = {
                "thread_id": doc["thread_id"],
                "checkpoint_ns": doc["checkpoint_ns"],
                "checkpoint_id": doc["checkpoint_id"],
            }
            serialized_writes = self.writes_collection.find(config_values)
            pending_writes = [
                (
                    wrt["task_id"],
                    wrt["channel"],
                    self.serde.loads_typed((wrt["type"], wrt["value"])),
                )
                for wrt in serialized_writes
            ]

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": doc["checkpoint_id"],
                    }
                },
                checkpoint=self.serde.loads_typed((doc["type"], doc["checkpoint"])),
                metadata=loads_metadata(doc["metadata"]),
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": doc["thread_id"],
                            "checkpoint_ns": doc["checkpoint_ns"],
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes=pending_writes,
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the MongoDB database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Examples:

            >>> from langgraph.checkpoint.mongodb import MongoDBSaver
            >>> with MongoDBSaver.from_conn_string("mongodb://localhost:27017") as memory:
            >>>     config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            >>>     checkpoint = {"ts": "2024-05-04T06:32:42.235444+00:00", "id": "1ef4f797-8335-6428-8001-8a1503f9b875", "data": {"key": "value"}}
            >>>     saved_config = memory.put(config, checkpoint, {"source": "input", "step": 1, "writes": {"key": "value"}}, {})
            >>> print(saved_config)
            {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef4f797-8335-6428-8001-8a1503f9b875'}}
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        doc = {
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "type": type_,
            "checkpoint": serialized_checkpoint,
            "metadata": dumps_metadata(metadata),
        }
        upsert_query = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
        self.checkpoint_collection.update_one(upsert_query, {"$set": doc}, upsert=True)
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

        This method saves intermediate writes associated with a checkpoint to the MongoDB database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        set_method = (  # Allow replacement on existing writes only if there were errors.
            "$set" if all(w[0] in WRITES_IDX_MAP for w in writes) else "$setOnInsert"
        )
        operations = []
        for idx, (channel, value) in enumerate(writes):
            upsert_query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "idx": WRITES_IDX_MAP.get(channel, idx),
            }
            type_, serialized_value = self.serde.dumps_typed(value)
            operations.append(
                UpdateOne(
                    upsert_query,
                    {
                        set_method: {
                            "channel": channel,
                            "type": type_,
                            "value": serialized_value,
                        }
                    },
                    upsert=True,
                )
            )
        self.writes_collection.bulk_write(operations)
