"""
A checkpoint saver that stores checkpoints in a MongoDB database synchronously.
"""

import random
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from pymongo import MongoClient, UpdateOne

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import ChannelProtocol


class MongoDBSaver(BaseCheckpointSaver[str]):
    """A checkpoint saver that stores checkpoints in a MongoDB database synchronously.

    ```python
    from typing import Literal
    from langchain_core.runnables import ConfigurableField
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.mongodb import MongoDBSaver
    from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
    from langgraph.prebuilt import create_react_agent

    @tool
    def get_weather(city: Literal["nyc", "sf"]):
        if city == "nyc":
            return "It might be cloudy in nyc"
        elif city == "sf":
            return "It's always sunny in sf"
        else:
            raise AssertionError("Unknown city")


    tools = [get_weather]  # List of tools to be used by the agent
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key="your_api_key")

    # Create a MongoDBSaver instance and use it as a checkpointer
    with MongoDBSaver.from_conn_info(
        url="mongodb://localhost:27017", db_name="checkpoints"
    ) as checkpointer:

        # Create a React agent using the model and tools, along with the MongoDB checkpointer
        graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)

        # Example configuration for the agent's execution
        config = {"configurable": {"thread_id": "1"}}

        # Invoke the agent with a query
        res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

        # Retrieve the latest checkpoint
        latest_checkpoint = checkpointer.get(config)
        latest_checkpoint_tuple = checkpointer.get_tuple(config)

        # List all checkpoint tuples
        checkpoint_tuples = list(checkpointer.list(config))
    """

    def __init__(self, client: MongoClient, db_name: str) -> None:
        """Initialize the MongoDBSaver instance."""
        super().__init__()
        self.client = client
        self.db = self.client[db_name]

    @classmethod
    @contextmanager
    def from_conn_info(cls, url: str, db_name: str) -> Iterator["MongoDBSaver"]:
        """Initialize a MongoDBSaver instance with a connection.

        Args:
            url (str): The MongoDB connection URL.
            db_name (str): The name of the database.

        Returns:
            MongoDBSaver: An instance of the MongoDBSaver class.
        """
        client = None
        try:
            client = MongoClient(url)
            yield cls(client, db_name)
        finally:
            if client:
                client.close()

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        Args:
            config (Dict[str, Any]): The configuration dictionary.

        Returns:
            Optional[CheckpointTuple]: A tuple representing the checkpoint, or None if not found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        query = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
        if checkpoint_id:
            query["checkpoint_id"] = checkpoint_id

        doc = self.db["checkpoints"].find_one(query, sort=[("checkpoint_id", -1)])
        if doc:
            config_values = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["checkpoint_id"],
            }
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            pending_writes = [
                (
                    write_doc["task_id"],
                    write_doc["channel"],
                    self.serde.loads_typed((write_doc["type"], write_doc["value"])),
                )
                for write_doc in self.db["checkpoint_writes"].find(config_values)
            ]
            return CheckpointTuple(
                {"configurable": config_values},
                checkpoint,
                self.serde.loads(doc["metadata"]),
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
        return None

    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        Args:
            config (Optional[Dict[str, Any]]): The configuration dictionary.
            filter (Optional[Dict[str, Any]]): Additional filters for metadata.
            before (Optional[Dict[str, Any]]): Config to fetch checkpoints before a specific ID.
            limit (Optional[int]): Limit the number of results.

        Yields:
            CheckpointTuple: A tuple representing the checkpoint.
        """
        query: Dict[str, Any] = {}
        if config:
            query.update(
                {
                    "thread_id": config["configurable"]["thread_id"],
                    "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
                }
            )

        if filter:
            for key, value in filter.items():
                query[f"metadata.{key}"] = value

        if before:
            query["checkpoint_id"] = {"$lt": before["configurable"]["checkpoint_id"]}

        cursor = self.db["checkpoints"].find(query).sort("checkpoint_id", -1)
        if limit:
            cursor = cursor.limit(limit)

        for doc in cursor:
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": doc["checkpoint_id"],
                    }
                },
                checkpoint,
                self.serde.loads(doc["metadata"]),
                (
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
            )

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Dict[str, Any]:
        """Save a checkpoint to the database.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            checkpoint (Checkpoint): The checkpoint object to save.
            metadata (CheckpointMetadata): Metadata associated with the checkpoint.
            new_versions (ChannelVersions): Channel versions for the checkpoint.

        Returns:
            Dict[str, Any]: The updated configuration dictionary.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        doc = {
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "type": type_,
            "checkpoint": serialized_checkpoint,
            "metadata": self.serde.dumps(metadata),
        }
        upsert_query = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
        self.db["checkpoints"].update_one(upsert_query, {"$set": doc}, upsert=True)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            writes (Sequence[Tuple[str, Any]]): The writes to store.
            task_id (str): The task identifier.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        operations = []
        for idx, (channel, value) in enumerate(writes):
            upsert_query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "idx": idx,
            }
            type_, serialized_value = self.serde.dumps_typed(value)
            operations.append(
                UpdateOne(
                    upsert_query,
                    {
                        "$set": {
                            "channel": channel,
                            "type": type_,
                            "value": serialized_value,
                        }
                    },
                    upsert=True,
                )
            )
        self.db["checkpoint_writes"].bulk_write(operations)

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        """Generate the next version ID for a channel.

        Args:
            current (Optional[str]): The current version ID.
            channel (ChannelProtocol): The channel protocol.

        Returns:
            str: The next version ID.
        """
        current_v = int(current.split(".")[0]) if current else 0
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
