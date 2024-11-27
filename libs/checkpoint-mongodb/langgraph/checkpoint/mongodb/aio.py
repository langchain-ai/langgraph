"""Async MongoDB Checkpoint Saver implementation for LangGraph."""

import random
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import ChannelProtocol


class AsyncMongoDBSaver(BaseCheckpointSaver[str]):
    """A checkpoint saver that stores checkpoints in a MongoDB database asynchronously.

    ```python
    from typing import Literal
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
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

    # Create a AsyncMongoDBSaver instance and use it as a checkpointer
    async with AsyncMongoDBSaver.from_conn_info(
        url="mongodb://localhost:27017", db_name="checkpoints"
    ) as checkpointer:

        # Create a React agent using the model and tools, along with the MongoDB checkpointer
        graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)

        # Example configuration for the agent's execution
        config = {"configurable": {"thread_id": "1"}}

        # ainvoke the agent with a query
        res = graph.ainvoke({"messages": [("human", "what's the weather in sf")]}, config)

        # Retrieve the latest checkpoint
        latest_checkpoint = checkpointer.get(config)
        latest_checkpoint_tuple = checkpointer.get_tuple(config)

        # List all checkpoint tuples
        checkpoint_tuples = list(checkpointer.list(config))
    ```
    """

    def __init__(self, client: AsyncIOMotorClient, db_name: str) -> None:
        """
        Initialize the MongoDB saver.

        Args:
            client (AsyncIOMotorClient): The MongoDB client instance.
            db_name (str): The name of the database.
        """
        super().__init__()
        self.client = client
        self.db = self.client[db_name]

    @classmethod
    @asynccontextmanager
    async def from_conn_info(
        cls, url: str, db_name: str
    ) -> AsyncIterator["AsyncMongoDBSaver"]:
        """
        Create an instance of AsyncMongoDBSaver using connection details.

        Args:
            url (str): The MongoDB connection URL.
            db_name (str): The database name.

        Yields:
            AsyncMongoDBSaver: The instance of the MongoDB saver.
        """
        client = AsyncIOMotorClient(url)
        try:
            yield cls(client, db_name)
        finally:
            client.close()

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        Retrieve a checkpoint tuple from the database asynchronously.

        Args:
            config (RunnableConfig): The runnable configuration.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple or None.
        """
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: Optional[str] = get_checkpoint_id(config)

        query = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
        if checkpoint_id:
            query["checkpoint_id"] = checkpoint_id

        cursor = self.db["checkpoints"].find(query).sort("checkpoint_id", -1).limit(1)
        async for doc in cursor:
            config_values = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["checkpoint_id"],
            }
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            serialized_writes = self.db["checkpoint_writes"].find(config_values)
            pending_writes = [
                (
                    doc["task_id"],
                    doc["channel"],
                    self.serde.loads_typed((doc["type"], doc["value"])),
                )
                async for doc in serialized_writes
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

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """
        List checkpoints from the database asynchronously.

        Args:
            config (Optional[RunnableConfig]): The runnable configuration.
            filter (Optional[Dict[str, Any]]): Metadata filter criteria.
            before (Optional[RunnableConfig]): Filter to include only older checkpoints.
            limit (Optional[int]): Maximum number of checkpoints to retrieve.

        Yields:
            CheckpointTuple: Retrieved checkpoint tuple.
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
            query.update({f"metadata.{key}": value for key, value in filter.items()})

        if before:
            query["checkpoint_id"] = {"$lt": before["configurable"]["checkpoint_id"]}

        cursor = self.db["checkpoints"].find(query).sort("checkpoint_id", -1)
        if limit:
            cursor = cursor.limit(limit)
        async for doc in cursor:
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

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Save a checkpoint to the database asynchronously.

        Args:
            config (RunnableConfig): The runnable configuration.
            checkpoint (Checkpoint): The checkpoint data.
            metadata (CheckpointMetadata): Metadata associated with the checkpoint.
            new_versions (ChannelVersions): New channel versions.

        Returns:
            RunnableConfig: Updated configuration.
        """
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"]["checkpoint_ns"]
        checkpoint_id: str = checkpoint["id"]
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
        await self.db["checkpoints"].update_one(
            upsert_query, {"$set": doc}, upsert=True
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
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Store intermediate writes linked to a checkpoint asynchronously.

        Args:
            config (RunnableConfig): The runnable configuration.
            writes (Sequence[Tuple[str, Any]]): The writes to store.
            task_id (str): The task identifier.
        """
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"]["checkpoint_ns"]
        checkpoint_id: str = config["configurable"]["checkpoint_id"]
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
        await self.db["checkpoint_writes"].bulk_write(operations)

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        """
        Generate the next version ID for a channel.

        Args:
            current (Optional[str]): The current version string.
            channel (ChannelProtocol): The channel protocol.

        Returns:
            str: The next version ID.
        """
        current_v: int = int(current.split(".")[0]) if current else 0
        next_v: int = current_v + 1
        next_h: float = random.random()
        return f"{next_v:032}.{next_h:016}"
