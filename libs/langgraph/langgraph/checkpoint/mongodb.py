import pickle
from contextlib import AbstractContextManager, contextmanager
from types import TracebackType
from typing import Iterator, Optional

from langchain_core.runnables import RunnableConfig
from typing_extensions import Self
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointAt,
    CheckpointTuple,
    SerializerProtocol,
)
from pymongo import MongoClient


class MongoDBSaver(BaseCheckpointSaver, AbstractContextManager):
    """A checkpoint saver that stores checkpoints in a MongoDB database.

    Note: This saver is not thread-safe. If you need to use it in a multi-threaded environment,
    you should create a separate saver instance for each thread. This is because the saver uses
    a single MongoDB connection, which is not thread-safe.

    Args:
        conn: The MongoDB connection object.
        serde: The serializer/deserializer to use for serializing and deserializing checkpoints.
        at: The checkpoint-at strategy to use for determining the checkpoint timestamp.

    Examples:

        >>> from pymongo import MongoClient
        >>> from langgraph.checkpoint.mongodb import MongoDBSaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> builder = StateGraph(int)
        >>> builder.add_node("add_one", lambda x: x + 1)
        >>> builder.set_entry_point("add_one")
        >>> builder.set_finish_point("add_one")
        >>> memory = MongoDBSaver.from_conn_string(MONGO_CONN_STRING)
        >>> graph = builder.compile(checkpointer=memory)
        >>> config = {"configurable": {"thread_id": "1"}}
        >>> graph.get_state(config)
        >>> result = graph.invoke(3, config)
        >>> graph.get_state(config)
        StateSnapshot(values=4, next=(), config={'configurable': {'thread_id': '1', 'thread_ts': '2024-05-04T06:32:42.235444+00:00'}}, parent_config=None)
    """

    serde = pickle
    conn: MongoClient
    is_setup: bool

    def __init__(
        self,
        conn: MongoClient,
        *,
        serde: Optional[SerializerProtocol] = None,
        at: Optional[CheckpointAt] = None,
    ) -> None:
        super().__init__(serde=serde, at=at)
        self.conn = conn
        self.is_setup = False

    @classmethod
    def from_conn_string(cls, conn_string: str) -> "MongoDBSaver":
        """Create a new MongoDBSaver instance from a MongoDB connection string."""
        return MongoDBSaver(conn=MongoClient(conn_string))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        __exc_type: Optional[type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return self.conn.close()

    def setup(self) -> None:
        """Set up the MongoDB collection for storing checkpoints.
        This method creates an index on the `thread_id` and `thread_ts` fields to ensure that the
        collection is efficient for querying checkpoints by thread ID and timestamp.

        Note: This method is idempotent and can be called multiple times without any side effects.
        """
        if self.is_setup:
            return

        collection = self.conn["langgraph_checkpoint_db"][
            "langgraph_checkpoint_db_collection"
        ]
        collection.create_index([("thread_id", 1), ("thread_ts", 1)], unique=True)
        self.is_setup = True

    @contextmanager
    def cursor(self, transaction: bool = True):
        """Get a cursor to the MongoDB collection for storing checkpoints."""
        self.setup()
        collection = self.conn["langgraph_checkpoint_db"][
            "langgraph_checkpoint_db_collection"
        ]
        try:
            yield collection
        finally:
            if transaction:
                pass

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get the latest checkpoint for the given thread ID.

        This method returns the latest checkpoint for the given thread ID. If the thread ID has
        multiple checkpoints, the latest one is returned. If the thread ID has no checkpoints, this
        method returns `None`.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The latest checkpoint for the given thread ID, or `None` if
            the thread ID has no checkpoints.

        """
        with self.cursor(transaction=False) as collection:
            if config["configurable"].get("thread_ts"):
                if value := collection.find_one(
                    {
                        "thread_id": config["configurable"]["thread_id"],
                        "thread_ts": config["configurable"]["thread_ts"],
                    }
                ):
                    return CheckpointTuple(
                        config,
                        self.serde.loads(value["checkpoint"]),
                        (
                            {
                                "configurable": {
                                    "thread_id": config["configurable"]["thread_id"],
                                    "thread_ts": value["parent_ts"],
                                }
                            }
                            if value.get("parent_ts")
                            else None
                        ),
                    )
            else:
                if value := collection.find_one(
                    {"thread_id": config["configurable"]["thread_id"]},
                    sort=[("thread_ts", -1)],
                ):
                    return CheckpointTuple(
                        {
                            "configurable": {
                                "thread_id": value["thread_id"],
                                "thread_ts": value["thread_ts"],
                            }
                        },
                        self.serde.loads(value["checkpoint"]),
                        (
                            {
                                "configurable": {
                                    "thread_id": value["thread_id"],
                                    "thread_ts": value["parent_ts"],
                                }
                            }
                            if value.get("parent_ts")
                            else None
                        ),
                    )

    def list(self, config: RunnableConfig) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves all checkpoints for the given thread ID in reverse chronological order.

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.

        Returns:
            Iterator[CheckpointTuple]: An iterator that yields all checkpoints for the given thread
            ID in reverse chronological order.
        """
        with self.cursor(transaction=False) as collection:
            for value in collection.find(
                {"thread_id": config["configurable"]["thread_id"]},
                sort=[("thread_ts", -1)],
            ):
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "thread_ts": value["thread_ts"],
                        }
                    },
                    self.serde.loads(value["checkpoint"]),
                    (
                        {
                            "configurable": {
                                "thread_id": value["thread_id"],
                                "thread_ts": value["parent_ts"],
                            }
                        }
                        if value.get("parent_ts")
                        else None
                    ),
                )

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves the given checkpoint to the database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to use for saving the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.

        Returns:
            RunnableConfig: The updated config with the thread ID and timestamp of the saved
            checkpoint.
        """
        with self.cursor() as collection:
            collection.insert_one(
                {
                    "thread_id": config["configurable"]["thread_id"],
                    "thread_ts": checkpoint["ts"],
                    "parent_ts": config["configurable"].get("thread_ts"),
                    "checkpoint": self.serde.dumps(checkpoint),
                }
            )
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["ts"],
            }
        }
