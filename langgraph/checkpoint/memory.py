import asyncio
from collections import defaultdict
from typing import AsyncIterator, Iterator, Optional

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointAt,
    CheckpointTuple,
    SerializerProtocol,
)


class MemorySaver(BaseCheckpointSaver):
    """An in-memory checkpoint saver.

    This checkpoint saver stores checkpoints in memory using a defaultdict.

    Args:
        serde (Optional[SerializerProtocol]): The serializer to use for serializing and deserializing checkpoints. Defaults to None.
        at (Optional[CheckpointAt]): The checkpoint strategy to use. Defaults to None.

    Examples:
            import asyncio

            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.graph import StateGraph

            builder = StateGraph(int)
            builder.add_node("add_one", lambda x: x + 1)
            builder.set_entry_point("add_one")
            builder.set_finish_point("add_one")

            memory = MemorySaver()
            graph = builder.compile(checkpointer=memory)
            coro = graph.ainvoke(1, {"configurable": {"thread_id": "thread-1"}})
            asyncio.run(coro)  # Output: 2
    """

    storage: defaultdict[str, dict[str, Checkpoint]]

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
        at: Optional[CheckpointAt] = None,
    ) -> None:
        super().__init__(serde=serde, at=at)
        self.storage = defaultdict(dict)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the in-memory storage.

        This method retrieves a checkpoint tuple from the in-memory storage based on the
        provided config. If the config contains a "thread_ts" key, the checkpoint with
        the matching thread ID and timestamp is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        if ts := config["configurable"].get("thread_ts"):
            if checkpoint := self.storage[thread_id].get(ts):
                return CheckpointTuple(
                    config=config, checkpoint=self.serde.loads(checkpoint)
                )
        else:
            if checkpoints := self.storage[thread_id]:
                ts = max(checkpoints.keys())
                return CheckpointTuple(
                    config={"configurable": {"thread_id": thread_id, "thread_ts": ts}},
                    checkpoint=self.serde.loads(checkpoints[ts]),
                )

    def list(
        self,
        config: RunnableConfig,
        *,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the in-memory storage.

        This method retrieves a list of checkpoint tuples from the in-memory storage based
        on the provided config. The checkpoints are ordered by timestamp in descending order.

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified timestamp are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        thread_id = config["configurable"]["thread_id"]
        for ts, checkpoint in self.storage[thread_id].items():
            if before and ts >= before["configurable"]["thread_ts"]:
                continue
            if limit is not None and limit <= 0:
                break
            limit -= 1
            yield CheckpointTuple(
                config={"configurable": {"thread_id": thread_id, "thread_ts": ts}},
                checkpoint=self.serde.loads(checkpoint),
            )

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        """Save a checkpoint to the in-memory storage.

        This method saves a checkpoint to the in-memory storage. The checkpoint is associated
        with the provided config.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.

        Returns:
            RunnableConfig: The updated config containing the saved checkpoint's timestamp.
        """
        self.storage[config["configurable"]["thread_id"]].update(
            {checkpoint["ts"]: self.serde.dumps(checkpoint)}
        )
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["ts"],
            }
        }

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronous version of get_tuple.

        This method is an asynchronous wrapper around get_tuple that runs the synchronous
        method in a separate thread using asyncio.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, self.get_tuple, config
        )

    async def alist(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
        """Asynchronous version of list.

        This method is an asynchronous wrapper around list that runs the synchronous
        method in a separate thread using asyncio.

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of checkpoint tuples.
        """
        loop = asyncio.get_running_loop()
        iter = loop.run_in_executor(None, self.list, config)
        while True:
            try:
                yield await loop.run_in_executor(None, next, iter)
            except StopIteration:
                return

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> RunnableConfig:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint
        )
