import asyncio
from collections import defaultdict
from typing import Any, AsyncIterator, Iterator, Optional

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointAt,
    CheckpointTuple,
    SerializerProtocol,
)


class MemorySaver(BaseCheckpointSaver):
    storage: defaultdict[str, dict[str, tuple[bytes, bytes]]]

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
        at: Optional[CheckpointAt] = None,
    ) -> None:
        super().__init__(serde=serde, at=at)
        self.storage = defaultdict(dict)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        if ts := config["configurable"].get("thread_ts"):
            if saved := self.storage[thread_id].get(ts):
                checkpoint, metadata = saved
                return CheckpointTuple(
                    config=config,
                    checkpoint=self.serde.loads(checkpoint),
                    metadata=self.serde.loads(metadata),
                )
        else:
            if checkpoints := self.storage[thread_id]:
                ts = max(checkpoints.keys())
                checkpoint, metadata = checkpoints[ts]
                return CheckpointTuple(
                    config={"configurable": {"thread_id": thread_id, "thread_ts": ts}},
                    checkpoint=self.serde.loads(checkpoint),
                    metadata=self.serde.loads(metadata),
                )

    def list(
        self,
        config: RunnableConfig,
        *,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        for ts, (checkpoint, metadata) in self.storage[thread_id].items():
            if before and ts >= before["configurable"]["thread_ts"]:
                continue
            if limit is not None and limit <= 0:
                break
            limit -= 1
            yield CheckpointTuple(
                config={"configurable": {"thread_id": thread_id, "thread_ts": ts}},
                checkpoint=self.serde.loads(checkpoint),
                metadata=self.serde.loads(metadata),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: dict[str, Any] = None,
    ) -> RunnableConfig:
        self.storage[config["configurable"]["thread_id"]].update(
            {
                checkpoint["ts"]: (
                    self.serde.dumps(checkpoint),
                    self.serde.dumps(metadata or {}),
                )
            }
        )
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["ts"],
            }
        }

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.get_tuple, config
        )

    async def alist(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
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
