import asyncio
import pickle
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
    serde = pickle

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

    def list(self, config: RunnableConfig) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        for ts, checkpoint in self.storage[thread_id].items():
            yield CheckpointTuple(
                config={"configurable": {"thread_id": thread_id, "thread_ts": ts}},
                checkpoint=self.serde.loads(checkpoint),
            )

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
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
