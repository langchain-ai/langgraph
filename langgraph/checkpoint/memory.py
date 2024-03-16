from collections import defaultdict
from typing import Optional

from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointTuple


class MemorySaver(BaseCheckpointSaver):
    storage: defaultdict[str, dict[str, Checkpoint]] = Field(
        default_factory=lambda: defaultdict(dict)
    )

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        if value := self.get_tuple(config):
            return value.checkpoint

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        if config["configurable"].get("thread_ts"):
            if checkpoint := self.storage[config["configurable"]["thread_id"]].get(
                config["configurable"]["thread_ts"]
            ):
                return CheckpointTuple(config=config, checkpoint=checkpoint)
        else:
            if checkpoints := self.storage[config["configurable"]["thread_id"]]:
                thread_ts = max(checkpoints.keys())
                return CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": config["configurable"]["thread_id"],
                            "thread_ts": thread_ts,
                        }
                    },
                    checkpoint=checkpoints[thread_ts],
                )

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        self.storage[config["configurable"]["thread_id"]].update(
            {checkpoint["ts"]: checkpoint}
        )
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["ts"],
            }
        }
