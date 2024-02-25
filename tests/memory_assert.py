from langchain_core.pydantic_v1 import Field

from langgraph.checkpoint.base import Checkpoint, CheckpointAt, copy_checkpoint
from langgraph.checkpoint.memory import MemorySaver


class MemorySaverAssertImmutable(MemorySaver):
    storage_for_copies: dict[str, Checkpoint] = Field(default_factory=dict)

    at = CheckpointAt.END_OF_STEP

    def put(self, config: dict, checkpoint: dict) -> None:
        # assert checkpoint hasn't been modified since last written
        thread_id = config["configurable"]["thread_id"]
        if saved := super().get(config):
            assert self.storage_for_copies[thread_id] == saved
        self.storage_for_copies[thread_id] = copy_checkpoint(checkpoint)
        # call super to write checkpoint
        super().put(config, checkpoint)
