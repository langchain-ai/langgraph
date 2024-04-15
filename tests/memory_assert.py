from collections import defaultdict
from typing import Any, Optional

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointAt,
    SerializerProtocol,
    copy_checkpoint,
)
from langgraph.checkpoint.memory import MemorySaver


class NoopSerializer(SerializerProtocol):
    def loads(self, data: bytes) -> Any:
        return data

    def dumps(self, obj: Any) -> bytes:
        return obj


class MemorySaverAssertImmutable(MemorySaver):
    serde = NoopSerializer()

    at = CheckpointAt.END_OF_STEP

    storage_for_copies: defaultdict[str, dict[str, Checkpoint]]

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
        at: Optional[CheckpointAt] = None,
    ) -> None:
        super().__init__(serde=serde, at=at)
        self.storage_for_copies = defaultdict(dict)

    def put(self, config: dict, checkpoint: Checkpoint) -> None:
        # assert checkpoint hasn't been modified since last written
        thread_id = config["configurable"]["thread_id"]
        if saved := super().get(config):
            assert self.storage_for_copies[thread_id][saved["ts"]] == saved
        self.storage_for_copies[thread_id][checkpoint["ts"]] = copy_checkpoint(
            checkpoint
        )
        # call super to write checkpoint
        return super().put(config, checkpoint)
