from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph_checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph_checkpoint.id import uuid6
from langgraph_checkpoint.memory import MemorySaver
from langgraph_checkpoint.types import ChannelProtocol


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, ChannelProtocol],
    step: int,
    *,
    id: Optional[str] = None,
) -> Checkpoint:
    """Create a checkpoint for the given channels."""
    ts = datetime.now(timezone.utc).isoformat()
    values: dict[str, Any] = {}
    for k, v in channels.items():
        values[k] = v.checkpoint()
    return Checkpoint(
        v=1,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=checkpoint["channel_versions"],
        versions_seen=checkpoint["versions_seen"],
        pending_sends=checkpoint.get("pending_sends", []),
        current_tasks={},
    )


class TestMemorySaver:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.memory_saver = MemorySaver()

        # objects for test setup
        self.config_1: RunnableConfig = {
            "configurable": {"thread_id": "thread-1", "thread_ts": "1"}
        }
        self.config_2: RunnableConfig = {
            "configurable": {"thread_id": "thread-2", "thread_ts": "2"}
        }

        self.chkpnt_1: Checkpoint = empty_checkpoint()
        self.chkpnt_2: Checkpoint = create_checkpoint(self.chkpnt_1, {}, 1)

        self.metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 2,
            "writes": {},
            "score": 1,
        }
        self.metadata_2: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
            "writes": {"foo": "bar"},
            "score": None,
        }

    async def test_search(self):
        # set up test
        # save checkpoints
        self.memory_saver.put(self.config_1, self.chkpnt_1, self.metadata_1)
        self.memory_saver.put(self.config_2, self.chkpnt_2, self.metadata_2)

        # call method / assertions
        query_1: CheckpointMetadata = {"source": "input"}  # search by 1 key
        query_2: CheckpointMetadata = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: CheckpointMetadata = {}  # search by no keys, return all checkpoints
        query_4: CheckpointMetadata = {"source": "update", "step": 1}  # no match

        search_results_1 = list(self.memory_saver.list(None, filter=query_1))
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == self.metadata_1

        search_results_2 = list(self.memory_saver.list(None, filter=query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == self.metadata_2

        search_results_3 = list(self.memory_saver.list(None, filter=query_3))
        assert len(search_results_3) == 2

        search_results_4 = list(self.memory_saver.list(None, filter=query_4))
        assert len(search_results_4) == 0

        # TODO: test before and limit params

    async def test_asearch(self):
        # set up test
        # save checkpoints
        self.memory_saver.put(self.config_1, self.chkpnt_1, self.metadata_1)
        self.memory_saver.put(self.config_2, self.chkpnt_2, self.metadata_2)

        # call method / assertions
        query_1: CheckpointMetadata = {"source": "input"}  # search by 1 key
        query_2: CheckpointMetadata = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: CheckpointMetadata = {}  # search by no keys, return all checkpoints
        query_4: CheckpointMetadata = {"source": "update", "step": 1}  # no match

        search_results_1 = [
            c async for c in self.memory_saver.alist(None, filter=query_1)
        ]
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == self.metadata_1

        search_results_2 = [
            c async for c in self.memory_saver.alist(None, filter=query_2)
        ]
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == self.metadata_2

        search_results_3 = [
            c async for c in self.memory_saver.alist(None, filter=query_3)
        ]
        assert len(search_results_3) == 2

        search_results_4 = [
            c async for c in self.memory_saver.alist(None, filter=query_4)
        ]
        assert len(search_results_4) == 0
