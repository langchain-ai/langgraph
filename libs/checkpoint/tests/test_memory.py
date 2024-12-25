from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointKey,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import MemorySaver


class TestMemorySaver:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.memory_saver = MemorySaver()

        # objects for test setup
        self.config_1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                # for backwards compatibility testing
                "thread_ts": "1",
            }
        }
        self.config_2: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "checkpoint_id": "2",
            }
        }
        self.config_3: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2-inner",
                "checkpoint_ns": "inner",
            }
        }

        self.chkpnt_1: Checkpoint = empty_checkpoint()
        self.chkpnt_2: Checkpoint = create_checkpoint(self.chkpnt_1, {}, 1)
        self.chkpnt_3: Checkpoint = empty_checkpoint()

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
        self.metadata_3: CheckpointMetadata = {}

    async def test_search(self) -> None:
        # set up test
        # save checkpoints
        self.memory_saver.put(self.config_1, self.chkpnt_1, self.metadata_1, {})
        self.memory_saver.put(self.config_2, self.chkpnt_2, self.metadata_2, {})
        self.memory_saver.put(self.config_3, self.chkpnt_3, self.metadata_3, {})

        # call method / assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
        query_4 = {"source": "update", "step": 1}  # no match

        search_results_1 = list(self.memory_saver.list(None, filter=query_1))
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == self.metadata_1

        search_results_2 = list(self.memory_saver.list(None, filter=query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == self.metadata_2

        search_results_3 = list(self.memory_saver.list(None, filter=query_3))
        assert len(search_results_3) == 3

        search_results_4 = list(self.memory_saver.list(None, filter=query_4))
        assert len(search_results_4) == 0

        # search by config (defaults to checkpoints across all namespaces)
        search_results_5 = list(
            self.memory_saver.list({"configurable": {"thread_id": "thread-2"}})
        )
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}

        # TODO: test before and limit params

    async def test_asearch(self) -> None:
        # set up test
        # save checkpoints
        self.memory_saver.put(self.config_1, self.chkpnt_1, self.metadata_1, {})
        self.memory_saver.put(self.config_2, self.chkpnt_2, self.metadata_2, {})
        self.memory_saver.put(self.config_3, self.chkpnt_3, self.metadata_3, {})

        # call method / assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
        query_4 = {"source": "update", "step": 1}  # no match

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
        assert len(search_results_3) == 3

        search_results_4 = [
            c async for c in self.memory_saver.alist(None, filter=query_4)
        ]
        assert len(search_results_4) == 0

    async def test_put_and_get_writes(self) -> None:
        CONFIG = self.config_2
        CKPT_ID = CONFIG["configurable"]["checkpoint_id"]
        THREAD_ID = CONFIG["configurable"]["thread_id"]
        CKPT_KEY = CheckpointKey(
            thread_id=THREAD_ID, checkpoint_ns="", checkpoint_id=CKPT_ID
        )

        TASK1 = "task1"
        TASK2 = "task2"

        # Test that writes are empty by default.
        assert self.memory_saver.get_writes(CONFIG, task_id=TASK1) is None

        # Test that writes are saved and retrieved correctly.
        writes1 = (("node1", 1), ("node2", "a"), ("node3", 1.0), ("node4", True))
        self.memory_saver.put_writes(CONFIG, writes1, TASK1)
        assert self.memory_saver.get_writes(CONFIG, task_id=TASK1) == (
            CKPT_KEY,
            writes1,
        )

        # Write to another task and check that writes are saved and retrieved correctly.
        writes2 = (("node1", 2), ("node2", "b"), ("node3", 2.0), ("node4", False))
        self.memory_saver.put_writes(CONFIG, writes2, TASK2)
        assert self.memory_saver.get_writes(CONFIG, task_id=TASK2) == (
            CKPT_KEY,
            writes2,
        )

        # Test that writes are not overwritten
        assert self.memory_saver.get_writes(CONFIG, task_id=TASK1) == (
            CKPT_KEY,
            writes1,
        )

    async def test_get_writes_when_multiple_entries_exist_pick_the_latest(self) -> None:
        TASK_ID = "task1"

        # Write writes associated with checkpoint 000.
        cfg1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "000",
            }
        }
        writes1 = [("node", 1)]
        self.memory_saver.put_writes(cfg1, writes1, TASK_ID)

        # Write writes associated with checkpoint 001.
        cfg2: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "001",
            }
        }
        writes2 = [("node", 2)]
        self.memory_saver.put_writes(cfg2, writes2, TASK_ID)

        # Check that the latest writes are returned.
        cfg: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "any",
                "checkpoint_id": "any",
            }
        }
        ckpt_key = CheckpointKey(
            thread_id="thread-1", checkpoint_ns="", checkpoint_id="001"
        )
        assert self.memory_saver.get_writes(cfg, task_id=TASK_ID) == (
            ckpt_key,
            tuple(writes2),
        )
