import logging
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class MemoryPydantic(BaseModel):
    foo: str


class TestMemorySaver:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.memory_saver = InMemorySaver()

        # objects for test setup
        self.config_1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "1",
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

    def test_combined_metadata(self) -> None:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "__super_private_key": "super_private_value",
            },
            "metadata": {"run_id": "my_run_id"},
        }
        self.memory_saver.put(
            config, self.chkpnt_2, self.metadata_2, self.chkpnt_2["channel_versions"]
        )
        checkpoint = self.memory_saver.get_tuple(config)
        assert checkpoint is not None
        assert checkpoint.metadata == {
            **self.metadata_2,
            "run_id": "my_run_id",
        }

    async def test_search(self) -> None:
        # set up test
        # save checkpoints
        self.memory_saver.put(
            self.config_1,
            self.chkpnt_1,
            self.metadata_1,
            self.chkpnt_1["channel_versions"],
        )
        self.memory_saver.put(
            self.config_2,
            self.chkpnt_2,
            self.metadata_2,
            self.chkpnt_2["channel_versions"],
        )
        self.memory_saver.put(
            self.config_3,
            self.chkpnt_3,
            self.metadata_3,
            self.chkpnt_3["channel_versions"],
        )

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

    def test_list_before(self) -> None:
        """Test that `before` parameter filters checkpoints created before a given config."""
        self.memory_saver.put(
            self.config_1,
            self.chkpnt_1,
            self.metadata_1,
            self.chkpnt_1["channel_versions"],
        )
        self.memory_saver.put(
            self.config_2,
            self.chkpnt_2,
            self.metadata_2,
            self.chkpnt_2["channel_versions"],
        )

        # checkpoint_2 has a later ID, so `before=config_2_result` should
        # return only checkpoint_1 (for thread-2) — but they are on different
        # threads, so let's put two checkpoints on the same thread instead.
        chkpnt_2b = create_checkpoint(self.chkpnt_2, {}, 1)
        config_same_thread: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": self.chkpnt_1["id"],
            }
        }
        config_2b_result = self.memory_saver.put(
            config_same_thread,
            chkpnt_2b,
            {"source": "loop", "step": 2, "writes": {}},
            chkpnt_2b["channel_versions"],
        )

        # Without `before`, both checkpoints for thread-1 should appear
        all_results = list(
            self.memory_saver.list(
                {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
            )
        )
        assert len(all_results) == 2

        # With `before=config_2b_result`, only chkpnt_1 should be returned
        before_results = list(
            self.memory_saver.list(
                {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}},
                before=config_2b_result,
            )
        )
        assert len(before_results) == 1
        assert before_results[0].checkpoint["id"] == self.chkpnt_1["id"]

    def test_list_limit(self) -> None:
        """Test that `limit` parameter caps the number of returned checkpoints."""
        # Create 3 checkpoints on the same thread
        config_a: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-limit",
                "checkpoint_ns": "",
            }
        }
        chkpnt_a = empty_checkpoint()
        self.memory_saver.put(
            config_a,
            chkpnt_a,
            {"source": "input", "step": 0, "writes": {}},
            chkpnt_a["channel_versions"],
        )

        config_b: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-limit",
                "checkpoint_ns": "",
                "checkpoint_id": chkpnt_a["id"],
            }
        }
        chkpnt_b = create_checkpoint(chkpnt_a, {}, 1)
        self.memory_saver.put(
            config_b,
            chkpnt_b,
            {"source": "loop", "step": 1, "writes": {}},
            chkpnt_b["channel_versions"],
        )

        config_c: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-limit",
                "checkpoint_ns": "",
                "checkpoint_id": chkpnt_b["id"],
            }
        }
        chkpnt_c = create_checkpoint(chkpnt_b, {}, 1)
        self.memory_saver.put(
            config_c,
            chkpnt_c,
            {"source": "loop", "step": 2, "writes": {}},
            chkpnt_c["channel_versions"],
        )

        # All checkpoints
        all_results = list(
            self.memory_saver.list(
                {"configurable": {"thread_id": "thread-limit", "checkpoint_ns": ""}}
            )
        )
        assert len(all_results) == 3

        # Limit to 2
        limited_results = list(
            self.memory_saver.list(
                {"configurable": {"thread_id": "thread-limit", "checkpoint_ns": ""}},
                limit=2,
            )
        )
        assert len(limited_results) == 2

        # Limit to 1
        limited_results_1 = list(
            self.memory_saver.list(
                {"configurable": {"thread_id": "thread-limit", "checkpoint_ns": ""}},
                limit=1,
            )
        )
        assert len(limited_results_1) == 1

        # Limit to 0 returns nothing
        limited_results_0 = list(
            self.memory_saver.list(
                {"configurable": {"thread_id": "thread-limit", "checkpoint_ns": ""}},
                limit=0,
            )
        )
        assert len(limited_results_0) == 0

    def test_get_tuple_returns_none_for_missing_thread(self) -> None:
        """Test that get_tuple returns None when no checkpoint exists for a thread."""
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "nonexistent-thread",
                "checkpoint_ns": "",
            }
        }
        result = self.memory_saver.get_tuple(config)
        assert result is None

    def test_get_tuple_returns_none_for_missing_checkpoint_id(self) -> None:
        """Test that get_tuple returns None when checkpoint_id doesn't match."""
        self.memory_saver.put(
            self.config_1,
            self.chkpnt_1,
            self.metadata_1,
            self.chkpnt_1["channel_versions"],
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "nonexistent-checkpoint-id",
            }
        }
        result = self.memory_saver.get_tuple(config)
        assert result is None

    def test_get_tuple_returns_latest_without_checkpoint_id(self) -> None:
        """Test that get_tuple returns the latest checkpoint when no checkpoint_id is specified."""
        config_no_id: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-latest",
                "checkpoint_ns": "",
            }
        }
        chkpnt_a = empty_checkpoint()
        self.memory_saver.put(
            config_no_id,
            chkpnt_a,
            {"source": "input", "step": 0, "writes": {}},
            chkpnt_a["channel_versions"],
        )

        chkpnt_b = create_checkpoint(chkpnt_a, {}, 1)
        config_with_parent: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-latest",
                "checkpoint_ns": "",
                "checkpoint_id": chkpnt_a["id"],
            }
        }
        self.memory_saver.put(
            config_with_parent,
            chkpnt_b,
            {"source": "loop", "step": 1, "writes": {}},
            chkpnt_b["channel_versions"],
        )

        # Retrieve without checkpoint_id — should get the latest
        result = self.memory_saver.get_tuple(config_no_id)
        assert result is not None
        assert result.checkpoint["id"] == chkpnt_b["id"]
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["checkpoint_id"] == chkpnt_a["id"]

    def test_delete_thread(self) -> None:
        """Test that delete_thread removes all checkpoints and writes for a thread."""
        saved_config_1 = self.memory_saver.put(
            self.config_1,
            self.chkpnt_1,
            self.metadata_1,
            self.chkpnt_1["channel_versions"],
        )
        saved_config_2 = self.memory_saver.put(
            self.config_2,
            self.chkpnt_2,
            self.metadata_2,
            self.chkpnt_2["channel_versions"],
        )

        # Verify thread-1 checkpoint exists
        assert self.memory_saver.get_tuple(saved_config_1) is not None

        # Delete thread-1
        self.memory_saver.delete_thread("thread-1")

        # thread-1 should be gone
        assert self.memory_saver.get_tuple(saved_config_1) is None

        # thread-2 should still exist
        assert self.memory_saver.get_tuple(saved_config_2) is not None

    def test_delete_nonexistent_thread(self) -> None:
        """Test that deleting a thread that doesn't exist does not raise."""
        # Should not raise any exception
        self.memory_saver.delete_thread("ghost-thread")

    def test_put_writes_and_pending_writes(self) -> None:
        """Test that put_writes stores writes and they appear in pending_writes."""
        result_config = self.memory_saver.put(
            self.config_1,
            self.chkpnt_1,
            self.metadata_1,
            self.chkpnt_1["channel_versions"],
        )

        # Store some writes
        self.memory_saver.put_writes(
            result_config,
            [("channel_a", "value_1"), ("channel_b", "value_2")],
            task_id="task-1",
        )

        # Retrieve and check pending writes
        checkpoint_tuple = self.memory_saver.get_tuple(result_config)
        assert checkpoint_tuple is not None
        assert checkpoint_tuple.pending_writes is not None
        assert len(checkpoint_tuple.pending_writes) == 2

        channels = [pw[1] for pw in checkpoint_tuple.pending_writes]
        assert "channel_a" in channels
        assert "channel_b" in channels

    def test_put_writes_deduplication(self) -> None:
        """Test that duplicate writes with the same task_id and idx are not stored twice."""
        result_config = self.memory_saver.put(
            self.config_1,
            self.chkpnt_1,
            self.metadata_1,
            self.chkpnt_1["channel_versions"],
        )

        # Write twice with same task_id
        self.memory_saver.put_writes(
            result_config,
            [("channel_a", "value_1")],
            task_id="task-dup",
        )
        self.memory_saver.put_writes(
            result_config,
            [("channel_a", "value_1_updated")],
            task_id="task-dup",
        )

        checkpoint_tuple = self.memory_saver.get_tuple(result_config)
        assert checkpoint_tuple is not None
        assert checkpoint_tuple.pending_writes is not None
        # Should still be 1 write due to deduplication
        assert len(checkpoint_tuple.pending_writes) == 1

    def test_get_next_version(self) -> None:
        """Test that get_next_version produces monotonically increasing version strings."""
        v1 = self.memory_saver.get_next_version(None, None)
        # Format is "{counter:032}.{random:016}", counter starts at 1
        assert v1.split(".")[0].lstrip("0") == "1"

        v2 = self.memory_saver.get_next_version(v1, None)
        assert v2.split(".")[0].lstrip("0") == "2"

        # Ensure v2 > v1 lexicographically (they are zero-padded)
        assert v2 > v1

    def test_get_next_version_from_int(self) -> None:
        """Test get_next_version when current is an int (legacy format)."""
        v = self.memory_saver.get_next_version(5, None)  # type: ignore[arg-type]
        assert v.split(".")[0].lstrip("0") == "6"

    async def test_async_put_and_get(self) -> None:
        """Test async versions of put and get_tuple."""
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "async-thread",
                "checkpoint_ns": "",
            }
        }
        chkpnt = empty_checkpoint()
        result_config = await self.memory_saver.aput(
            config,
            chkpnt,
            {"source": "input", "step": 0, "writes": {}},
            chkpnt["channel_versions"],
        )
        result = await self.memory_saver.aget_tuple(result_config)
        assert result is not None
        assert result.checkpoint["id"] == chkpnt["id"]

    async def test_async_delete_thread(self) -> None:
        """Test async version of delete_thread."""
        saved_config = self.memory_saver.put(
            self.config_1,
            self.chkpnt_1,
            self.metadata_1,
            self.chkpnt_1["channel_versions"],
        )
        assert self.memory_saver.get_tuple(saved_config) is not None

        await self.memory_saver.adelete_thread("thread-1")
        assert self.memory_saver.get_tuple(saved_config) is None

    async def test_async_put_writes(self) -> None:
        """Test async version of put_writes."""
        result_config = self.memory_saver.put(
            self.config_1,
            self.chkpnt_1,
            self.metadata_1,
            self.chkpnt_1["channel_versions"],
        )
        await self.memory_saver.aput_writes(
            result_config,
            [("chan", "val")],
            task_id="async-task",
        )
        checkpoint_tuple = self.memory_saver.get_tuple(result_config)
        assert checkpoint_tuple is not None
        assert checkpoint_tuple.pending_writes is not None
        assert len(checkpoint_tuple.pending_writes) == 1

    async def test_asearch(self) -> None:
        # set up test
        # save checkpoints
        self.memory_saver.put(
            self.config_1,
            self.chkpnt_1,
            self.metadata_1,
            self.chkpnt_1["channel_versions"],
        )
        self.memory_saver.put(
            self.config_2,
            self.chkpnt_2,
            self.metadata_2,
            self.chkpnt_2["channel_versions"],
        )
        self.memory_saver.put(
            self.config_3,
            self.chkpnt_3,
            self.metadata_3,
            self.chkpnt_3["channel_versions"],
        )

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


async def test_memory_saver() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    memory_saver = InMemorySaver()
    assert isinstance(memory_saver, InMemorySaver)

    async with memory_saver as async_memory_saver:
        assert async_memory_saver is memory_saver

    with memory_saver as sync_memory_saver:
        assert sync_memory_saver is memory_saver


def test_memory_saver_warns_on_unregistered_msgpack(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = JsonPlusSerializer()
    memory_saver = InMemorySaver(serde=serde)
    obj = MemoryPydantic(foo="bar")

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"foo": obj}
    checkpoint["channel_versions"] = {"foo": 1}

    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }

    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    new_config = memory_saver.put(config, checkpoint, {}, {"foo": 1})
    result = memory_saver.get_tuple(new_config)

    assert result is not None
    assert "unregistered type" in caplog.text.lower()
    assert result.checkpoint["channel_values"]["foo"] == obj


def test_memory_saver_allowlist_silences_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("tests.test_memory", "MemoryPydantic")]
    )
    memory_saver = InMemorySaver(serde=serde)
    obj = MemoryPydantic(foo="bar")

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"foo": obj}
    checkpoint["channel_versions"] = {"foo": 1}

    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }

    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    new_config = memory_saver.put(config, checkpoint, {}, {"foo": 1})
    result = memory_saver.get_tuple(new_config)

    assert result is not None
    assert "unregistered type" not in caplog.text.lower()
    assert result.checkpoint["channel_values"]["foo"] == obj


def test_memory_saver_strict_blocks_unregistered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    memory_saver = InMemorySaver(serde=serde)
    obj = MemoryPydantic(foo="bar")

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"foo": obj}
    checkpoint["channel_versions"] = {"foo": 1}

    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }

    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    new_config = memory_saver.put(config, checkpoint, {}, {"foo": 1})
    result = memory_saver.get_tuple(new_config)

    assert result is not None
    assert "blocked" in caplog.text.lower()
    expected = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
    assert result.checkpoint["channel_values"]["foo"] == expected


def test_memory_saver_with_allowlist_proxy_isolated() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    memory_saver = InMemorySaver(serde=serde)
    proxy = memory_saver.with_allowlist([("tests.test_memory", "MemoryPydantic")])

    obj = MemoryPydantic(foo="bar")

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"foo": obj}
    checkpoint["channel_versions"] = {"foo": 1}

    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }

    new_config = proxy.put(config, checkpoint, {}, {"foo": 1})

    proxied = proxy.get_tuple(new_config)
    assert proxied is not None
    assert proxied.checkpoint["channel_values"]["foo"] == obj

    direct = memory_saver.get_tuple(new_config)
    assert direct is not None
    expected = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
    assert direct.checkpoint["channel_values"]["foo"] == expected
