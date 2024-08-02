import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.utils import _metadata_predicate, search_where


class TestSqliteSaver:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.sqlite_saver = SqliteSaver.from_conn_string(":memory:")

        # objects for test setup
        self.config_1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                # for backwards compatibility testing
                "thread_ts": "1",
                "checkpoint_ns": "",
            }
        }
        self.config_2: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2",
                "checkpoint_ns": "",
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

    def test_search(self):
        # set up test
        # save checkpoints
        self.sqlite_saver.put(self.config_1, self.chkpnt_1, self.metadata_1)
        self.sqlite_saver.put(self.config_2, self.chkpnt_2, self.metadata_2)
        self.sqlite_saver.put(self.config_3, self.chkpnt_3, self.metadata_3)

        # call method / assertions
        query_1: CheckpointMetadata = {"source": "input"}  # search by 1 key
        query_2: CheckpointMetadata = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: CheckpointMetadata = {}  # search by no keys, return all checkpoints
        query_4: CheckpointMetadata = {"source": "update", "step": 1}  # no match

        search_results_1 = list(self.sqlite_saver.list(None, filter=query_1))
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == self.metadata_1

        search_results_2 = list(self.sqlite_saver.list(None, filter=query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == self.metadata_2

        search_results_3 = list(self.sqlite_saver.list(None, filter=query_3))
        assert len(search_results_3) == 3

        search_results_4 = list(self.sqlite_saver.list(None, filter=query_4))
        assert len(search_results_4) == 0

        # search by config (defaults to root graph checkpoints)
        search_results_5 = list(
            self.sqlite_saver.list({"configurable": {"thread_id": "thread-2"}})
        )
        assert len(search_results_5) == 1
        assert search_results_5[0].config["configurable"]["checkpoint_ns"] == ""

        # search by config and checkpoint_ns
        search_results_6 = list(
            self.sqlite_saver.list(
                {"configurable": {"thread_id": "thread-2", "checkpoint_ns": "inner"}}
            )
        )
        assert len(search_results_6) == 1
        assert search_results_6[0].config["configurable"]["checkpoint_ns"] == "inner"

        # TODO: test before and limit params

    def test_search_where(self):
        # call method / assertions
        expected_predicate_1 = "WHERE json_extract(CAST(metadata AS TEXT), '$.source') = ? AND json_extract(CAST(metadata AS TEXT), '$.step') = ? AND json_extract(CAST(metadata AS TEXT), '$.writes') = ? AND json_extract(CAST(metadata AS TEXT), '$.score') = ? AND checkpoint_id < ?"
        expected_param_values_1 = ["input", 2, "{}", 1, "1"]
        assert search_where(None, self.metadata_1, self.config_1) == (
            expected_predicate_1,
            expected_param_values_1,
        )

    def test_metadata_predicate(self):
        # call method / assertions
        expected_predicate_1 = [
            "json_extract(CAST(metadata AS TEXT), '$.source') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.step') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.writes') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.score') = ?",
        ]
        expected_predicate_2 = [
            "json_extract(CAST(metadata AS TEXT), '$.source') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.step') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.writes') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.score') IS ?",
        ]
        expected_predicate_3 = []

        expected_param_values_1 = ["input", 2, "{}", 1]
        expected_param_values_2 = ["loop", 1, '{"foo":"bar"}', None]
        expected_param_values_3 = []

        assert _metadata_predicate(self.metadata_1) == (
            expected_predicate_1,
            expected_param_values_1,
        )
        assert _metadata_predicate(self.metadata_2) == (
            expected_predicate_2,
            expected_param_values_2,
        )
        assert _metadata_predicate(self.metadata_3) == (
            expected_predicate_3,
            expected_param_values_3,
        )

    async def test_informative_async_errors(self):
        # call method / assertions
        with pytest.raises(NotImplementedError, match="AsyncSqliteSaver"):
            await self.sqlite_saver.aget(self.config_1)
        with pytest.raises(NotImplementedError, match="AsyncSqliteSaver"):
            await self.sqlite_saver.aget_tuple(self.config_1)
        with pytest.raises(NotImplementedError, match="AsyncSqliteSaver"):
            async for _ in self.sqlite_saver.alist(self.config_1):
                pass
