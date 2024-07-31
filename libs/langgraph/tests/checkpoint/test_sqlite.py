import pytest
from langchain_core.runnables import RunnableConfig
from langgraph_checkpoint.base import Checkpoint, CheckpointMetadata, empty_checkpoint

from langgraph.channels.manager import create_checkpoint
from langgraph.checkpoint.sqlite import (
    _AIO_ERROR_MSG,
    SqliteSaver,
    _metadata_predicate,
    search_where,
)


class TestSqliteSaver:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.sqlite_saver = SqliteSaver.from_conn_string(":memory:")

        # objects for test setup
        self.config_1: RunnableConfig = {
            "configurable": {"thread_id": "thread-1", "checkpoint_id": "1"}
        }
        self.config_2: RunnableConfig = {
            "configurable": {"thread_id": "thread-2", "checkpoint_id": "2"}
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
        self.metadata_3: CheckpointMetadata = {}

    def test_search(self):
        # set up test
        # save checkpoints
        self.sqlite_saver.put(self.config_1, self.chkpnt_1, self.metadata_1)
        self.sqlite_saver.put(self.config_2, self.chkpnt_2, self.metadata_2)

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
        assert len(search_results_3) == 2

        search_results_4 = list(self.sqlite_saver.list(None, filter=query_4))
        assert len(search_results_4) == 0

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
        with pytest.raises(NotImplementedError, match=_AIO_ERROR_MSG):
            await self.sqlite_saver.aget(self.config_1)
        with pytest.raises(NotImplementedError, match=_AIO_ERROR_MSG):
            await self.sqlite_saver.aget_tuple(self.config_1)
        with pytest.raises(NotImplementedError, match=_AIO_ERROR_MSG):
            async for _ in self.sqlite_saver.alist(self.config_1):
                pass
