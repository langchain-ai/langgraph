import pytest

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.checkpoint.sqlite import SqliteSaver


class TestMemorySaver:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.sqlite_saver = SqliteSaver.from_conn_string(":memory:")

        # objects for test setup
        self.config_1: RunnableConfig = {"configurable": {"thread_id": "thread-1", "thread_ts": "1"}}
        self.config_2: RunnableConfig = {"configurable": {"thread_id": "thread-2", "thread_ts": "2"}}

        self.chkpnt_1: Checkpoint = {
            "v": 1,
            "ts": "1",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {}
        }
        self.chkpnt_2: Checkpoint = {
            "v": 2,
            "ts": "2",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {}
        }

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

    def test_search(self):
        # set up test
        # save checkpoints
        self.sqlite_saver.put(self.config_1, self.chkpnt_1, self.metadata_1)
        self.sqlite_saver.put(self.config_2, self.chkpnt_2, self.metadata_2)

        # call method / assertions
        query_1: CheckpointMetadata = {"source": "input"}  # search by 1 key
        query_2: CheckpointMetadata = {"step": 1, "writes": {"foo": "bar"}}  # search by multiple keys
        query_3: CheckpointMetadata = {}  # search by no keys, return all checkpoints
        query_4: CheckpointMetadata = {"source": "update", "step": 1}  # no match

        search_results_1 = list(self.sqlite_saver.search(query_1))
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == self.metadata_1

        search_results_2 = list(self.sqlite_saver.search(query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == self.metadata_2

        search_results_3 = list(self.sqlite_saver.search(query_3))
        assert len(search_results_3) == 2

        search_results_4 = list(self.sqlite_saver.search(query_4))
        assert len(search_results_4) == 0

        # TODO: test before and limit params

    def test_create_where(self):
        # call method / assertions
        expected_where = "WHERE json_extract(CAST(metadata AS TEXT), '$.source') = 'loop' AND json_extract(CAST(metadata AS TEXT), '$.step') = 1 AND json_extract(CAST(metadata AS TEXT), '$.writes') = '{\"foo\":\"bar\"}' AND json_extract(CAST(metadata AS TEXT), '$.score') IS NULL "
        assert self.sqlite_saver.search_where(self.metadata_2) == expected_where
