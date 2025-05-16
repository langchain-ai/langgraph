from typing import Any, cast

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
    def setup(self) -> None:
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

    def test_combined_metadata(self) -> None:
        with SqliteSaver.from_conn_string(":memory:") as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-2",
                    "checkpoint_ns": "",
                    "__super_private_key": "super_private_value",
                },
                "metadata": {"run_id": "my_run_id"},
            }
            saver.put(config, self.chkpnt_2, self.metadata_2, {})
            checkpoint = saver.get_tuple(config)
            assert checkpoint is not None
            assert checkpoint.metadata == {
                **self.metadata_2,
                "thread_id": "thread-2",
                "run_id": "my_run_id",
            }

    def test_search(self) -> None:
        with SqliteSaver.from_conn_string(":memory:") as saver:
            # set up test
            # save checkpoints
            saver.put(self.config_1, self.chkpnt_1, self.metadata_1, {})
            saver.put(self.config_2, self.chkpnt_2, self.metadata_2, {})
            saver.put(self.config_3, self.chkpnt_3, self.metadata_3, {})

            # call method / assertions
            query_1 = {"source": "input"}  # search by 1 key
            query_2 = {
                "step": 1,
                "writes": {"foo": "bar"},
            }  # search by multiple keys
            query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
            query_4 = {"source": "update", "step": 1}  # no match

            search_results_1 = list(saver.list(None, filter=query_1))
            assert len(search_results_1) == 1
            assert search_results_1[0].metadata == {
                "thread_id": "thread-1",
                "thread_ts": "1",
                **self.metadata_1,
            }

            search_results_2 = list(saver.list(None, filter=query_2))
            assert len(search_results_2) == 1
            assert search_results_2[0].metadata == {
                "thread_id": "thread-2",
                **self.metadata_2,
            }

            search_results_3 = list(saver.list(None, filter=query_3))
            assert len(search_results_3) == 3

            search_results_4 = list(saver.list(None, filter=query_4))
            assert len(search_results_4) == 0

            # search by config (defaults to checkpoints across all namespaces)
            search_results_5 = list(
                saver.list({"configurable": {"thread_id": "thread-2"}})
            )
            assert len(search_results_5) == 2
            assert {
                search_results_5[0].config["configurable"]["checkpoint_ns"],
                search_results_5[1].config["configurable"]["checkpoint_ns"],
            } == {"", "inner"}

            # TODO: test before and limit params

    def test_search_where(self) -> None:
        # call method / assertions
        expected_predicate_1 = "WHERE json_extract(CAST(metadata AS TEXT), '$.source') = ? AND json_extract(CAST(metadata AS TEXT), '$.step') = ? AND json_extract(CAST(metadata AS TEXT), '$.writes') = ? AND json_extract(CAST(metadata AS TEXT), '$.score') = ? AND checkpoint_id < ?"
        expected_param_values_1 = ["input", 2, "{}", 1, "1"]
        assert search_where(
            None, cast(dict[str, Any], self.metadata_1), self.config_1
        ) == (
            expected_predicate_1,
            expected_param_values_1,
        )

    def test_metadata_predicate(self) -> None:
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
        expected_predicate_3: list[str] = []

        expected_param_values_1 = ["input", 2, "{}", 1]
        expected_param_values_2 = ["loop", 1, '{"foo":"bar"}', None]
        expected_param_values_3: list[Any] = []

        assert _metadata_predicate(cast(dict[str, Any], self.metadata_1)) == (
            expected_predicate_1,
            expected_param_values_1,
        )
        assert _metadata_predicate(cast(dict[str, Any], self.metadata_2)) == (
            expected_predicate_2,
            expected_param_values_2,
        )
        assert _metadata_predicate(cast(dict[str, Any], self.metadata_3)) == (
            expected_predicate_3,
            expected_param_values_3,
        )

    async def test_informative_async_errors(self) -> None:
        with SqliteSaver.from_conn_string(":memory:") as saver:
            # call method / assertions
            with pytest.raises(NotImplementedError, match="AsyncSqliteSaver"):
                await saver.aget(self.config_1)
            with pytest.raises(NotImplementedError, match="AsyncSqliteSaver"):
                await saver.aget_tuple(self.config_1)
            with pytest.raises(NotImplementedError, match="AsyncSqliteSaver"):
                async for _ in saver.alist(self.config_1):
                    pass
