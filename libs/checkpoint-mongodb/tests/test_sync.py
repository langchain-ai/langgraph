from typing import Any

import pytest  # type: ignore
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.mongodb import MongoDBSaver

MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "langchain-test"


@pytest.fixture(scope="session")
def setup() -> dict:
    """Setup and store conveniently in a single dictionary."""
    test_inputs = {}
    test_inputs["config_1"] : RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            # for backwards compatibility testing
            "thread_ts": "1",
            "checkpoint_ns": "",
        }
    }
    test_inputs["config_2"]: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    test_inputs["config_3"]: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2-inner",
            "checkpoint_ns": "inner",
        }
    }

    test_inputs["chkpnt_1"]: Checkpoint = empty_checkpoint()
    test_inputs["chkpnt_2"]: Checkpoint = create_checkpoint(test_inputs["chkpnt_1"], {}, 1)
    test_inputs["chkpnt_3"]: Checkpoint = empty_checkpoint()

    test_inputs["metadata_1"]: CheckpointMetadata = {
        "source": "input",
        "step": 2,
        "writes": {},
        "score": 1,
    }
    test_inputs["metadata_2"]: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "writes": {"foo": "bar"},
        "score": None,
    }
    test_inputs["metadata_3"]: CheckpointMetadata = dict()
    with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME) as saver:
        saver.setup()

    def test_search(self) -> None:
        with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME) as saver:
            # save checkpoints
            saver.put(test_inputs["config_1"], test_inputs["chkpnt_1"], test_inputs["metadata_1"], {})
            saver.put(test_inputs["config_2"], test_inputs["chkpnt_2"], test_inputs["metadata_2"], {})
            saver.put(test_inputs["config_3"], test_inputs["chkpnt_3"], test_inputs["metadata_3"], {})

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
            assert search_results_1[0].metadata == test_inputs["metadata_1"]

            search_results_2 = list(saver.list(None, filter=query_2))
            assert len(search_results_2) == 1
            assert search_results_2[0].metadata == test_inputs["metadata_2"]

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

    def test_null_chars(self) -> None:
        with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME) as saver:
            config = saver.put(test_inputs["config_1"], test_inputs["chkpnt_1"], {"my_key": "\x00abc"}, {})
            assert saver.get_tuple(config).metadata["my_key"] == "abc"  # type: ignore
            assert (
                list(saver.list(None, filter={"my_key": "abc"}))[0].metadata["my_key"]  # type: ignore
                == "abc"
            )
