"""
Test suite for MongoDB integration with MongoDBSaver class in langgraph.checkpoint module.
"""

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from pymongo import MongoClient

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.mongodb import MongoDBSaver


class TestMongoDBSaver:
    """Test class for testing MongoDBSaver functionality including the `put` and `list` methods for storing and searching checkpoints."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """
        Set up the test environment, including MongoDB connection and test data.
        Initializes configurations, checkpoints, and metadata for the tests.
        """
        # MongoDB setup for test
        self.db_name = "test_db"
        self.collection_name = "checkpoints"
        self.conn_string = "mongodb://localhost:27017"

        self.client = MongoClient(self.conn_string)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.collection.delete_many({})  # Clear collection before each test

        self.config_1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
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

    @pytest.fixture
    def mongo_saver(self):
        """
        Fixture for creating an instance of MongoDBSaver.
        """
        with MongoDBSaver.from_conn_info(self.conn_string, self.db_name) as saver:
            yield saver

    def test_asearch(self, mongo_saver: MongoDBSaver) -> None:
        """
        Test the `list` and `put` methods of MongoDBSaver for storing and searching checkpoints.
        Verifies the correctness of search queries, including different filters and configurations.
        """
        mongo_saver.put(self.config_1, self.chkpnt_1, self.metadata_1, {})
        mongo_saver.put(self.config_2, self.chkpnt_2, self.metadata_2, {})
        mongo_saver.put(self.config_3, self.chkpnt_3, self.metadata_3, {})

        # Assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
        query_4 = {"source": "update", "step": 1}  # no match

        search_results_1 = list(mongo_saver.list(filter=query_1))
