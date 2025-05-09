import os
from typing import Any

import pytest
from bson.errors import InvalidDocument
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pymongo import MongoClient

from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint
from langgraph.checkpoint.mongodb import MongoDBSaver

# Setup:
# docker run --name mongodb -d -p 27017:27017 mongodb/mongodb-community-server
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "sync_checkpoints"


def test_search(input_data: dict[str, Any]) -> None:
    # Clear collections if they exist
    client: MongoClient = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    for clxn_name in db.list_collection_names():
        db.drop_collection(clxn_name)

    with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME, COLLECTION_NAME) as saver:
        # save checkpoints
        saver.put(
            input_data["config_1"],
            input_data["chkpnt_1"],
            input_data["metadata_1"],
            {},
        )
        saver.put(
            input_data["config_2"],
            input_data["chkpnt_2"],
            input_data["metadata_2"],
            {},
        )
        saver.put(
            input_data["config_3"],
            input_data["chkpnt_3"],
            input_data["metadata_3"],
            {},
        )

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
        assert search_results_1[0].metadata == input_data["metadata_1"]

        search_results_2 = list(saver.list(None, filter=query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == input_data["metadata_2"]

        search_results_3 = list(saver.list(None, filter=query_3))
        assert len(search_results_3) == 3

        search_results_4 = list(saver.list(None, filter=query_4))
        assert len(search_results_4) == 0

        # search by config (defaults to checkpoints across all namespaces)
        search_results_5 = list(saver.list({"configurable": {"thread_id": "thread-2"}}))
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}


def test_null_chars(input_data: dict[str, Any]) -> None:
    """In MongoDB string *values* can be any valid UTF-8 including nulls.
    *Field names*, however, cannot contain nulls characters."""
    with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME, COLLECTION_NAME) as saver:
        null_str = "\x00abc"  # string containing null character

        # 1. null string in field *value*
        null_value_cfg = saver.put(
            input_data["config_1"],
            input_data["chkpnt_1"],
            {"my_key": null_str},
            {},
        )
        assert saver.get_tuple(null_value_cfg).metadata["my_key"] == null_str  # type: ignore
        assert (
            list(saver.list(None, filter={"my_key": null_str}))[0].metadata["my_key"]  # type: ignore
            == null_str
        )

        # 2. null string in field *name*
        with pytest.raises(InvalidDocument):
            saver.put(
                input_data["config_1"],
                input_data["chkpnt_1"],
                {null_str: "my_value"},  # type: ignore
                {},
            )


def test_nested_filter() -> None:
    """Test one can filter on nested structure of non-trivial objects.

    This test highlights MongoDBSaver's _loads/(_dumps)_metadata methods,
    which enable MongoDB's ability to query nested documents,
    with the caveat that all keys are strings.

    We use a HumanMessage instance as found in the examples.
    The MQL query created is {metadata.writes.message: <serde dumped HumanMessage>}

    We also use the same message to check values in the Checkpoint.
    """

    input_message = HumanMessage(content="MongoDB is awesome!")
    clxn_name = "writes_message"
    thread_id = "thread-3"

    config = RunnableConfig(
        configurable=dict(thread_id=thread_id, checkpoint_id="1", checkpoint_ns="")
    )
    chkpt = empty_checkpoint()
    chkpt["channel_values"] = input_message

    metadata = CheckpointMetadata(
        source="loop", step=1, writes={"message": input_message}
    )

    with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME, clxn_name) as saver:
        saver.put(config, chkpt, metadata, {})

        results = list(saver.list(None, filter={"writes.message": input_message}))
        for cptpl in results:
            assert cptpl.metadata["writes"]["message"] == input_message
            break

        # Confirm serialization structure of data in collection
        doc: dict[str, Any] = saver.checkpoint_collection.find_one(
            {"thread_id": thread_id}
        )  # type: ignore
        assert isinstance(doc["checkpoint"], bytes)
        assert (
            isinstance(doc["metadata"], dict)
            and isinstance(doc["metadata"]["writes"], dict)
            and isinstance(doc["metadata"]["writes"]["message"], bytes)
        )

        # Test values of checkpoint
        # From checkpointer
        assert cptpl.checkpoint["channel_values"] == input_message
        # In database
        chkpt_db = saver.serde.loads_typed((doc["type"], doc["checkpoint"]))
        assert chkpt_db["channel_values"] == input_message

        # Drop collections
        saver.checkpoint_collection.drop()
        saver.writes_collection.drop()
