import os
from typing import Any

import pytest
from bson.errors import InvalidDocument
from motor.motor_asyncio import AsyncIOMotorClient

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "sync_checkpoints_aio"


async def test_asearch(input_data: dict[str, Any]) -> None:
    # Clear collections if they exist
    client: AsyncIOMotorClient = AsyncIOMotorClient(MONGODB_URI)
    db = client[DB_NAME]

    for clxn in await db.list_collection_names():
        await db.drop_collection(clxn)

    async with AsyncMongoDBSaver.from_conn_string(
        MONGODB_URI, DB_NAME, COLLECTION_NAME
    ) as saver:
        # save checkpoints
        await saver.aput(
            input_data["config_1"],
            input_data["chkpnt_1"],
            input_data["metadata_1"],
            {},
        )
        await saver.aput(
            input_data["config_2"],
            input_data["chkpnt_2"],
            input_data["metadata_2"],
            {},
        )
        await saver.aput(
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

        search_results_1 = [c async for c in saver.alist(None, filter=query_1)]
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == input_data["metadata_1"]

        search_results_2 = [c async for c in saver.alist(None, filter=query_2)]
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == input_data["metadata_2"]

        search_results_3 = [c async for c in saver.alist(None, filter=query_3)]
        assert len(search_results_3) == 3

        search_results_4 = [c async for c in saver.alist(None, filter=query_4)]
        assert len(search_results_4) == 0

        # search by config (defaults to checkpoints across all namespaces)
        search_results_5 = [
            c async for c in saver.alist({"configurable": {"thread_id": "thread-2"}})
        ]
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}


async def test_null_chars(input_data: dict[str, Any]) -> None:
    """In MongoDB string *values* can be any valid UTF-8 including nulls.
    *Field names*, however, cannot contain nulls characters."""
    async with AsyncMongoDBSaver.from_conn_string(
        MONGODB_URI, DB_NAME, COLLECTION_NAME
    ) as saver:
        null_str = "\x00abc"  # string containing null character

        # 1. null string in field *value*
        null_value_cfg = await saver.aput(
            input_data["config_1"],
            input_data["chkpnt_1"],
            {"my_key": null_str},
            {},
        )
        null_tuple = await saver.aget_tuple(null_value_cfg)
        assert null_tuple.metadata["my_key"] == null_str  # type: ignore
        cps = [c async for c in saver.alist(None, filter={"my_key": null_str})]
        assert cps[0].metadata["my_key"] == null_str

        # 2. null string in field *name*
        with pytest.raises(InvalidDocument):
            await saver.aput(
                input_data["config_1"],
                input_data["chkpnt_1"],
                {null_str: "my_value"},  # type: ignore
                {},
            )
