from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.postgres import PostgresSaver

# objects for test setup
config_1: RunnableConfig = {
    "configurable": {
        "thread_id": "thread-1",
        # for backwards compatibility testing
        "thread_ts": "1",
        "checkpoint_ns": "",
    }
}
config_2: RunnableConfig = {
    "configurable": {
        "thread_id": "thread-2",
        "checkpoint_id": "2",
        "checkpoint_ns": "",
    }
}
config_3: RunnableConfig = {
    "configurable": {
        "thread_id": "thread-2",
        "checkpoint_id": "2-inner",
        "checkpoint_ns": "inner",
    }
}

chkpnt_1: Checkpoint = empty_checkpoint()
chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
chkpnt_3: Checkpoint = empty_checkpoint()

metadata_1: CheckpointMetadata = {
    "source": "input",
    "step": 2,
    "writes": {},
    "score": 1,
}
metadata_2: CheckpointMetadata = {
    "source": "loop",
    "step": 1,
    "writes": {"foo": "bar"},
    "score": None,
}
metadata_3: CheckpointMetadata = {}


async def test_asearch(conn):
    saver = PostgresSaver(conn)
    # set up test
    # save checkpoints
    await saver.aput(config_1, chkpnt_1, metadata_1)
    await saver.aput(config_2, chkpnt_2, metadata_2)
    await saver.aput(config_3, chkpnt_3, metadata_3)

    # call method / assertions
    query_1: CheckpointMetadata = {"source": "input"}  # search by 1 key
    query_2: CheckpointMetadata = {
        "step": 1,
        "writes": {"foo": "bar"},
    }  # search by multiple keys
    query_3: CheckpointMetadata = {}  # search by no keys, return all checkpoints
    query_4: CheckpointMetadata = {"source": "update", "step": 1}  # no match

    search_results_1 = [c async for c in saver.alist(None, filter=query_1)]
    assert len(search_results_1) == 1
    assert search_results_1[0].metadata == metadata_1

    search_results_2 = [c async for c in saver.alist(None, filter=query_2)]
    assert len(search_results_2) == 1
    assert search_results_2[0].metadata == metadata_2

    search_results_3 = [c async for c in saver.alist(None, filter=query_3)]
    assert len(search_results_3) == 3

    search_results_4 = [c async for c in saver.alist(None, filter=query_4)]
    assert len(search_results_4) == 0

    # search by config (defaults to root graph checkpoints)
    search_results_5 = [
        c async for c in saver.alist({"configurable": {"thread_id": "thread-2"}})
    ]
    assert len(search_results_5) == 1
    assert search_results_5[0].config["configurable"]["checkpoint_ns"] == ""

    # search by config and checkpoint_ns
    search_results_6 = [
        c
        async for c in saver.alist(
            {
                "configurable": {
                    "thread_id": "thread-2",
                    "checkpoint_ns": "inner",
                }
            }
        )
    ]
    assert len(search_results_6) == 1
    assert search_results_6[0].config["configurable"]["checkpoint_ns"] == "inner"

    # TODO: test before and limit params
