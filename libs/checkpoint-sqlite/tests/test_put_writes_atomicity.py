import sqlite3

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

FAIL_ON_CHANNEL = """
CREATE TRIGGER fail_boom BEFORE INSERT ON writes
WHEN NEW.channel = 'boom'
BEGIN
    SELECT RAISE(ABORT, 'boom');
END;
"""


def _config(config: dict) -> dict:
    return {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}


def test_put_writes_is_all_or_nothing(tmp_path) -> None:
    saver = SqliteSaver(sqlite3.connect(tmp_path / "db.sqlite", check_same_thread=False))
    saver.setup()
    config = saver.put(_config({}), empty_checkpoint(), {}, {})
    saver.put_writes(config, [("seed", "value")], "seed-task")
    saver.conn.execute(FAIL_ON_CHANNEL)

    with pytest.raises(sqlite3.DatabaseError):
        saver.put_writes(config, [("first", "a"), ("boom", "b")], "batch-task")

    assert saver.conn.execute("SELECT channel FROM writes").fetchall() == [("seed",)]


async def test_async_put_writes_is_all_or_nothing(tmp_path) -> None:
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "db.sqlite")) as saver:
        config = await saver.aput(_config({}), empty_checkpoint(), {}, {})
        await saver.aput_writes(config, [("seed", "value")], "seed-task")
        await saver.conn.execute(FAIL_ON_CHANNEL)

        with pytest.raises(sqlite3.DatabaseError):
            await saver.aput_writes(
                config, [("first", "a"), ("boom", "b")], "batch-task"
            )

        async with saver.conn.execute("SELECT channel FROM writes") as cursor:
            rows = await cursor.fetchall()
        assert rows == [("seed",)]
