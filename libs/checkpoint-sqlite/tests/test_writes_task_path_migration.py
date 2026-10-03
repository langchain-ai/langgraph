import sqlite3
from pathlib import Path

import aiosqlite
import pytest
from langgraph.checkpoint.base import empty_checkpoint

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

WRITES_BEFORE_TASK_PATH = """
CREATE TABLE writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
INSERT INTO writes VALUES ('t', '', 'c', 'old-task', 0, 'ch', 'null', X'');
"""


@pytest.fixture
def legacy_db(tmp_path: Path) -> Path:
    db = tmp_path / "legacy.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(WRITES_BEFORE_TASK_PATH)
    return db


def test_setup_migrates_legacy_writes_table_repeatably(legacy_db: Path) -> None:
    for _ in range(2):
        with SqliteSaver.from_conn_string(str(legacy_db)) as saver:
            saver.setup()
            rows = saver.conn.execute(
                "SELECT task_id, task_path FROM writes"
            ).fetchall()
            assert rows == [("old-task", "")]


@pytest.mark.parametrize("fresh", [True, False], ids=["fresh", "legacy"])
def test_put_writes_persists_task_path(
    tmp_path: Path, legacy_db: Path, fresh: bool
) -> None:
    db = tmp_path / "fresh.sqlite" if fresh else legacy_db
    with SqliteSaver.from_conn_string(str(db)) as saver:
        config = saver.put(
            {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
            empty_checkpoint(),
            {},
            {},
        )
        saver.put_writes(config, [("ch", "v")], "task-1", "~__pregel_pull, node")
        stored = saver.conn.execute(
            "SELECT task_path FROM writes WHERE task_id = 'task-1'"
        ).fetchall()
    assert stored == [("~__pregel_pull, node",)]


async def test_async_setup_migrates_legacy_writes_table_repeatably(
    legacy_db: Path,
) -> None:
    for _ in range(2):
        async with AsyncSqliteSaver.from_conn_string(str(legacy_db)) as saver:
            await saver.setup()
            config = await saver.aput(
                {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
                empty_checkpoint(),
                {},
                {},
            )
            await saver.aput_writes(
                config, [("ch", "v")], "task-1", "~__pregel_pull, node"
            )

    async with aiosqlite.connect(legacy_db) as conn:
        async with conn.execute(
            "SELECT DISTINCT task_id, task_path FROM writes ORDER BY task_id"
        ) as cur:
            assert await cur.fetchall() == [
                ("old-task", ""),
                ("task-1", "~__pregel_pull, node"),
            ]


def _legacy_database_with_history(db: Path) -> dict:
    root = empty_checkpoint()
    root["channel_values"] = {"ch": "seed"}
    root["channel_versions"] = {"ch": 1}
    with SqliteSaver.from_conn_string(str(db)) as saver:
        root_config = saver.put(
            {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
            root,
            {},
            {"ch": 1},
        )
        saver.put_writes(root_config, [("ch", "write")], "task", "~__pregel_pull, n")
        child = saver.put(root_config, empty_checkpoint(), {}, {})
        saver.conn.execute("ALTER TABLE writes DROP COLUMN task_path")
        saver.conn.commit()
    return child


def test_read_only_legacy_database_still_reads_delta_history(tmp_path: Path) -> None:
    db = tmp_path / "legacy.sqlite"
    child = _legacy_database_with_history(db)

    saver = SqliteSaver(sqlite3.connect(f"file:{db}?mode=ro", uri=True))
    got = saver.get_delta_channel_history(config=child, channels=["ch"])

    assert got["ch"] == {"seed": "seed", "writes": [("task", "ch", "write")]}


async def test_async_read_only_legacy_database_still_reads_delta_history(
    tmp_path: Path,
) -> None:
    db = tmp_path / "legacy.sqlite"
    child = _legacy_database_with_history(db)

    async with aiosqlite.connect(f"file:{db}?mode=ro", uri=True) as conn:
        saver = AsyncSqliteSaver(conn)
        got = await saver.aget_delta_channel_history(config=child, channels=["ch"])

    assert got["ch"] == {"seed": "seed", "writes": [("task", "ch", "write")]}
