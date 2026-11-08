import sqlite3
from collections.abc import Iterator
from contextlib import closing
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


@pytest.fixture
def busy_db(tmp_path: Path) -> Iterator[Path]:
    db = tmp_path / "busy.sqlite"
    with SqliteSaver.from_conn_string(str(db)) as saver:
        saver.setup()
    with closing(sqlite3.connect(db, isolation_level=None)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        yield db
        writer.execute("ROLLBACK")


def test_setup_does_not_wait_on_another_writer(busy_db: Path) -> None:
    with closing(sqlite3.connect(busy_db, timeout=0)) as conn:
        SqliteSaver(conn).setup()


async def test_async_setup_does_not_wait_on_another_writer(busy_db: Path) -> None:
    async with aiosqlite.connect(busy_db, timeout=0) as conn:
        await AsyncSqliteSaver(conn).setup()


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


class _DriverOperationalError(Exception):
    """Stands in for a driver error that shares the name but not the class.

    sqlcipher3.dbapi2 raises `sqlcipher3.dbapi2.OperationalError`, which is not a
    subclass of `sqlite3.OperationalError`, so a guard written against the
    std-lib class never fires.
    """


class _DriverConnection:
    """A connection whose errors are raised by the driver, not by sqlite3."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)

    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
        try:
            return self._conn.execute(sql, *args)
        except sqlite3.OperationalError as exc:
            raise _DriverOperationalError(str(exc)) from None

    def executescript(self, sql: str) -> sqlite3.Cursor:
        try:
            return self._conn.executescript(sql)
        except sqlite3.OperationalError as exc:
            raise _DriverOperationalError(str(exc)) from None


class _AsyncDriverConnection:
    """The aiosqlite connection, with the same substitution of the error class."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)

    async def execute(self, sql: str, *args: object) -> aiosqlite.Cursor:
        try:
            return await self._conn.execute(sql, *args)
        except aiosqlite.OperationalError as exc:
            raise _DriverOperationalError(str(exc)) from None

    def executescript(self, sql: str) -> object:
        return self._conn.executescript(sql)


def test_setup_on_a_fresh_database_with_a_driver_specific_error() -> None:
    """setup() must not depend on which module raised the error.

    `CREATE TABLE` already declares `task_path`, so the migration's `ALTER TABLE`
    raises "duplicate column name". The guard around it names the std-lib class,
    so a driver that raises its own OperationalError leaked the exception and
    every fresh database failed to set up.
    """
    saver = SqliteSaver(_DriverConnection(sqlite3.connect(":memory:")))

    saver.setup()

    assert saver.is_setup
    assert saver._has_task_path is True
    assert "task_path" in [row[1] for row in saver.conn.execute("PRAGMA table_info(writes)")]


async def test_async_setup_on_a_fresh_database_with_a_driver_specific_error(
    tmp_path: Path,
) -> None:
    conn = await aiosqlite.connect(tmp_path / "fresh.sqlite")
    try:
        saver = AsyncSqliteSaver(_AsyncDriverConnection(conn))

        await saver.setup()

        assert saver.is_setup
        assert "task_path" in [
            row[1] for row in await (await conn.execute("PRAGMA table_info(writes)")).fetchall()
        ]
    finally:
        await conn.close()


def test_setup_migrates_a_legacy_table_with_a_driver_specific_error(
    legacy_db: Path,
) -> None:
    saver = SqliteSaver(_DriverConnection(sqlite3.connect(str(legacy_db))))

    saver.setup()

    assert saver.conn.execute("SELECT task_id, task_path FROM writes").fetchall() == [
        ("old-task", "")
    ]


def test_setup_still_raises_an_unrelated_driver_error(legacy_db: Path) -> None:
    """Only the migration's own errors are tolerated."""

    class _Broken(_DriverConnection):
        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
            if sql.lstrip().upper().startswith("ALTER TABLE"):
                raise _DriverOperationalError("disk I/O error")
            return super().execute(sql, *args)

    saver = SqliteSaver(_Broken(sqlite3.connect(str(legacy_db))))

    with pytest.raises(_DriverOperationalError, match="disk I/O error"):
        saver.setup()
