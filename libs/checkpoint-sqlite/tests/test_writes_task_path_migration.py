"""Tests for the additive `writes.task_path` migration (#8382).

`task_path` records the path of the task that produced a write, so delta
channel replay can restore the order `apply_writes` applied a super-step's
writes in. The sqlite savers previously accepted `task_path` on `put_writes`
and dropped it, so the column has to be added to databases created by earlier
versions as well as to fresh ones.

Sqlite has no `ADD COLUMN IF NOT EXISTS`, so `setup()` probes
`pragma_table_info` before issuing the `ALTER` — these tests pin that the
probe makes the migration both effective and repeatable.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest
from langgraph.checkpoint.base import empty_checkpoint

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite._schema import ADD_WRITES_TASK_PATH_SQL
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# The `writes` table as created before `task_path` existed.
LEGACY_SCHEMA = """
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
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
"""


def _write_legacy_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(LEGACY_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_fresh_database_has_task_path(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "fresh.sqlite")) as saver:
        saver.setup()
        assert "task_path" in _columns(saver.conn, "writes")


def test_legacy_database_gains_task_path(tmp_path: Path) -> None:
    db = tmp_path / "legacy.sqlite"
    _write_legacy_db(db)

    with SqliteSaver.from_conn_string(str(db)) as saver:
        saver.setup()
        columns = _columns(saver.conn, "writes")

    assert "task_path" in columns
    # Existing columns are untouched — this is additive, not a table rebuild.
    assert columns[:8] == [
        "thread_id",
        "checkpoint_ns",
        "checkpoint_id",
        "task_id",
        "idx",
        "channel",
        "type",
        "value",
    ]


def test_setup_is_repeatable_on_migrated_database(tmp_path: Path) -> None:
    """A second `setup()` must not re-issue the `ALTER`.

    Sqlite raises `duplicate column name` rather than ignoring it, so an
    unguarded `ALTER` would break every reopen of a migrated database.
    """
    db = tmp_path / "legacy.sqlite"
    _write_legacy_db(db)

    with SqliteSaver.from_conn_string(str(db)) as saver:
        saver.setup()
        saver.is_setup = False
        saver.setup()
        assert "task_path" in _columns(saver.conn, "writes")

    # And again through a fresh connection to the migrated file.
    with SqliteSaver.from_conn_string(str(db)) as saver:
        saver.setup()
        assert "task_path" in _columns(saver.conn, "writes")


def test_legacy_rows_keep_default_and_sort_first(tmp_path: Path) -> None:
    """Rows predating the column read back as `''` and order ahead of paths.

    `''` precedes every `task_path_str` output, which puts pre-migration
    writes before path-carrying ones within their checkpoint instead of
    interleaving them under a rule that never applied to them.
    """
    db = tmp_path / "legacy.sqlite"
    _write_legacy_db(db)

    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id,"
            " idx, channel, type, value) VALUES ('t', '', 'c', 'task', 0, 'ch',"
            " 'null', X'')"
        )
        conn.commit()
    finally:
        conn.close()

    with SqliteSaver.from_conn_string(str(db)) as saver:
        saver.setup()
        stored = saver.conn.execute("SELECT task_path FROM writes").fetchall()
        assert stored == [("",)]

        ordered = saver.conn.execute(
            "SELECT task_path FROM writes ORDER BY task_path, task_id, idx"
        ).fetchall()
        assert ordered[0] == ("",)


def test_setup_survives_losing_the_migration_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`setup()` succeeds when another connection migrates first.

    The `pragma_table_info` probe is not a lock. Two connections opening the
    same file can both see the column missing, and whichever issues the `ALTER`
    second gets `duplicate column name` — `ALTER TABLE ADD COLUMN` has no
    `IF NOT EXISTS` form to fall back on, unlike the `CREATE TABLE`s above it.

    Stubbing the probe to always report the column missing reproduces exactly
    the losing interleaving (probe says absent, another connection adds it,
    then we `ALTER`) without depending on thread timing.
    """
    db = tmp_path / "legacy.sqlite"
    _write_legacy_db(db)

    # Winner of the race: migrates the file out from under the saver below.
    winner = sqlite3.connect(db)
    try:
        winner.execute(ADD_WRITES_TASK_PATH_SQL)
        winner.commit()
    finally:
        winner.close()

    monkeypatch.setattr(
        "langgraph.checkpoint.sqlite.HAS_WRITES_TASK_PATH_SQL", "SELECT 1 WHERE 0"
    )
    with SqliteSaver.from_conn_string(str(db)) as loser:
        loser.setup()
        assert "task_path" in _columns(loser.conn, "writes")


def test_put_writes_persists_task_path(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "fresh.sqlite")) as saver:
        config = saver.put(
            {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
            empty_checkpoint(),
            {},
            {},
        )
        saver.put_writes(config, [("ch", "v")], "task-1", "~__pregel_pull, node")

        stored = saver.conn.execute("SELECT task_id, task_path FROM writes").fetchall()
        assert stored == [("task-1", "~__pregel_pull, node")]


@pytest.mark.asyncio
async def test_async_saver_migrates_and_persists_task_path(tmp_path: Path) -> None:
    db = tmp_path / "legacy.sqlite"
    _write_legacy_db(db)

    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        await saver.setup()
        config = await saver.aput(
            {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
            empty_checkpoint(),
            {},
            {},
        )
        await saver.aput_writes(config, [("ch", "v")], "task-1", "~__pregel_pull, node")
        # Idempotent for the async saver too.
        saver.is_setup = False
        await saver.setup()

    async with aiosqlite.connect(db) as conn:
        async with conn.execute("SELECT task_id, task_path FROM writes") as cur:
            assert await cur.fetchall() == [("task-1", "~__pregel_pull, node")]
