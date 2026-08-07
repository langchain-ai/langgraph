"""Additive schema migrations shared by the sqlite savers.

`SqliteSaver.setup` and `AsyncSqliteSaver.setup` create their tables with
`CREATE TABLE IF NOT EXISTS`, which leaves a database created by an earlier
version on the earlier schema. Sqlite has no `ADD COLUMN IF NOT EXISTS`
(the postgres savers rely on that form), and re-running a plain
`ALTER TABLE ... ADD COLUMN` raises `OperationalError: duplicate column
name`. So each migration pairs an `ALTER` with a probe against
`pragma_table_info` that tells us whether this database still needs it.

Databases created fresh already carry every column from the `CREATE TABLE`
statements, so the probe finds the column and the `ALTER` never runs.
"""

from __future__ import annotations

# `writes.task_path` records the path of the task that produced a write.
# Delta channel replay orders a checkpoint's writes by
# (task_path, task_id, idx) to reproduce the order `apply_writes` applied
# them in live; without the column, replay can only order by
# (task_id, idx), which permutes writes made by parallel tasks in the same
# super-step. Rows written before this migration keep the `''` default and
# so sort ahead of path-carrying rows within their checkpoint.
HAS_WRITES_TASK_PATH_SQL = (
    "SELECT 1 FROM pragma_table_info('writes') WHERE name = 'task_path'"
)

ADD_WRITES_TASK_PATH_SQL = (
    "ALTER TABLE writes ADD COLUMN task_path TEXT NOT NULL DEFAULT ''"
)

# Substring of the `OperationalError` sqlite raises when the column is already
# there. The probe above is not enough on its own: two connections opening the
# same file can both pass it and both issue the `ALTER`, and unlike
# `CREATE TABLE IF NOT EXISTS` the loser of that race raises. Callers treat it
# as success — whoever won did the same migration.
DUPLICATE_COLUMN_ERROR = "duplicate column name"
