"""
This module contains SQL queries for managing checkpoints in a SQL Server database.
"""

from langgraph.checkpoint.serde.types import TASKS

MIGRATIONS_SQL = [
    """
    IF OBJECT_ID(N'${SCHEMA}.checkpoint_migrations', N'U') IS NULL
    BEGIN
        CREATE TABLE ${SCHEMA}.checkpoint_migrations (
            v INT PRIMARY KEY
        )
    END
    """,
    """
    IF OBJECT_ID(N'${SCHEMA}.checkpoints', N'U') IS NULL
    BEGIN
        CREATE TABLE ${SCHEMA}.checkpoints (
            thread_id NVARCHAR(255) NOT NULL,
            checkpoint_ns NVARCHAR(255) NOT NULL CONSTRAINT DF_${SCHEMA}_checkpoints_checkpoint_ns DEFAULT '',
            checkpoint_id NVARCHAR(255) NOT NULL,
            parent_checkpoint_id NVARCHAR(255),
            [type] NVARCHAR(MAX),
            [checkpoint] NVARCHAR(MAX) NOT NULL,
            metadata NVARCHAR(MAX) NOT NULL CONSTRAINT DF_${SCHEMA}_checkpoints_metadata DEFAULT '{}',
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        )
    END
    """,
    """
    IF OBJECT_ID(N'${SCHEMA}.checkpoint_blobs', N'U') IS NULL
    BEGIN
        CREATE TABLE ${SCHEMA}.checkpoint_blobs (
            thread_id NVARCHAR(255) NOT NULL,
            checkpoint_ns NVARCHAR(255) NOT NULL CONSTRAINT DF_${SCHEMA}_checkpoint_blobs_checkpoint_ns DEFAULT '',
            channel NVARCHAR(255) NOT NULL,
            version NVARCHAR(255) NOT NULL,
            type NVARCHAR(MAX) NOT NULL,
            blob VARBINARY(MAX) DEFAULT NULL,
            PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
        )
    END
    """,
    """
    IF OBJECT_ID(N'${SCHEMA}.checkpoint_writes', N'U') IS NULL
    BEGIN
        CREATE TABLE ${SCHEMA}.checkpoint_writes (
            thread_id NVARCHAR(255) NOT NULL,
            checkpoint_ns NVARCHAR(255) NOT NULL CONSTRAINT DF_${SCHEMA}_checkpoint_writes_checkpoint_ns DEFAULT '',
            checkpoint_id NVARCHAR(255) NOT NULL,
            task_id NVARCHAR(255) NOT NULL,
            idx INT NOT NULL,
            channel NVARCHAR(255) NOT NULL,
            type NVARCHAR(MAX),
            blob VARBINARY(MAX) NOT NULL,
            task_path NVARCHAR(MAX) NOT NULL CONSTRAINT DF_${SCHEMA}_checkpoint_writes_task_path DEFAULT '',
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        )
    END
    """,
    """
    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'checkpoints_thread_id_idx' AND object_id = OBJECT_ID('${SCHEMA}.checkpoints'))
    BEGIN
        CREATE INDEX checkpoints_thread_id_idx ON ${SCHEMA}.checkpoints(thread_id);
    END
    """,
    """
    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'checkpoint_blobs_thread_id_idx' AND object_id = OBJECT_ID('${SCHEMA}.checkpoint_blobs'))
    BEGIN
        CREATE INDEX checkpoint_blobs_thread_id_idx ON ${SCHEMA}.checkpoint_blobs(thread_id);
    END
    """,
    """
    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'checkpoint_writes_thread_id_idx' AND object_id = OBJECT_ID('${SCHEMA}.checkpoint_writes'))
    BEGIN
        CREATE INDEX checkpoint_writes_thread_id_idx ON ${SCHEMA}.checkpoint_writes(thread_id);
    END
    """,
]

SELECT_SQL = """
    SELECT
        checkpoints.thread_id,
        checkpoints.[checkpoint],
        checkpoints.checkpoint_ns,
        checkpoints.checkpoint_id,
        checkpoints.parent_checkpoint_id,
        checkpoints.metadata,
        (
            SELECT 
                CONCAT(
                    '[',
                    STRING_AGG(
                        CONCAT('["', bl.channel, '", "', bl.type, '" ,"', CONVERT(NVARCHAR(MAX), bl.blob, 2), '"]'), 
                        ','
                    ),
                    ']'
                )
            FROM OPENJSON(${SCHEMA}.checkpoints.[checkpoint], '$.channel_versions')
            WITH (
                [key] NVARCHAR(MAX) '$.key',
                [value] NVARCHAR(MAX) '$.value'
            ) as json_table
            INNER JOIN ${SCHEMA}.checkpoint_blobs bl
                ON bl.thread_id = checkpoints.thread_id
                AND bl.checkpoint_ns = checkpoints.checkpoint_ns
                AND bl.channel = json_table.[key]
                AND bl.version = json_table.[value]
        ) AS channel_values,
        (
            SELECT 
                CONCAT(
                    '[',
                    STRING_AGG(
                        CONCAT('["', cw.task_id, '", "', cw.channel, '", "', cw.type,'", "', CONVERT(NVARCHAR(MAX), cw.blob, 2), '"]'),
                        ', '
                    ) WITHIN GROUP (ORDER BY cw.task_id, cw.idx),
                    ']'
                )
            FROM ${SCHEMA}.checkpoint_writes cw
            WHERE cw.thread_id = checkpoints.thread_id
                AND cw.checkpoint_ns = checkpoints.checkpoint_ns
                AND cw.checkpoint_id = checkpoints.checkpoint_id
        ) AS pending_writes
    FROM ${SCHEMA}.checkpoints
"""

SELECT_PENDING_SENDS_SQL = f"""
    SELECT
        checkpoint_id,
        CONCAT(
            '[',
            STRING_AGG(
                CONCAT('["', type, '","', CONVERT(NVARCHAR(MAX), blob, 2), '"]'), 
                ','
            ) 
            WITHIN GROUP (ORDER BY task_path, task_id, idx),
            ']'
        ) AS sends
    FROM ${{SCHEMA}}.checkpoint_writes
    WHERE thread_id = ?
        AND checkpoint_id IN (SELECT value FROM STRING_SPLIT(?, ','))
        AND channel = '{TASKS}'
    GROUP BY checkpoint_id
"""

UPSERT_CHECKPOINT_BLOBS_SQL = """
    MERGE INTO ${SCHEMA}.checkpoint_blobs AS target
    USING (VALUES (?, ?, ?, ?, ?, ?)) AS source
        (thread_id, checkpoint_ns, channel, version, type, blob)
    ON target.thread_id = source.thread_id 
        AND target.checkpoint_ns = source.checkpoint_ns
        AND target.channel = source.channel
        AND target.version = source.version
    WHEN NOT MATCHED THEN
        INSERT (thread_id, checkpoint_ns, channel, version, type, blob)
        VALUES (source.thread_id, source.checkpoint_ns, source.channel, source.version, source.type, source.blob);
"""

UPSERT_CHECKPOINTS_SQL = """
    MERGE INTO ${SCHEMA}.checkpoints AS target
    USING (VALUES (?, ?, ?, ?, ?, ?)) AS source
        (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, [checkpoint], metadata)
    ON target.thread_id = source.thread_id 
        AND target.checkpoint_ns = source.checkpoint_ns
        AND target.checkpoint_id = source.checkpoint_id
    WHEN MATCHED THEN
        UPDATE SET 
            target.[checkpoint] = source.[checkpoint],
            target.metadata = source.metadata
    WHEN NOT MATCHED THEN
        INSERT (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, [checkpoint], metadata)
        VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id, source.parent_checkpoint_id, source.[checkpoint], source.metadata);
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
    MERGE INTO ${SCHEMA}.checkpoint_writes AS target
    USING (VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)) AS source
        (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    ON target.thread_id = source.thread_id 
        AND target.checkpoint_ns = source.checkpoint_ns
        AND target.checkpoint_id = source.checkpoint_id
        AND target.task_id = source.task_id
        AND target.idx = source.idx
    WHEN MATCHED THEN
        UPDATE SET 
            target.channel = source.channel,
            target.type = source.type,
            target.blob = source.blob
    WHEN NOT MATCHED THEN
        INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
        VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id, source.task_id, source.task_path, source.idx, source.channel, source.type, source.blob);
"""

INSERT_CHECKPOINT_WRITES_SQL = """
    MERGE INTO ${SCHEMA}.checkpoint_writes AS target
    USING (VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)) AS source
        (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    ON target.thread_id = source.thread_id 
        AND target.checkpoint_ns = source.checkpoint_ns
        AND target.checkpoint_id = source.checkpoint_id
        AND target.task_id = source.task_id
        AND target.idx = source.idx
    WHEN MATCHED THEN
        UPDATE SET 
            target.channel = source.channel,
            target.type = source.type,
            target.blob = source.blob
    WHEN NOT MATCHED THEN
        INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
        VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id, source.task_id, source.task_path, source.idx, source.channel, source.type, source.blob);
"""

DELETE_THREAD_CHECKPOINTS = "DELETE FROM ${SCHEMA}.checkpoints WHERE thread_id = ?"

DELETE_THREAD_CHECKPOINT_BLOBS = (
    "DELETE FROM ${SCHEMA}.checkpoint_blobs WHERE thread_id = ?"
)

DELETE_THREAD_CHECKPOINT_WRITES = (
    "DELETE FROM ${SCHEMA}.checkpoint_writes WHERE thread_id = ?"
)
