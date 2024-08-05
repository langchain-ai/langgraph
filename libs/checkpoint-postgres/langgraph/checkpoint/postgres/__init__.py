import asyncio
from base64 import b64decode, b64encode
from hashlib import md5
from typing import Any, AsyncIterator, List, Optional, Tuple

from langchain_core.runnables import RunnableConfig
from psycopg import AsyncConnection, AsyncPipeline
from psycopg.types.json import Jsonb

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    EmptyChannelError,
    get_checkpoint_id,
)
from langgraph.checkpoint.postgres.serde import JsonAndBinarySerializer
from langgraph.checkpoint.serde.types import ChannelProtocol

MetadataInput = Optional[dict[str, Any]]

SELECT_SQL = """
select
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    (
        select array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob])
        from jsonb_each_text(checkpoint -> 'channel_versions')
        inner join checkpoint_blobs bl
            on bl.thread_id = checkpoints.thread_id
            and bl.channel = jsonb_each_text.key
            and bl.version = jsonb_each_text.value
    ) as channel_values,
    (
        select
        array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob])
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns = checkpoints.checkpoint_ns
            and cw.checkpoint_id = checkpoints.checkpoint_id
    ) as pending_writes
from checkpoints """


class PostgresSaver(BaseCheckpointSaver):
    serde: JsonAndBinarySerializer

    lock: asyncio.Lock
    latest_iter: Optional[AsyncIterator[CheckpointTuple]]
    latest_tuple: Optional[CheckpointTuple]

    is_setup: bool

    def __init__(
        self,
        conn: AsyncConnection,
        pipe: AsyncPipeline | None = None,
        latest: Optional[AsyncIterator[CheckpointTuple]] = None,
    ) -> None:
        super().__init__(serde=JsonAndBinarySerializer())
        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.latest_iter = latest
        self.latest_tuple: Optional[CheckpointTuple] = None
        self.is_setup = False

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the SQLite database if they don't
        already exist. It is called automatically when needed and should not be called
        directly by the user.
        """
        async with self.lock:
            if self.is_setup:
                return

            create_table_queries = [
                """CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint JSONB NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )""",
                """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                    thread_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    version TEXT NOT NULL,
                    type TEXT NOT NULL,
                    blob BYTEA NOT NULL,
                    PRIMARY KEY (thread_id, channel, version)
                );""",
                """CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    blob BYTEA NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                );
                """,
            ]
            for query in create_table_queries:
                await self.conn.execute(query)

            if self.pipe:
                await self.pipe.sync()

            self.is_setup = True

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        await self.setup()
        where, args = self._search_where(config, filter, before)
        query = SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        async for value in await self.conn.execute(query, args, binary=True):
            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": value["thread_id"],
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["checkpoint_id"],
                    }
                },
                {
                    **self._load_checkpoint(value["checkpoint"]),
                    "channel_values": await asyncio.to_thread(
                        self._load_blobs, value["channel_values"]
                    ),
                },
                value["metadata"],
                {
                    "configurable": {
                        "thread_id": value["thread_id"],
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["parent_checkpoint_id"],
                    }
                }
                if value["parent_checkpoint_id"]
                else None,
            )

    async def aget_iter(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        cur = await self.conn.execute(
            SELECT_SQL + where,
            args,
            binary=True,
        )

        return (
            CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": value["checkpoint_id"],
                    }
                },
                {
                    **self._load_checkpoint(value["checkpoint"]),
                    "channel_values": await asyncio.to_thread(
                        self._load_blobs, value["channel_values"]
                    ),
                },
                value["metadata"],
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": value["parent_checkpoint_id"],
                    }
                }
                if value["parent_checkpoint_id"]
                else None,
                await asyncio.to_thread(self._load_writes, value["pending_writes"]),
            )
            async for value in cur
        )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        await self.setup()
        if (
            self.latest_tuple is not None
            and self.latest_tuple.config["configurable"]["thread_id"]
            == config["configurable"]["thread_id"]
        ):
            return self.latest_tuple
        elif self.latest_iter is not None:
            try:
                self.latest_tuple = await anext(self.latest_iter, None)
                if not self.latest_tuple:
                    return None
                elif (
                    self.latest_tuple.config["configurable"]["thread_id"]
                    == config["configurable"]["thread_id"]
                ):
                    return self.latest_tuple
            finally:
                self.latest_iter = None

        return await anext(await self.aget_iter(config), None)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        await self.setup()
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )
        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }
        previous = (
            self.latest_tuple
            if self.latest_tuple
            and checkpoint_id
            and self.latest_tuple.config["configurable"]["thread_id"] == thread_id
            and self.latest_tuple.config["configurable"]["checkpoint_ns"]
            == checkpoint_ns
            and self.latest_tuple.config["configurable"]["checkpoint_id"]
            == checkpoint_id
            else None
        )
        self.latest_tuple = CheckpointTuple(
            config=next_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=config,
        )

        await self.conn.cursor(binary=True).executemany(
            """INSERT INTO checkpoint_blobs (thread_id, channel, version, type, blob)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (thread_id, channel, version) DO NOTHING""",
            await asyncio.to_thread(
                self._dump_blobs,
                thread_id,
                copy.pop("channel_values"),
                copy["channel_versions"],
                previous.checkpoint["channel_versions"] if previous else None,
            ),
        )
        await self.conn.execute(
            """
            INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
            DO UPDATE SET
                checkpoint = EXCLUDED.checkpoint,
                metadata = EXCLUDED.metadata;""",
            (
                thread_id,
                checkpoint_ns,
                checkpoint["id"],
                checkpoint_id,
                Jsonb(self._dump_checkpoint(copy)),
                # Merging `configurable` and `metadata` will persist graph_id,
                # assistant_id, and all assistant and run configurable fields
                # to the checkpoint metadata.
                Jsonb({**configurable, **config.get("metadata", {}), **metadata}),
            ),
            binary=True,
        )
        if self.pipe:
            await self.pipe.sync()
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        await self.setup()
        await self.conn.cursor(binary=True).executemany(
            """INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING""",
            await asyncio.to_thread(
                self._dump_writes,
                config["configurable"]["thread_id"],
                config["configurable"]["checkpoint_ns"],
                config["configurable"]["checkpoint_id"],
                task_id,
                writes,
            ),
        )
        if self.pipe:
            await self.pipe.sync()

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        try:
            next_h = md5(self.serde.dumps(channel.checkpoint())[1]).hexdigest()
        except EmptyChannelError:
            next_h = ""
        return f"{next_v:032}.{next_h}"

    def _load_checkpoint(self, checkpoint: dict[str, Any]) -> Checkpoint:
        if len(checkpoint["pending_sends"]) == 2 and all(
            isinstance(a, str) for a in checkpoint["pending_sends"]
        ):
            type, bs = checkpoint["pending_sends"]
            return {
                **checkpoint,
                "pending_sends": self.serde.loads((type, b64decode(bs))),
            }

        return checkpoint

    def _dump_checkpoint(self, checkpoint: Checkpoint) -> dict[str, Any]:
        type, bs = self.serde.dumps(checkpoint["pending_sends"])
        return {
            **checkpoint,
            "pending_sends": (type, b64encode(bs).decode()),
        }

    def _load_blobs(
        self, blob_values: list[tuple[bytes, bytes, bytes]]
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k.decode(): self.serde.loads((t.decode(), v))
            for k, t, v in blob_values
            if t.decode() != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        values: dict[str, Any],
        versions: dict[str, str],
        previous_versions: Optional[dict[str, str]],
    ) -> list[tuple[str, str, str, str, bytes]]:
        if not versions:
            return []
        if previous_versions is not None:
            version_type = type(next(iter(versions.values()), None))
            null_version = version_type()
            versions = {
                k: v
                for k, v in versions.items()
                if v > previous_versions.get(k, null_version)
            }
        return [
            (
                thread_id,
                k,
                ver,
                *(self.serde.dumps(values[k]) if k in values else ("empty", None)),
            )
            for k, ver in versions.items()
        ]

    def _load_writes(
        self, writes: list[tuple[bytes, bytes, bytes, bytes]]
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid.decode(),
                    channel.decode(),
                    self.serde.loads((t.decode(), v)),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_id: str,
        task_id: str,
        writes: list[tuple[str, Any]],
    ) -> list[tuple[str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_id,
                task_id,
                idx,
                channel,
                *self.serde.dumps(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def _search_where(
        self,
        config: Optional[RunnableConfig],
        filter: MetadataInput,
        before: Optional[RunnableConfig] = None,
    ) -> Tuple[str, List[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, cursor.

        This method returns a tuple of a string and a tuple of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = $1 AND column2 IS $2". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            wheres.append("checkpoint_ns = %s")
            param_values.append(checkpoint_ns)

        # construct predicate for metadata filter
        if filter:
            wheres.append("metadata @> %s ")
            param_values.append(Jsonb(filter))

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < %s ")
            param_values.append(get_checkpoint_id(before))

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )
