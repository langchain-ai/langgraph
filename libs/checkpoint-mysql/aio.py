import asyncio
import random
import ssl
import json
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
import aiomysql
from aiomysql.cursors import DictCursor
from urllib.parse import urlparse, parse_qs
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import ChannelProtocol

T = TypeVar("T", bound=callable)

def _metadata_predicate(
    metadata_filter: Dict[str, Any],
) -> Tuple[Sequence[str], Sequence[Any]]:
    """Return WHERE clause predicates for (a)search() given metadata filter.

    This method returns a tuple of a string and a tuple of values. The string
    is the parametered WHERE clause predicate (excluding the WHERE keyword):
    "column1 = ? AND column2 IS ?". The tuple of values contains the values
    for each of the corresponding parameters.
    """

    def _where_value(query_value: Any) -> Tuple[str, Any]:
        """Return tuple of operator and value for WHERE clause predicate."""
        if query_value is None:
            return ("IS ?", None)
        elif (
            isinstance(query_value, str)
            or isinstance(query_value, int)
            or isinstance(query_value, float)
        ):
            return ("= ?", query_value)
        elif isinstance(query_value, bool):
            return ("= ?", 1 if query_value else 0)
        elif isinstance(query_value, dict) or isinstance(query_value, list):
            # query value for JSON object cannot have trailing space after separators (, :)
            # SQLite json_extract() returns JSON string without whitespace
            return ("= CAST(? AS JSON)", json.dumps(query_value, separators=(",", ":")))
        else:
            return ("= ?", str(query_value))

    predicates = []
    param_values = []

    # process metadata query
    for query_key, query_value in metadata_filter.items():
        operator, param_value = _where_value(query_value)
        predicates.append(
            f"JSON_EXTRACT(metadata, '$.{query_key}') {operator}"
        )
        param_values.append(param_value)

    return (predicates, param_values)


def search_where(
    config: Optional[RunnableConfig],
    filter: Optional[Dict[str, Any]],
    before: Optional[RunnableConfig] = None,
) -> Tuple[str, Sequence[Any]]:
    """Return WHERE clause predicates for (a)search() given metadata filter
    and `before` config.

    This method returns a tuple of a string and a tuple of values. The string
    is the parametered WHERE clause predicate (including the WHERE keyword):
    "WHERE column1 = ? AND column2 IS ?". The tuple of values contains the
    values for each of the corresponding parameters.
    """
    wheres = []
    param_values = []

    # construct predicate for config filter
    if config is not None:
        wheres.append("thread_id = ?")
        param_values.append(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns")
        if checkpoint_ns is not None:
            wheres.append("checkpoint_ns = ?")
            param_values.append(checkpoint_ns)

        if checkpoint_id := get_checkpoint_id(config):
            wheres.append("checkpoint_id = ?")
            param_values.append(checkpoint_id)

    # construct predicate for metadata filter
    if filter:
        metadata_predicates, metadata_values = _metadata_predicate(filter)
        wheres.extend(metadata_predicates)
        param_values.extend(metadata_values)

    # construct predicate for `before`
    if before is not None:
        wheres.append("checkpoint_id < ?")
        param_values.append(get_checkpoint_id(before))

    return ("WHERE " + " AND ".join(wheres) if wheres else "", param_values)

class AsyncMySQLSaver(BaseCheckpointSaver):
    """An asynchronous checkpoint saver that stores checkpoints in a MySQL database.

    This class provides an asynchronous interface for saving and retrieving checkpoints
    using a MySQL database. It's designed for use in asynchronous environments and
    offers better performance for I/O-bound operations compared to synchronous alternatives.

    Attributes:
        conn (aiomysql.Connection): The asynchronous MySQL database connection.
        serde (SerializerProtocol): The serializer used for encoding/decoding checkpoints.
    """

    lock: asyncio.Lock
    is_setup: bool

    def __init__(
        self,
        conn: aiomysql.Connection,
        *,
        serde: Optional[SerializerProtocol] = None,
    ):
        super().__init__(serde=serde)
        self.jsonplus_serde = JsonPlusSerializer()
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.is_setup = False

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, conn_string: str
    ) -> AsyncIterator["AsyncMySQLSaver"]:
        """Create a new AsyncMySQLSaver instance from a connection string.

        Args:
            conn_string (str): The MySQL connection string.

        Yields:
            AsyncMySQLSaver: A new AsyncMySQLSaver instance.
        """
        parsed = urlparse(conn_string)
        query_params = parse_qs(parsed.query)
        db_config = {
            "host": parsed.hostname,
            "port": parsed.port,
            "user": parsed.username,
            "password": parsed.password,
            "db": parsed.path.lstrip('/'),
            "autocommit": True,
        }
        ssl_config = {}
        if 'ssl_ca' in query_params:
            ssl_context = ssl.create_default_context(cafile=query_params['ssl_ca'][0])
            ssl_context.check_hostname = query_params.get('ssl_verify_identity', ['true'])[0].lower() == 'true'
            ssl_context.verify_mode = ssl.CERT_REQUIRED if query_params.get('ssl_verify_cert', ['true'])[0].lower() == 'true' else ssl.CERT_NONE
            ssl_config['ssl'] = ssl_context
        # Create the connection
        conn = await aiomysql.connect(**db_config, **ssl_config)
        try:
            yield AsyncMySQLSaver(conn)
        finally:
            conn.close()

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the MySQL database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the MySQL database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
        """
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_), self.loop
                ).result()
            except StopAsyncIteration:
                break

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the MySQL database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self, config: RunnableConfig, writes: List[Tuple[str, Any]], task_id: str
    ) -> None:
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id), self.loop
        ).result()

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method checks if the necessary tables exist in the MySQL database.
        If they don't exist, it creates them. This method is called automatically
        when needed and should not be called directly by the user.
        """
        async with self.lock:
            async with self.conn.cursor() as cur:
                # Check if tables exist
                await cur.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                AND table_name IN ('dev_ai_agent_checkpoints', 'dev_ai_agent_writes')
                """)
                result = await cur.fetchone()
                tables_exist = result[0] == 2

                if not tables_exist:
                    # Create tables if they don't exist
                    await cur.execute("""
                    CREATE TABLE IF NOT EXISTS dev_ai_agent_checkpoints (
                        thread_id VARCHAR(128) NOT NULL,
                        checkpoint_ns VARCHAR(128) NOT NULL DEFAULT '',
                        checkpoint_id VARCHAR(128) NOT NULL,
                        parent_checkpoint_id VARCHAR(128),
                        type VARCHAR(255),
                        checkpoint LONGBLOB,
                        metadata LONGBLOB,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                    """)
                    await cur.execute("""
                    CREATE TABLE IF NOT EXISTS dev_ai_agent_writes (
                        thread_id VARCHAR(128) NOT NULL,
                        checkpoint_ns VARCHAR(128) NOT NULL DEFAULT '',
                        checkpoint_id VARCHAR(128) NOT NULL,
                        task_id VARCHAR(128) NOT NULL,
                        idx INT NOT NULL,
                        channel VARCHAR(255) NOT NULL,
                        type VARCHAR(255),
                        value LONGBLOB,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    )
                    """)
                    await self.conn.commit()

            self.is_setup = True

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the MySQL database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        await self.setup()
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        async with self.lock, self.conn.cursor(DictCursor) as cur:
            # find the latest checkpoint for the thread_id
            if checkpoint_id := get_checkpoint_id(config):
                await cur.execute(
                    "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM dev_ai_agent_checkpoints WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s",
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        checkpoint_id,
                    ),
                )
            else:
                await cur.execute(
                    "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM dev_ai_agent_checkpoints WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1",
                    (str(config["configurable"]["thread_id"]), checkpoint_ns),
                )
            # if a checkpoint is found, return it
            if value := await cur.fetchone():
                if not get_checkpoint_id(config):
                    config = {
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    }
                await cur.execute(
                    "SELECT task_id, channel, type, value FROM dev_ai_agent_writes WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s ORDER BY task_id, idx",
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        str(config["configurable"]["checkpoint_id"]),
                    ),
                )
                writes = [
                    (row["task_id"], row["channel"], self.serde.loads_typed((row["type"], row["value"])))
                    async for row in cur
                ]
                return CheckpointTuple(
                    config,
                    self.serde.loads_typed((value["type"], value["checkpoint"])),
                    self.jsonplus_serde.loads(value["metadata"]) if value["metadata"] is not None else {},
                    (
                        {
                            "configurable": {
                                "thread_id": value["thread_id"],
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": value["parent_checkpoint_id"],
                            }
                        }
                        if value["parent_checkpoint_id"]
                        else None
                    ),
                    writes,
                )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the MySQL database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        await self.setup()
        where, params = search_where(config, filter, before)
        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
        FROM dev_ai_agent_checkpoints
        {where}
        ORDER BY checkpoint_id DESC"""
        if limit:
            query += f" LIMIT {limit}"
        async with self.lock, self.conn.cursor(DictCursor) as cur, self.conn.cursor(DictCursor) as wcur:
            await cur.execute(query, params)
            async for row in cur:
                await wcur.execute(
                    "SELECT task_id, channel, type, value FROM dev_ai_agent_writes WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s ORDER BY task_id, idx",
                    (row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"])
                )
                writes = [
                    (write["task_id"], write["channel"], self.serde.loads_typed((write["type"], write["value"])))
                    async for write in wcur
                ]
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": row["thread_id"],
                            "checkpoint_ns": row["checkpoint_ns"],
                            "checkpoint_id": row["checkpoint_id"],
                        }
                    },
                    self.serde.loads_typed((row["type"], row["checkpoint"])),
                    self.jsonplus_serde.loads(row["metadata"]) if row["metadata"] is not None else {},
                    (
                        {
                            "configurable": {
                                "thread_id": row["thread_id"],
                                "checkpoint_ns": row["checkpoint_ns"],
                                "checkpoint_id": row["parent_checkpoint_id"],
                            }
                        }
                        if row["parent_checkpoint_id"]
                        else None
                    ),
                    writes,
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the MySQL database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.jsonplus_serde.dumps(metadata)
        async with self.lock, self.conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO dev_ai_agent_checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE type=%s, checkpoint=%s, metadata=%s",
                (
                    str(thread_id),
                    checkpoint_ns,
                    checkpoint["id"],
                    config["configurable"].get("checkpoint_id"),
                    type_,
                    serialized_checkpoint,
                    serialized_metadata,
                    type_,
                    serialized_checkpoint,
                    serialized_metadata,
                ),
            )
            await self.conn.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        query = (
            "INSERT INTO dev_ai_agent_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE type = VALUES(type), value = VALUES(value)"
        )

        await self.setup()

        async with self.lock, self.conn.cursor() as cur:
            await cur.executemany(
                query,
                [
                    (
                        str(config["configurable"]["thread_id"]),
                        str(config["configurable"]["checkpoint_ns"]),
                        str(config["configurable"]["checkpoint_id"]),
                        task_id,
                        WRITES_IDX_MAP.get(channel, idx),
                        channel,
                        *self.serde.dumps_typed(value),
                    )
                    for idx, (channel, value) in enumerate(writes)
                ],
            )

            await self.conn.commit()

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        """Generate the next version ID for a channel.

        This method creates a new version identifier for a channel based on its current version.

        Args:
            current (Optional[str]): The current version identifier of the channel.
            channel (BaseChannel): The channel being versioned.

        Returns:
            str: The next version identifier, which is guaranteed to be monotonically increasing.
        """
        if current is None:
            current_v = 0
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
