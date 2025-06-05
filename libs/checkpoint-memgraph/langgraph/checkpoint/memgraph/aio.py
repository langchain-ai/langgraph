"""
Async Memgraph Checkpointer for LangGraph.
Implements the async BaseCheckpointSaver interface.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, AsyncIterator

from neo4j import AsyncDriver
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
    Checkpoint,
    CheckpointMetadata,
    ChannelVersions,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.memgraph._ainternal import Conn, AsyncMemgraphConn, aget_session


class AsyncMemgraphSaver(BaseCheckpointSaver):
    """
    Async version of MemgraphSaver. Uses neo4j's AsyncDriver for Memgraph.
    """

    def __init__(
        self,
        conn: Conn,
        serde=None,
    ) -> None:
        super().__init__(serde=serde or JsonPlusSerializer())
        self.conn = conn
        self._setup_done = False
        self._lock = asyncio.Lock()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, uri: str, user: Optional[str] = None, password: Optional[str] = None
    ) -> AsyncIterator["AsyncMemgraphSaver"]:
        """
        Usage:
            async with AsyncMemgraphSaver.from_conn_string("bolt://localhost:7687") as saver:
                await saver.setup()
                ...
        """
        mgconn = AsyncMemgraphConn(uri, user, password)
        try:
            yield cls(conn=mgconn)
        finally:
            await mgconn.close()

    async def __aenter__(self) -> "AsyncMemgraphSaver":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self):
        if isinstance(self.conn, AsyncMemgraphConn):
            await self.conn.close()
        elif isinstance(self.conn, AsyncDriver):
            await self.conn.close()

    async def setup(self) -> None:
        if self._setup_done:
            return
        async with aget_session(self.conn) as session:
            try:
                await session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS ON (c:Checkpoint) ASSERT (c.thread_id, c.checkpoint_id) IS UNIQUE
                """)
            except:
                pass
            try:
                await session.run("CREATE INDEX IF NOT EXISTS ON :Checkpoint(thread_id)")
            except:
                pass
        self._setup_done = True

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        async with self._lock:
            return await self._put_locked(config, checkpoint, metadata)

    async def _put_locked(
        self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata
    ) -> RunnableConfig:
        thread_id = str(config["configurable"].get("thread_id"))
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        parent_id = config["configurable"].get("checkpoint_id")
        new_cp_id = checkpoint.get("id") or parent_id
        if not new_cp_id:
            new_cp_id = self.serde.generate_id()
            checkpoint["id"] = new_cp_id

        cp_bytes = self.serde.dumps(checkpoint)
        cp_str = cp_bytes.decode("utf-8") if isinstance(cp_bytes, bytes) else cp_bytes

        meta_dict = get_checkpoint_metadata(config, metadata)
        meta_bytes = self.serde.dumps(meta_dict)
        meta_str = meta_bytes.decode("utf-8") if isinstance(meta_bytes, bytes) else meta_bytes

        await self.setup()

        async with aget_session(self.conn) as session:
            query = """
            MERGE (c:Checkpoint {
                thread_id: $thread_id,
                checkpoint_ns: $checkpoint_ns,
                checkpoint_id: $checkpoint_id
            })
            ON CREATE SET c.parent_id = $parent_id,
                          c.data = $cp_data,
                          c.metadata = $cp_meta
            ON MATCH SET c.data = $cp_data,
                         c.metadata = $cp_meta,
                         c.parent_id = $parent_id
            """
            await session.run(
                query,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=new_cp_id,
                parent_id=parent_id,
                cp_data=cp_str,
                cp_meta=meta_str,
            )

        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": new_cp_id,
            }
        }
        return next_config

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        cp_id = config["configurable"].get("checkpoint_id")
        if not thread_id:
            return None

        await self.setup()
        record = None
        async with aget_session(self.conn) as session:
            if cp_id:
                query = """
                MATCH (c:Checkpoint {thread_id:$thread_id, checkpoint_ns:$checkpoint_ns, checkpoint_id:$cp_id})
                RETURN c.data as data, c.metadata as metadata, c.parent_id as parent_id
                """
                rec = await session.run(
                    query,
                    thread_id=str(thread_id),
                    checkpoint_ns=str(checkpoint_ns),
                    cp_id=str(cp_id),
                )
                record = await rec.single()
            else:
                query = """
                MATCH (c:Checkpoint {thread_id:$thread_id, checkpoint_ns:$checkpoint_ns})
                RETURN c.data as data, c.metadata as metadata, c.parent_id as parent_id
                LIMIT 1
                """
                rec = await session.run(
                    query,
                    thread_id=str(thread_id),
                    checkpoint_ns=str(checkpoint_ns),
                )
                record = await rec.single()

        if not record:
            return None

        raw_cp = record["data"]
        raw_meta = record["metadata"]
        parent_id = record["parent_id"]
        cp_dict = self.serde.loads(raw_cp.encode("utf-8") if raw_cp else b"")
        meta_dict = self.serde.loads(raw_meta.encode("utf-8") if raw_meta else b"")
        parent_config = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }
        return CheckpointTuple(
            config, cp_dict, meta_dict, parent_config
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        thread_id = None
        checkpoint_ns = ""
        if config and "thread_id" in config["configurable"]:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        await self.setup()
        params = {}
        conditions = []
        if thread_id is not None:
            conditions.append("c.thread_id = $thread_id")
            params["thread_id"] = str(thread_id)
        if checkpoint_ns is not None:
            conditions.append("c.checkpoint_ns = $checkpoint_ns")
            params["checkpoint_ns"] = str(checkpoint_ns)
        cypher = "MATCH (c:Checkpoint) "
        if conditions:
            cypher += " WHERE " + " AND ".join(conditions)
        cypher += " RETURN c.data as data, c.metadata as metadata, c.parent_id as parent_id, c.checkpoint_id as cp_id "
        if limit is not None:
            cypher += f" LIMIT {limit}"

        async with aget_session(self.conn) as session:
            cursor = await session.run(cypher, **params)
            async for rec in cursor:
                cp_id = rec["cp_id"]
                raw_cp = rec["data"]
                raw_meta = rec["metadata"]
                parent_id = rec["parent_id"]
                cp_dict = self.serde.loads(raw_cp.encode("utf-8") if raw_cp else b"")
                meta_dict = self.serde.loads(raw_meta.encode("utf-8") if raw_meta else b"")
                parent_config = None
                if parent_id:
                    parent_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_id,
                        }
                    }
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": cp_id,
                        }
                    },
                    cp_dict,
                    meta_dict,
                    parent_config
                )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Any,
    ) -> None:
        thread_id = str(config["configurable"].get("thread_id"))
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        cp_id = config["configurable"].get("checkpoint_id")
        if not cp_id:
            return
        writes_bytes = self.serde.dumps(writes)
        writes_str = writes_bytes.decode("utf-8") if isinstance(writes_bytes, bytes) else writes_bytes

        await self.setup()

        async with aget_session(self.conn) as session:
            query = """
            MATCH (cp:Checkpoint {thread_id:$thread_id, checkpoint_ns:$checkpoint_ns, checkpoint_id:$cp_id})
            CREATE (w:CheckpointWrite {content:$content})
            CREATE (cp)-[:HAS_WRITE]->(w)
            """
            await session.run(
                query,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                cp_id=cp_id,
                content=writes_str,
            )
