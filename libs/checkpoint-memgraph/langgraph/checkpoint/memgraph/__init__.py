"""
Memgraph Checkpointer (sync) for LangGraph.
Implements the BaseCheckpointSaver interface storing data in Memgraph.
"""
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from neo4j import Driver
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.memgraph._internal import MemgraphConn, Conn, get_session

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


class MemgraphSaver(BaseCheckpointSaver):
    """
    Sync Memgraph checkpointer for LangGraph.
    Each checkpoint is stored as a node labeled `Checkpoint`.
    """

    def __init__(
        self,
        conn: Conn,
        serde=None,
    ) -> None:
        """
        Create a MemgraphSaver instance.

        Args:
            conn: A MemgraphConn or raw neo4j Driver for connecting to Memgraph.
            serde: Optional custom serializer. Defaults to JsonPlusSerializer.
        """
        super().__init__(serde=serde or JsonPlusSerializer())
        self.conn = conn
        self.lock = threading.Lock()
        self._setup_done = False

    @classmethod
    @contextmanager
    def from_conn_string(
        cls, uri: str, user: Optional[str] = None, password: Optional[str] = None
    ) -> Iterator["MemgraphSaver"]:
        """
        Usage:
            with MemgraphSaver.from_conn_string("bolt://localhost:7687") as saver:
                saver.setup()
                # ...
        """
        mgconn = MemgraphConn(uri, user, password)
        try:
            yield cls(conn=mgconn)
        finally:
            mgconn.close()

    def close(self):
        """Close the underlying driver if present."""
        if isinstance(self.conn, MemgraphConn):
            self.conn.close()
        elif isinstance(self.conn, Driver):
            self.conn.close()

    def setup(self) -> None:
        """
        Create unique constraints for (thread_id, checkpoint_id).
        This ensures no duplicates.
        """
        if self._setup_done:
            return
        with get_session(self.conn) as session:
            # Create a uniqueness constraint in Memgraph:
            #   CREATE CONSTRAINT IF NOT EXISTS ON (c:Checkpoint) ASSERT (c.thread_id, c.checkpoint_id) IS UNIQUE
            # Memgraph doesn't strictly have CREATE CONSTRAINT IF NOT EXISTS, so we do a check if needed.
            # In newest versions, the 'IF NOT EXISTS' is recognized. Otherwise we catch and ignore.
            try:
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS ON (c:Checkpoint) ASSERT (c.thread_id, c.checkpoint_id) IS UNIQUE
                """)
            except Exception:
                pass
            # Also index thread_id
            try:
                session.run("CREATE INDEX IF NOT EXISTS ON :Checkpoint(thread_id)")
            except Exception:
                pass
        self._setup_done = True

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Store a checkpoint in Memgraph. If a checkpoint_id was not present, we generate or read from checkpoint.
        """
        with self.lock:
            return self._put_locked(config, checkpoint, metadata)

    def _put_locked(
        self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata
    ) -> RunnableConfig:
        thread_id = str(config["configurable"].get("thread_id"))
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        parent_id = config["configurable"].get("checkpoint_id")
        # The new checkpoint_id from the checkpoint data or generate
        new_cp_id = checkpoint.get("id") or parent_id
        if not new_cp_id:
            new_cp_id = self.serde.generate_id()
            checkpoint["id"] = new_cp_id

        # Convert checkpoint to JSON
        cp_bytes = self.serde.dumps(checkpoint)  # typically bytes
        cp_str = cp_bytes.decode("utf-8") if isinstance(cp_bytes, bytes) else cp_bytes

        # Convert metadata to JSON
        meta_dict = get_checkpoint_metadata(config, metadata)
        meta_bytes = self.serde.dumps(meta_dict)
        meta_str = (
            meta_bytes.decode("utf-8") if isinstance(meta_bytes, bytes) else meta_bytes
        )

        self.setup()

        with get_session(self.conn) as session:
            # MERGE the node
            # We'll store the main data in properties: (thread_id, checkpoint_ns, checkpoint_id, parent_id, data, metadata)
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
            session.run(
                query,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=new_cp_id,
                parent_id=parent_id,
                cp_data=cp_str,
                cp_meta=meta_str,
            )

        # Return updated config with new checkpoint_id
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": new_cp_id,
            }
        }
        return next_config

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        Retrieve a checkpoint tuple from Memgraph. If no checkpoint_id is in config,
        fetch the latest for that thread.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        cp_id = config["configurable"].get("checkpoint_id")
        if not thread_id:
            return None

        self.setup()
        with get_session(self.conn) as session:
            if cp_id:
                query = """
                MATCH (c:Checkpoint {thread_id:$thread_id, checkpoint_ns:$checkpoint_ns, checkpoint_id:$cp_id})
                RETURN c.data as data, c.metadata as metadata, c.parent_id as parent_id
                """
                rec = session.run(
                    query,
                    thread_id=str(thread_id),
                    checkpoint_ns=str(checkpoint_ns),
                    cp_id=str(cp_id),
                ).single()
            else:
                # If no cp_id, pick the latest by creation order if we stored them that way,
                # but we only stored them as data property. We might store a separate timestamp if we want to do ORDER BY.
                # For now, let's just pick any existing if there's no cp_id. We'll assume there's only one for shallow usage.
                query = """
                MATCH (c:Checkpoint {thread_id:$thread_id, checkpoint_ns:$checkpoint_ns})
                RETURN c.data as data, c.metadata as metadata, c.parent_id as parent_id
                LIMIT 1
                """
                rec = session.run(
                    query,
                    thread_id=str(thread_id),
                    checkpoint_ns=str(checkpoint_ns),
                ).single()

            if not rec:
                return None

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
            # Return the CheckpointTuple
            return CheckpointTuple(
                config=config,
                checkpoint=cp_dict,
                metadata=meta_dict,
                parent_config=parent_config,
            )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """
        List all checkpoints for the given thread (optionally with limit).
        """
        thread_id = None
        checkpoint_ns = ""
        if config and "thread_id" in config["configurable"]:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        self.setup()

        with get_session(self.conn) as session:
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

            results = session.run(cypher, **params)
            for rec in results:
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

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Any,
    ) -> None:
        """
        Store intermediate writes. We'll store them as a separate node or attach to the checkpoint node.
        For now, we attach them to the checkpoint node with a relationship :HAS_WRITE.
        """
        thread_id = str(config["configurable"].get("thread_id"))
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        cp_id = config["configurable"].get("checkpoint_id")
        if not cp_id:
            return
        writes_bytes = self.serde.dumps(writes)
        writes_str = writes_bytes.decode("utf-8") if isinstance(writes_bytes, bytes) else writes_bytes

        self.setup()

        with get_session(self.conn) as session:
            # Create a node for the writes
            query = """
            MATCH (cp:Checkpoint {thread_id:$thread_id, checkpoint_ns:$checkpoint_ns, checkpoint_id:$cp_id})
            CREATE (w:CheckpointWrite {content:$content})
            CREATE (cp)-[:HAS_WRITE]->(w)
            """
            session.run(
                query,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                cp_id=cp_id,
                content=writes_str,
            )
