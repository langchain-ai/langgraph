"""Base implementation for Kusto checkpointer."""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version as get_version
from typing import Any, cast

import orjson
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import TASKS

MetadataInput = dict[str, Any] | None

try:
    major, minor = get_version("langgraph").split(".")[:2]
    if int(major) == 0 and int(minor) < 5:
        warnings.warn(
            "You're using incompatible versions of langgraph and checkpoint-kusto. "
            "Please upgrade langgraph to avoid unexpected behavior.",
            DeprecationWarning,
            stacklevel=2,
        )
except Exception:
    # skip version check if running from source
    pass


# KQL Query Templates
# Note: Kusto uses different syntax than traditional SQL. Key differences:
# - No traditional JOIN syntax, use let statements and mv-expand
# - JSON parsing via parse_json()
# - Arrays via make_list() in summarize
# - Use materialized views with arg_max() for "latest" semantics (more efficient than ORDER BY + TAKE 1)
# - Ordering via | order by
# - Limiting via | take
# - Blobs stored as dynamic column in Checkpoints table (leveraging columnar storage)

# For querying the latest checkpoint, we use the LatestCheckpoints materialized view
# which pre-computes arg_max(checkpoint_id, *) by thread_id, checkpoint_ns
# This is significantly more efficient than scanning all checkpoints and sorting
SELECT_LATEST_CHECKPOINT_KQL = """
LatestCheckpoints
| where thread_id == thread_id_param and checkpoint_ns == checkpoint_ns_param
{checkpoint_id_filter}
| extend checkpoint_parsed = parse_json(checkpoint_json)
| extend metadata_parsed = parse_json(metadata_json)
| join kind=leftouter (
    CheckpointWrites
    | where thread_id == thread_id_param and checkpoint_ns == checkpoint_ns_param
    | order by task_id asc, idx asc
    | summarize pending_writes = make_list(pack("task_id", task_id, "channel", channel, "type", type, "value_json", value_json)) by thread_id, checkpoint_ns, checkpoint_id
) on thread_id, checkpoint_ns, checkpoint_id
| project thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, 
          checkpoint=checkpoint_parsed, metadata=metadata_parsed, 
          channel_values=coalesce(channel_values, dynamic([])),
          pending_writes=coalesce(pending_writes, dynamic([]))
"""

# For querying specific checkpoint or listing checkpoints, use the base Checkpoints table
# with ORDER BY and TAKE for filtering/pagination
SELECT_CHECKPOINT_KQL = """
Checkpoints
| where thread_id == thread_id_param and checkpoint_ns == checkpoint_ns_param
{checkpoint_id_filter}
| order by checkpoint_id desc
{limit_clause}
| extend checkpoint_parsed = parse_json(checkpoint_json)
| extend metadata_parsed = parse_json(metadata_json)
| join kind=leftouter (
    CheckpointWrites
    | where thread_id == thread_id_param and checkpoint_ns == checkpoint_ns_param
    | order by task_id asc, idx asc
    | summarize pending_writes = make_list(pack("task_id", task_id, "channel", channel, "type", type, "value_json", value_json)) by thread_id, checkpoint_ns, checkpoint_id
) on thread_id, checkpoint_ns, checkpoint_id
| project thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, 
          checkpoint=checkpoint_parsed, metadata=metadata_parsed, 
          channel_values=coalesce(channel_values, dynamic([])),
          pending_writes=coalesce(pending_writes, dynamic([]))
"""

DELETE_THREAD_KQL_CHECKPOINTS = """
.delete table Checkpoints records <| Checkpoints | where thread_id == "{thread_id}"
"""

DELETE_THREAD_KQL_WRITES = """
.delete table CheckpointWrites records <| CheckpointWrites | where thread_id == "{thread_id}"
"""


class BaseKustoSaver(BaseCheckpointSaver[str]):
    """Base class for Kusto checkpoint savers.
    
    This class provides shared functionality for both sync and async Kusto
    checkpoint savers, including serialization, query building, and data
    transformation logic.
    
    Uses Kusto materialized views with arg_max() for optimal "latest checkpoint"
    query performance instead of ORDER BY + TAKE 1 pattern.
    
    Blobs are stored in the Checkpoints table as a dynamic column, leveraging
    Kusto's columnar storage for efficient compression and querying.
    """

    SELECT_LATEST_CHECKPOINT_KQL = SELECT_LATEST_CHECKPOINT_KQL
    SELECT_CHECKPOINT_KQL = SELECT_CHECKPOINT_KQL
    DELETE_THREAD_KQL_CHECKPOINTS = DELETE_THREAD_KQL_CHECKPOINTS
    DELETE_THREAD_KQL_WRITES = DELETE_THREAD_KQL_WRITES

    def _migrate_pending_sends(
        self,
        pending_sends: list[tuple[bytes, bytes]],
        checkpoint: dict[str, Any],
        channel_values: list[dict[str, Any]],
    ) -> None:
        """Migrate pending sends from old format to new format.
        
        This method handles backwards compatibility for checkpoints created
        with older LangGraph versions.
        
        Args:
            pending_sends: List of pending send tuples (type, blob).
            checkpoint: The checkpoint dictionary to update.
            channel_values: The channel values list to append to.
        """
        if not pending_sends:
            return
        
        # Deserialize old format and re-serialize using JsonPlusSerializer
        tasks = [self.serde.loads_typed((c.decode(), b)) for c, b in pending_sends]
        # Use JsonPlusSerializer for consistent serialization
        type_str, blob = self.serde.dumps_typed(tasks)
        if isinstance(blob, bytes):
            blob_value = blob.decode('utf-8')
        else:
            blob_value = blob
        
        channel_values.append({
            "channel": TASKS,
            "type": type_str,
            "blob": blob_value,
        })
        # add to versions
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    def _load_blobs(
        self, blob_values: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Load and deserialize blob values using the configured serializer.
        
        Args:
            blob_values: List of blob dictionaries with channel, type, and blob.
            
        Returns:
            Dictionary mapping channel names to deserialized values.
        """
        if not blob_values:
            return {}
        result = {}
        for item in blob_values:
            if item["type"] == "empty":
                continue
            
            blob = item["blob"]
            type_str = item["type"]
            
            # Convert blob to bytes if it's a string (from Kusto)
            if isinstance(blob, str):
                blob = blob.encode('utf-8')
            
            # Use serde (JsonPlusSerializer) for all deserialization
            result[item["channel"]] = self.serde.loads_typed((type_str, blob))
        
        return result

    def _dump_blobs(
        self,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[dict[str, Any]]:
        """Serialize and dump blob values using the configured serializer.
        
        Returns a list of blob objects suitable for storing in the
        Checkpoints.channel_values dynamic column. This leverages Kusto's
        columnar storage for efficient compression.
        
        Uses the configured serializer (default: JsonStringSerializer) for
        consistent serialization of all types, including LangChain messages.
        
        Args:
            values: Dictionary of channel values to serialize.
            versions: Channel version mappings.
            
        Returns:
            List of dictionaries for the dynamic column (no thread_id/checkpoint_ns needed).
        """
        if not versions:
            return []

        records = []
        for k, ver in versions.items():
            if k in values:
                # Use serde (JsonPlusSerializer) for all serialization
                type_str, blob = self.serde.dumps_typed(values[k])
                # Convert bytes to string for Kusto storage
                if isinstance(blob, bytes):
                    blob_value = blob.decode('utf-8')
                else:
                    blob_value = blob
            else:
                type_str, blob_value = "empty", ""
            
            records.append({
                "channel": k,
                "version": cast(str, ver),
                "type": type_str,
                "blob": blob_value,
            })
        return records

    def _load_writes(
        self, writes: list[dict[str, Any]]
    ) -> list[tuple[str, str, Any]]:
        """Load and deserialize checkpoint writes using the configured serializer.
        
        Args:
            writes: List of write dictionaries from Kusto.
            
        Returns:
            List of (task_id, channel, value) tuples.
        """
        if not writes:
            return []
        
        result = []
        for write in writes:
            value_json = write["value_json"]
            type_str = write["type"]
            
            # Convert to bytes if string (from Kusto)
            if isinstance(value_json, str):
                value_json = value_json.encode('utf-8')
            
            # Use serde (JsonPlusSerializer) for all deserialization
            value = self.serde.loads_typed((type_str, value_json))
            
            result.append((
                write["task_id"],
                write["channel"],
                value,
            ))
        
        return result

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[dict[str, Any]]:
        """Serialize and dump checkpoint writes using the configured serializer.
        
        Uses the configured serializer (default: JsonStringSerializer) for
        consistent serialization of all types.
        
        Args:
            thread_id: The thread identifier.
            checkpoint_ns: The checkpoint namespace.
            checkpoint_id: The checkpoint identifier.
            task_id: The task identifier.
            task_path: The task path.
            writes: Sequence of (channel, value) tuples to serialize.
            
        Returns:
            List of dictionaries ready for Kusto ingestion.
        """
        records = []
        for idx, (channel, value) in enumerate(writes):
            # Use serde (JsonPlusSerializer) for all serialization
            type_str, blob = self.serde.dumps_typed(value)
            # Convert bytes to string for Kusto storage
            if isinstance(blob, bytes):
                value_str = blob.decode('utf-8')
            else:
                value_str = blob
            
            records.append({
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "type": type_str,
                "value_json": value_str,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        return records

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Generate the next version string for a channel.
        
        Versions follow the format: {major:032d}.{random:016f}
        This ensures proper ordering while avoiding collisions.
        
        Args:
            current: The current version string, or None.
            channel: The channel (unused, for interface compatibility).
            
        Returns:
            The next version string.
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _build_kql_filter(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Build KQL filter parameters for checkpoint queries.
        
        This method constructs filter conditions and parameters for Kusto queries.
        Kusto uses parameterized queries with the | where operator.
        
        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional metadata filtering criteria.
            before: If provided, filter to checkpoints before this ID.
            
        Returns:
            Dictionary with filter parameters and clauses.
        """
        params = {}
        filters = []

        # Extract config-based filters
        if config:
            params["thread_id_param"] = config["configurable"]["thread_id"]
            params["checkpoint_ns_param"] = config["configurable"].get("checkpoint_ns", "")
            
            if checkpoint_id := get_checkpoint_id(config):
                params["checkpoint_id_param"] = checkpoint_id
                filters.append("checkpoint_id == checkpoint_id_param")

        # Metadata filter (JSON containment check)
        if filter:
            # For Kusto, we use parse_json and check individual properties
            # Unlike traditional relational databases, Kusto requires explicit property checks
            filter_json = orjson.dumps(filter).decode()
            params["metadata_filter"] = filter_json
            # We'll handle this in the query by checking each key-value pair
            for key, value in filter.items():
                # Add filter for each metadata key
                # Note: This is a simplified approach; production may need more robust filtering
                filters.append(f'metadata_parsed.{key} == "{value}"')

        # Before filter (checkpoint_id comparison)
        if before is not None:
            before_id = get_checkpoint_id(before)
            params["before_checkpoint_id"] = before_id
            filters.append("checkpoint_id < before_checkpoint_id")

        return {
            "params": params,
            "filters": filters,
        }
