from __future__ import annotations

import json
import random
from collections.abc import Sequence
from typing import Any, Optional, cast, overload

from langchain_core.runnables import RunnableConfig
from pyodbc import Cursor, Row

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS

MetadataInput = Optional[dict[str, Any]]


class BaseSQLServerSaver(BaseCheckpointSaver[str]):
    def _migrate_pending_sends(
        self,
        pending_sends: list[tuple[str, str]],
        checkpoint: dict[str, Any],
        channel_values: list[tuple[str, str, str]],
    ) -> None:
        if not pending_sends:
            return
        # add to values
        enc, blob = self.serde.dumps_typed(
            [self.serde.loads_typed((c, bytes.fromhex(b))) for c, b in pending_sends],
        )
        channel_values.append((TASKS.encode(), enc.encode(), blob))
        # add to versions
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    def _load_blobs(
        self, blob_values: list[tuple[bytes, bytes, bytes]]
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k.decode(): self.serde.loads_typed((t.decode(), v))
            for k, t, v in blob_values
            if t.decode() != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, bytes | None]]:
        if not versions:
            return []

        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                cast(str, ver),
                *(
                    self.serde.dumps_typed(values[k])
                    if k in values
                    else ("empty", None)
                ),
            )
            for k, ver in versions.items()
        ]

    def _load_writes(
        self, writes: list[tuple[str, str, str, str]]
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid,
                    channel,
                    self.serde.loads_typed((t, bytes.fromhex(v))),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _search_where(
        self,
        config: RunnableConfig | None,
        before: RunnableConfig | None = None,
    ) -> tuple[str, list[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before.

        This method returns a tuple of a string and a tuple of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = $1 AND column2 IS $2". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = ? ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = ? ")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = ? ")
                param_values.append(checkpoint_id)

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < ? ")
            param_values.append(get_checkpoint_id(before))

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )

    def _filter_metadata(
        self, metadata: dict[str, Any], filter: dict[str, Any]
    ) -> bool:
        """
        Recursively checks if the metadata contains all key-value pairs in the filter.
        This function supports nested dictionaries and lists, allowing for complex
        queries against the metadata structure.

        This replaces PostgreSQL's `@>` operator that doesn't exist in SQLServer.

        Args:
            metadata (dict): The metadata dictionary to check against.
            filter (dict): The filter dictionary containing key-value pairs to match.

        Returns:
            bool: True if metadata contains all key-value pairs in filter, False otherwise.

        Examples:
            >>> self._filter_metadata(
            ...     {"key1": "value1", "key2": {"subkey": "subvalue"}},
            ...     {"key2": {"subkey": "subvalue"}}
            ... )
            True
            >>> self._filter_metadata(
            ...     {"key1": "value1", "key2": {"subkey": "subvalue"}},
            ...     {"key2": {"subkey": "wrongvalue"}}
            ... )
            False
        """
        if isinstance(filter, dict):
            if not isinstance(metadata, dict):
                return False
            return all(
                key in metadata and self._filter_metadata(metadata[key], val)
                for key, val in filter.items()
            )

        if isinstance(filter, list):
            if not isinstance(metadata, list):
                return False
            # Every element in the subset list must be found in the superset list.
            # This handles nested objects within lists.
            return all(
                any(
                    self._filter_metadata(super_item, sub_item)
                    for super_item in metadata
                )
                for sub_item in filter
            )

        # For primitive values, check for equality.
        return filter == metadata

    @overload
    def _postprocess_results(
        self, data: list[Row], cursor: Cursor, *, json_fields: list[str] | None = None
    ) -> list[dict[str, Any]]: ...

    @overload
    def _postprocess_results(
        self,
        data: Row,
        cursor: Cursor,
        *,
        json_fields: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def _postprocess_results(
        self,
        data: list[Row],
        cursor: Cursor,
        *,
        json_fields: list[str] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Post-proces raw data from a database query, converting it into a dictionary format and parsing JSON fields if necessary.

        Args:
            data: A tuple representing a database row or a list of such tuples representing multiple rows
            cursor: A database cursor from which to extract field names
            json_fields: A list of field names that should be parsed as JSON strings

        Returns:
            A dictionary mapping field names to their corresponding values in the row
        """
        if not data:
            return data

        columns = [column[0] for column in cursor.description]
        if isinstance(data, Row):
            data = dict(zip(columns, data))
            if json_fields:
                return self._parse_json_fields(data, json_fields)
            else:
                return data
        elif isinstance(data, list):
            data = [dict(zip(columns, row)) for row in data]
            if json_fields:
                return [self._parse_json_fields(row, json_fields) for row in data]
            else:
                return data
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. Expected Row or List[Row]."
            )

    def _parse_json_fields(
        self, data: dict[str, Any], json_fields: list[str]
    ) -> dict[str, Any]:
        """Parse specified fields in a dictionary as JSON strings.

        Args:
            data: The dictionary containing the data to be processed.
            json_fields: A list of field names that should be parsed as JSON strings.

        Returns:
            dict: The updated dictionary with specified fields parsed as JSON.
        """
        for json_field in json_fields or []:
            if json_field in data and isinstance(data[json_field], str):
                data[json_field] = json.loads(data[json_field])
        return data
