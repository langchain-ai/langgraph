#!/usr/bin/env python3
"""Recover delta-channel state from a Postgres-backed LangGraph thread.

Works with OSS ``PostgresSaver`` and LangGraph Server / langgraph-api on the
Postgres runtime (same checkpoint schema). Use after rolling back from
langgraph >= 1.2 / deepagents 0.6.x to an older runtime that does not
understand ``EXT_DELTA_SNAPSHOT`` msgpack blobs. The script walks the checkpoint
parent chain, decodes msgpack blobs, and emits a JSON dump of per-channel
``seed`` plus oldest-to-newest ``writes``. Apply the recovered values manually
via ``client.threads.update_state(...)`` (Server) or ``graph.update_state``
(OSS).

Install::

    pip install "psycopg[binary]" ormsgpack

Run::

    export DATABASE_URI=postgres://...
    python3 dump.py --thread-id <uuid> --channel messages --output recovery.json

Scope (v1): Postgres only; no AES/custom encryption; no reducer application.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import uuid
from typing import Any

import ormsgpack
import psycopg

# LangGraph msgpack EXT type codes (langgraph/checkpoint/serde/jsonplus.py).
EXT_CONSTRUCTOR_SINGLE_ARG = 0
EXT_CONSTRUCTOR_POS_ARGS = 1
EXT_CONSTRUCTOR_KW_ARGS = 2
EXT_METHOD_SINGLE_ARG = 3
EXT_PYDANTIC_V1 = 4
EXT_PYDANTIC_V2 = 5
EXT_NUMPY_ARRAY = 6
EXT_DELTA_SNAPSHOT = 7

_MSGPACK_OPTION = ormsgpack.OPT_NON_STR_KEYS


def ext_hook(code: int, data: bytes) -> Any:
    """Decode LangGraph msgpack EXT payloads to JSON-friendly Python values."""
    if code == EXT_DELTA_SNAPSHOT:
        inner = ormsgpack.unpackb(data, ext_hook=ext_hook, option=_MSGPACK_OPTION)
        return {"__delta_snapshot__": inner}
    if code == EXT_CONSTRUCTOR_SINGLE_ARG:
        try:
            tup = ormsgpack.unpackb(data, ext_hook=ext_hook, option=_MSGPACK_OPTION)
            if tup[0] == "uuid" and tup[1] == "UUID":
                hex_ = tup[2]
                return (
                    f"{hex_[:8]}-{hex_[8:12]}-{hex_[12:16]}-"
                    f"{hex_[16:20]}-{hex_[20:]}"
                )
            return tup[2]
        except Exception:
            return None
    if code == EXT_CONSTRUCTOR_POS_ARGS:
        try:
            tup = ormsgpack.unpackb(data, ext_hook=ext_hook, option=_MSGPACK_OPTION)
            if tup[0] == "langgraph.types" and tup[1] == "Send":
                args = tup[2]
                if len(args) == 2:
                    return {"__send__": {"node": args[0], "arg": args[1]}}
                return {
                    "__send__": {
                        "node": args[0],
                        "arg": args[1],
                        "timeout": args[2],
                    }
                }
            return tup[2]
        except Exception:
            return None
    if code in (EXT_CONSTRUCTOR_KW_ARGS, EXT_METHOD_SINGLE_ARG):
        try:
            tup = ormsgpack.unpackb(data, ext_hook=ext_hook, option=_MSGPACK_OPTION)
            return tup[2]
        except Exception:
            return None
    if code in (EXT_PYDANTIC_V1, EXT_PYDANTIC_V2):
        try:
            tup = ormsgpack.unpackb(data, ext_hook=ext_hook, option=_MSGPACK_OPTION)
            return tup[2]
        except Exception:
            return None
    if code == EXT_NUMPY_ARRAY:
        try:
            dtype_str, shape, order, buf = ormsgpack.unpackb(
                data, ext_hook=ext_hook, option=_MSGPACK_OPTION
            )
            return {
                "__numpy_array__": {
                    "dtype": dtype_str,
                    "shape": shape,
                    "order": order,
                    "data_b64": base64.b64encode(buf).decode("ascii"),
                }
            }
        except Exception:
            return None
    return None


def decode_blob(blob_type: str, blob_bytes: bytes | None) -> Any:
    if blob_type in ("empty", "null") or blob_bytes is None:
        return None
    if blob_type == "msgpack":
        return ormsgpack.unpackb(
            blob_bytes, ext_hook=ext_hook, option=_MSGPACK_OPTION
        )
    if blob_type in ("bytes", "bytearray"):
        return base64.b64encode(blob_bytes).decode("ascii")
    raise RuntimeError(
        f"Unknown blob type {blob_type!r}. "
        "AES/custom-encrypted deployments are out of scope for v1."
    )


def delta_unwrap(value: Any) -> Any:
    if isinstance(value, dict) and "__delta_snapshot__" in value:
        return value["__delta_snapshot__"]
    return value


def json_default(obj: Any) -> Any:
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def resolve_target_checkpoint_id(
    conn: psycopg.Connection[Any],
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str | None,
) -> str:
    if checkpoint_id is not None:
        return checkpoint_id
    row = conn.execute(
        """
        SELECT checkpoint_id::text
        FROM checkpoints
        WHERE thread_id = %s AND checkpoint_ns = %s
        ORDER BY checkpoint_id DESC
        LIMIT 1
        """,
        (thread_id, checkpoint_ns),
    ).fetchone()
    if row is None:
        raise SystemExit(
            f"No checkpoints found for thread_id={thread_id!r} "
            f"checkpoint_ns={checkpoint_ns!r}"
        )
    return row[0]


def _load_checkpoint(
    conn: psycopg.Connection[Any],
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
) -> tuple[dict[str, Any], str | None] | None:
    row = conn.execute(
        """
        SELECT checkpoint, parent_checkpoint_id::text
        FROM checkpoints
        WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
        """,
        (thread_id, checkpoint_ns, checkpoint_id),
    ).fetchone()
    if row is None:
        return None
    return row[0], row[1]


def _load_seed(
    conn: psycopg.Connection[Any],
    *,
    thread_id: str,
    checkpoint_ns: str,
    channel: str,
    checkpoint_id: str,
    channel_values: dict[str, Any],
    channel_versions: dict[str, str],
) -> dict[str, Any]:
    cv = channel_values[channel]
    version = channel_versions.get(channel)
    if cv is True:
        blob_row = conn.execute(
            """
            SELECT type, blob
            FROM checkpoint_blobs
            WHERE thread_id = %s AND checkpoint_ns = %s
              AND channel = %s AND version = %s
            """,
            (thread_id, checkpoint_ns, channel, version),
        ).fetchone()
        seed_value = None
        if blob_row is not None and blob_row[0] != "empty":
            seed_value = delta_unwrap(decode_blob(blob_row[0], blob_row[1]))
        return {
            "delta_kind": "snapshot",
            "seed_checkpoint_id": checkpoint_id,
            "seed_version": version,
            "seed": seed_value,
        }
    if isinstance(cv, (int, float, str, bool)) or cv is None:
        return {
            "delta_kind": "legacy_plain",
            "seed_checkpoint_id": checkpoint_id,
            "seed_version": version,
            "seed": cv,
        }
    blob_row = conn.execute(
        """
        SELECT type, blob
        FROM checkpoint_blobs
        WHERE thread_id = %s AND checkpoint_ns = %s
          AND channel = %s AND version = %s
        """,
        (thread_id, checkpoint_ns, channel, version),
    ).fetchone()
    seed_value = cv if blob_row is None else decode_blob(blob_row[0], blob_row[1])
    return {
        "delta_kind": "legacy_plain",
        "seed_checkpoint_id": checkpoint_id,
        "seed_version": version,
        "seed": seed_value,
    }


def _load_writes_for_checkpoint(
    conn: psycopg.Connection[Any],
    *,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    channel: str,
) -> list[dict[str, Any]]:
    """Load writes for one checkpoint, newest-first by ``(task_id, idx)``.

    ``walk_channel`` reverses the accumulated flat list before returning;
    DESC here yields oldest-first within each checkpoint in the final output,
    matching ``PostgresSaver._build_delta_channels_writes_history``.
    """
    rows = conn.execute(
        """
        SELECT task_id::text, idx, type, blob
        FROM checkpoint_writes
        WHERE thread_id = %s AND checkpoint_ns = %s
          AND checkpoint_id = %s AND channel = %s
        ORDER BY task_id DESC, idx DESC
        """,
        (thread_id, checkpoint_ns, checkpoint_id, channel),
    ).fetchall()
    return [
        {
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "idx": idx,
            "value": decode_blob(blob_type, blob_bytes),
        }
        for task_id, idx, blob_type, blob_bytes in rows
    ]


def walk_channel(
    conn: psycopg.Connection[Any],
    *,
    thread_id: str,
    checkpoint_ns: str,
    start_checkpoint_id: str | None,
    channel: str,
) -> dict[str, Any]:
    """Walk one channel's parent chain from target.parent backward to seed."""
    chain_writes_newest_first: list[dict[str, Any]] = []
    cur = start_checkpoint_id
    while cur is not None:
        loaded = _load_checkpoint(conn, thread_id, checkpoint_ns, cur)
        if loaded is None:
            break
        checkpoint_json, parent_id = loaded
        channel_values = checkpoint_json.get("channel_values") or {}
        channel_versions = checkpoint_json.get("channel_versions") or {}

        chain_writes_newest_first.extend(
            _load_writes_for_checkpoint(
                conn,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=cur,
                channel=channel,
            )
        )

        if channel in channel_values:
            seed = _load_seed(
                conn,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                channel=channel,
                checkpoint_id=cur,
                channel_values=channel_values,
                channel_versions=channel_versions,
            )
            seed["writes"] = list(reversed(chain_writes_newest_first))
            return seed

        cur = parent_id

    return {
        "delta_kind": "no_seed",
        "seed_checkpoint_id": None,
        "seed_version": None,
        "seed": None,
        "writes": list(reversed(chain_writes_newest_first)),
    }


def walk_parent_chain(
    conn: psycopg.Connection[Any],
    *,
    thread_id: str,
    checkpoint_ns: str,
    target_checkpoint_id: str,
    channels: list[str],
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """Walk parent chain and return per-channel seed + writes (oldest-first)."""
    parent_row = conn.execute(
        """
        SELECT parent_checkpoint_id::text
        FROM checkpoints
        WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
        """,
        (thread_id, checkpoint_ns, target_checkpoint_id),
    ).fetchone()
    if parent_row is None:
        raise SystemExit(
            f"Checkpoint {target_checkpoint_id!r} not found for thread {thread_id!r}"
        )
    parent_checkpoint_id = parent_row[0]

    result = {
        ch: walk_channel(
            conn,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            start_checkpoint_id=parent_checkpoint_id,
            channel=ch,
        )
        for ch in channels
    }
    return parent_checkpoint_id, result


def build_output(
    *,
    thread_id: str,
    checkpoint_ns: str,
    target_checkpoint_id: str,
    parent_checkpoint_id: str | None,
    channels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "target_checkpoint_id": target_checkpoint_id,
        "parent_checkpoint_id": parent_checkpoint_id,
        "channels": channels,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover delta-channel seed + writes from Postgres checkpoint data."
        ),
    )
    parser.add_argument("--thread-id", required=True, help="Thread UUID")
    parser.add_argument(
        "--channel",
        action="append",
        required=True,
        dest="channels",
        help="Channel name (repeatable)",
    )
    parser.add_argument(
        "--checkpoint-id",
        default=None,
        help="Target checkpoint UUID (default: latest for thread)",
    )
    parser.add_argument(
        "--checkpoint-ns",
        default="",
        help='Checkpoint namespace (default: "")',
    )
    parser.add_argument(
        "--database-uri",
        default=os.environ.get("DATABASE_URI"),
        help="Postgres URI (default: DATABASE_URI env var)",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output JSON file path (default: stdout)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_uri:
        print(
            "error: --database-uri or DATABASE_URI is required",
            file=sys.stderr,
        )
        return 2

    thread_id = str(uuid.UUID(args.thread_id))
    channels = list(dict.fromkeys(args.channels))

    with psycopg.connect(args.database_uri) as conn:
        target_checkpoint_id = resolve_target_checkpoint_id(
            conn,
            thread_id,
            args.checkpoint_ns,
            args.checkpoint_id,
        )
        parent_checkpoint_id, channel_data = walk_parent_chain(
            conn,
            thread_id=thread_id,
            checkpoint_ns=args.checkpoint_ns,
            target_checkpoint_id=target_checkpoint_id,
            channels=channels,
        )

    output = build_output(
        thread_id=thread_id,
        checkpoint_ns=args.checkpoint_ns,
        target_checkpoint_id=target_checkpoint_id,
        parent_checkpoint_id=parent_checkpoint_id,
        channels=channel_data,
    )
    payload = json.dumps(output, indent=2, default=json_default)

    if args.output == "-":
        print(payload)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
