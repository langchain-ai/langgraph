from __future__ import annotations

import struct
from collections.abc import Sequence

_AAD_VERSION = b"langgraph-aad-v1"


def build_aad(kind: str, fields: Sequence[str | int]) -> bytes:
    """Build canonical, unambiguous associated data for encrypted checkpoint rows.

    The encoding includes a protocol version, a domain separator, the field
    count, and a length prefix for every field.
    """
    encoded_kind = kind.encode("utf-8")
    encoded_fields = [str(field).encode("utf-8") for field in fields]

    parts = [
        _AAD_VERSION,
        struct.pack(">I", len(encoded_kind)),
        encoded_kind,
        struct.pack(">I", len(encoded_fields)),
    ]

    for field in encoded_fields:
        parts.extend(
            (
                struct.pack(">I", len(field)),
                field,
            )
        )

    return b"".join(parts)


def build_checkpoint_aad(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
) -> bytes:
    """Build AAD for a checkpoint row."""
    return build_aad(
        "checkpoint",
        (thread_id, checkpoint_ns, checkpoint_id),
    )


def build_metadata_aad(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
) -> bytes:
    """Build AAD for checkpoint metadata."""
    return build_aad(
        "metadata",
        (thread_id, checkpoint_ns, checkpoint_id),
    )


def build_blob_aad(
    thread_id: str,
    checkpoint_ns: str,
    channel: str,
    version: str,
) -> bytes:
    """Build AAD for a checkpoint blob."""
    return build_aad(
        "blob",
        (thread_id, checkpoint_ns, channel, version),
    )


def build_shallow_blob_aad(
    thread_id: str,
    checkpoint_ns: str,
    channel: str,
) -> bytes:
    """Build AAD for a shallow checkpoint blob."""
    return build_aad(
        "shallow_blob",
        (thread_id, checkpoint_ns, channel),
    )


def build_write_aad(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: int,
    channel: str,
) -> bytes:
    """Build AAD for a checkpoint write."""
    return build_aad(
        "write",
        (thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx), channel),
    )
