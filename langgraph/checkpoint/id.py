"""Adapted from
https://github.com/oittaa/uuid6-python/blob/main/src/uuid6/__init__.py#L95
Bundled in to avoid install issues with uuid6 package
"""

import secrets
import time
from typing import Optional
from uuid import UUID

_last_v6_timestamp = None


def uuid6(node: Optional[int] = None, clock_seq: Optional[int] = None) -> UUID:
    r"""UUID version 6 is a field-compatible version of UUIDv1, reordered for
    improved DB locality. It is expected that UUIDv6 will primarily be
    used in contexts where there are existing v1 UUIDs. Systems that do
    not involve legacy UUIDv1 SHOULD consider using UUIDv7 instead.

    If 'node' is not given, a random 48-bit number is chosen.

    If 'clock_seq' is given, it is used as the sequence number;
    otherwise a random 14-bit sequence number is chosen."""

    global _last_v6_timestamp

    nanoseconds = time.time_ns()
    # 0x01b21dd213814000 is the number of 100-ns intervals between the
    # UUID epoch 1582-10-15 00:00:00 and the Unix epoch 1970-01-01 00:00:00.
    timestamp = nanoseconds // 100 + 0x01B21DD213814000
    if _last_v6_timestamp is not None and timestamp <= _last_v6_timestamp:
        timestamp = _last_v6_timestamp + 1
    _last_v6_timestamp = timestamp
    if clock_seq is None:
        clock_seq = secrets.randbits(14)  # instead of stable storage
    if node is None:
        node = secrets.randbits(48)
    time_high_and_time_mid = (timestamp >> 12) & 0xFFFFFFFFFFFF
    time_low_and_version = timestamp & 0x0FFF
    uuid_int = time_high_and_time_mid << 80
    uuid_int |= time_low_and_version << 64
    uuid_int |= (clock_seq & 0x3FFF) << 48
    uuid_int |= node & 0xFFFFFFFFFFFF
    return UUID(int=uuid_int, version=6)
