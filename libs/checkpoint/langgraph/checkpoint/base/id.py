"""Adapted from
https://github.com/oittaa/uuid6-python/blob/main/src/uuid6/__init__.py#L95
Bundled in to avoid install issues with uuid6 package
"""

import random
import time
import uuid
from typing import Optional, Tuple

_last_v6_timestamp = None


class UUID(uuid.UUID):
    r"""UUID draft version objects"""

    __slots__ = ()

    def __init__(
        self,
        hex: Optional[str] = None,
        bytes: Optional[bytes] = None,
        bytes_le: Optional[bytes] = None,
        fields: Optional[Tuple[int, int, int, int, int, int]] = None,
        int: Optional[int] = None,
        version: Optional[int] = None,
        *,
        is_safe: uuid.SafeUUID = uuid.SafeUUID.unknown,
    ) -> None:
        r"""Create a UUID."""

        if int is None or [hex, bytes, bytes_le, fields].count(None) != 4:
            return super().__init__(
                hex=hex,
                bytes=bytes,
                bytes_le=bytes_le,
                fields=fields,
                int=int,
                version=version,
                is_safe=is_safe,
            )
        if not 0 <= int < 1 << 128:
            raise ValueError("int is out of range (need a 128-bit value)")
        if version is not None:
            if not 6 <= version <= 8:
                raise ValueError("illegal version number")
            # Set the variant to RFC 4122.
            int &= ~(0xC000 << 48)
            int |= 0x8000 << 48
            # Set the version number.
            int &= ~(0xF000 << 64)
            int |= version << 76
        super().__init__(int=int, is_safe=is_safe)

    @property
    def subsec(self) -> int:
        return ((self.int >> 64) & 0x0FFF) << 8 | ((self.int >> 54) & 0xFF)

    @property
    def time(self) -> int:
        if self.version == 6:
            return (
                (self.time_low << 28)
                | (self.time_mid << 12)
                | (self.time_hi_version & 0x0FFF)
            )
        if self.version == 7:
            return self.int >> 80
        if self.version == 8:
            return (self.int >> 80) * 10**6 + _subsec_decode(self.subsec)
        return super().time


def _subsec_decode(value: int) -> int:
    return -(-value * 10**6 // 2**20)


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
        clock_seq = random.getrandbits(14)  # instead of stable storage
    if node is None:
        node = random.getrandbits(48)
    time_high_and_time_mid = (timestamp >> 12) & 0xFFFFFFFFFFFF
    time_low_and_version = timestamp & 0x0FFF
    uuid_int = time_high_and_time_mid << 80
    uuid_int |= time_low_and_version << 64
    uuid_int |= (clock_seq & 0x3FFF) << 48
    uuid_int |= node & 0xFFFFFFFFFFFF
    return UUID(int=uuid_int, version=6)
