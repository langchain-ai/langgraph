"""Compressed serializer wrapper for reducing checkpoint storage bloat.

Wraps any `SerializerProtocol` implementation with transparent zlib
compression.  On write, the inner serializer's output bytes are
compressed and the type tag is suffixed with ``+zlib``.  On read, the
suffix is detected automatically — checkpoints written **before** the
wrapper was installed (without the suffix) are passed through to the
inner serializer unchanged, so existing in-flight threads continue to
work without any migration step.

Usage::

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.compressed import CompressedSerializer

    # Use with default JsonPlusSerializer
    checkpointer = InMemorySaver(serde=CompressedSerializer())

    # Or wrap a custom inner serializer
    checkpointer = InMemorySaver(
        serde=CompressedSerializer(my_inner_serde, level=9)
    )
"""

from __future__ import annotations

import zlib
from typing import Any

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# 4-byte marker prepended to every compressed payload so that a future
# format migration (e.g. switching from zlib to zstd) can distinguish
# codecs without inspecting the type tag.
_MARKER = b"LGZ\x01"


class CompressedSerializer(SerializerProtocol):
    """Serializer wrapper that adds transparent zlib compression.

    Wraps any existing `SerializerProtocol` and compresses the serialized
    bytes with zlib.  Compressed payloads are prefixed with a 4-byte magic
    marker (`LGZ`) and the inner type tag is suffixed with ``+zlib``.

    Checkpoints written **before** this wrapper was installed do *not*
    have the ``+zlib`` suffix — they are forwarded to the inner serde
    unchanged, so existing threads keep working without migration.

    Args:
        serde: Inner serializer to delegate actual (de)serialization to.
            Defaults to `JsonPlusSerializer()`.
        level: zlib compression level (0-9).  ``0`` means no compression,
            ``1`` is fastest, ``9`` is most compact.  Default ``6`` is a
            good balance of speed and ratio.

    Example::

        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.checkpoint.serde.compressed import CompressedSerializer

        checkpointer = InMemorySaver(serde=CompressedSerializer())
    """

    def __init__(
        self,
        serde: SerializerProtocol | None = None,
        *,
        level: int = 6,
    ) -> None:
        self.serde = serde or JsonPlusSerializer()
        self.level = level

    # -- SerializerProtocol ---------------------------------------------------

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize and compress.

        Returns a ``(type_tag, payload)`` tuple where *type_tag* carries
        the ``+zlib`` suffix and *payload* starts with the ``LGZ`` magic
        marker followed by zlib-compressed bytes.
        """
        typ, data = self.serde.dumps_typed(obj)
        compressed = zlib.compress(data, self.level)
        return f"{typ}+zlib", _MARKER + compressed

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Deserialize, decompressing if the payload was compressed.

        If the type tag does **not** contain ``+zlib``, the data is
        forwarded directly to the inner serde — this is the transparent
        backward-compatibility path for pre-existing checkpoints.
        """
        typ, payload = data
        if "+zlib" not in typ:
            return self.serde.loads_typed(data)
        # Strip the ``+zlib`` suffix to recover the inner type tag.
        inner_typ = typ.rsplit("+zlib", 1)[0]
        # Strip the 4-byte marker prefix and decompress.
        decompressed = zlib.decompress(payload[len(_MARKER) :])
        return self.serde.loads_typed((inner_typ, decompressed))
