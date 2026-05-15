"""Tests for CompressedSerializer — transparent zlib compression wrapper."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.compressed import CompressedSerializer, _MARKER
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleSerde(SerializerProtocol):
    """Minimal serde that uses JSON internally (no msgpack metadata)."""

    def dumps_typed(self, obj):
        return "json", json.dumps(obj, default=str).encode()

    def loads_typed(self, data):
        return json.loads(data[1])


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------


class TestCompressedSerializerRoundTrip:
    """Every value that goes through dumps_typed → loads_typed must survive."""

    @pytest.fixture()
    def cs(self):
        return CompressedSerializer()

    def test_none(self, cs: CompressedSerializer) -> None:
        assert cs.loads_typed(cs.dumps_typed(None)) is None

    def test_bytes(self, cs: CompressedSerializer) -> None:
        raw = b"hello bytes"
        assert cs.loads_typed(cs.dumps_typed(raw)) == raw

    def test_bytearray(self, cs: CompressedSerializer) -> None:
        raw = bytearray(b"hello bytearray")
        assert cs.loads_typed(cs.dumps_typed(raw)) == raw

    def test_string(self, cs: CompressedSerializer) -> None:
        assert cs.loads_typed(cs.dumps_typed("hello")) == "hello"

    def test_int(self, cs: CompressedSerializer) -> None:
        assert cs.loads_typed(cs.dumps_typed(42)) == 42

    def test_float(self, cs: CompressedSerializer) -> None:
        assert cs.loads_typed(cs.dumps_typed(3.14)) == 3.14

    def test_list(self, cs: CompressedSerializer) -> None:
        lst = [1, "two", 3.0, None, True]
        assert cs.loads_typed(cs.dumps_typed(lst)) == lst

    def test_dict(self, cs: CompressedSerializer) -> None:
        d = {"key": "value", "nested": {"a": 1}}
        assert cs.loads_typed(cs.dumps_typed(d)) == d

    def test_pydantic_model(self, cs: CompressedSerializer) -> None:
        msg = HumanMessage(content="What is the weather?")
        result = cs.loads_typed(cs.dumps_typed(msg))
        assert isinstance(result, HumanMessage)
        assert result.content == "What is the weather?"

    def test_nested_pydantic_models(self, cs: CompressedSerializer) -> None:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="What is 2+2?"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "calculator", "args": {"expr": "2+2"}}
                ],
            ),
            ToolMessage(content="4", tool_call_id="call_1"),
            AIMessage(content="2+2 equals 4."),
        ]
        result = cs.loads_typed(cs.dumps_typed(messages))
        assert len(result) == 7
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[5], ToolMessage)
        assert result[5].content == "4"


# ---------------------------------------------------------------------------
# Backward compatibility (no migration needed)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Uncompressed data from the inner serde must load correctly."""

    def test_uncompressed_data_passes_through(self) -> None:
        inner = JsonPlusSerializer()
        cs = CompressedSerializer(inner)

        # Simulate a checkpoint written WITHOUT compression (old format).
        typ, data = inner.dumps_typed("hello old world")
        assert "+zlib" not in typ

        # CompressedSerializer.loads_typed must handle it transparently.
        assert cs.loads_typed((typ, data)) == "hello old world"

    def test_compressed_data_uses_zlib_suffix(self) -> None:
        cs = CompressedSerializer()
        typ, _ = cs.dumps_typed("hello compressed")
        assert typ.endswith("+zlib")

    def test_compressed_payload_starts_with_marker(self) -> None:
        cs = CompressedSerializer()
        _, payload = cs.dumps_typed("test marker")
        assert payload[: len(_MARKER)] == _MARKER


# ---------------------------------------------------------------------------
# Compression actually reduces size
# ---------------------------------------------------------------------------


class TestCompressionReduction:
    """The compressed form should be smaller than the raw form."""

    def _ratio(self, obj) -> float:
        """Return compressed_size / uncompressed_size."""
        inner = JsonPlusSerializer()
        cs = CompressedSerializer(inner)

        _, raw = inner.dumps_typed(obj)
        _, compressed = cs.dumps_typed(obj)
        return len(compressed) / len(raw) if raw else 1.0

    def test_repeated_messages_compress_well(self) -> None:
        messages = []
        for i in range(20):
            messages.append(HumanMessage(content=f"User message number {i}"))
            messages.append(
                AIMessage(content=f"Assistant response number {i}")
            )
        ratio = self._ratio(messages)
        # With 40 messages, compressed should be significantly smaller.
        assert ratio < 0.7, f"Expected ratio < 0.7, got {ratio:.2f}"

    def test_dict_with_repeated_keys_compresses(self) -> None:
        d = {f"key_{i}": {"name": f"value_{i}", "metadata": {"a": 1, "b": 2}}
             for i in range(50)}
        ratio = self._ratio(d)
        assert ratio < 0.7, f"Expected ratio < 0.7, got {ratio:.2f}"

    def test_large_string_compresses(self) -> None:
        s = "The quick brown fox jumps over the lazy dog. " * 100
        ratio = self._ratio(s)
        assert ratio < 0.3, f"Expected ratio < 0.3, got {ratio:.2f}"


# ---------------------------------------------------------------------------
# Custom inner serde
# ---------------------------------------------------------------------------


class TestCustomInnerSerde:
    """CompressedSerializer should work with any SerializerProtocol."""

    def test_wraps_simple_serde(self) -> None:
        cs = CompressedSerializer(_SimpleSerde())
        obj = {"key": "value", "number": 42}
        assert cs.loads_typed(cs.dumps_typed(obj)) == obj

    def test_backward_compat_with_simple_serde(self) -> None:
        inner = _SimpleSerde()
        cs = CompressedSerializer(inner)

        # Simulate old uncompressed data from the inner serde.
        typ, data = inner.dumps_typed("old data")
        assert cs.loads_typed((typ, data)) == "old data"


# ---------------------------------------------------------------------------
# Compression level parameter
# ---------------------------------------------------------------------------


class TestCompressionLevel:
    def test_higher_level_produces_smaller_output(self) -> None:
        cs_fast = CompressedSerializer(level=1)
        cs_best = CompressedSerializer(level=9)
        s = "repeated content " * 200

        _, fast = cs_fast.dumps_typed(s)
        _, best = cs_best.dumps_typed(s)
        # Both should work.
        assert cs_fast.loads_typed(("json+zlib", fast)) == s
        assert cs_best.loads_typed(("json+zlib", best)) == s
        # Higher compression should be <= lower.
        assert len(best) <= len(fast)


# ---------------------------------------------------------------------------
# Integration with InMemorySaver
# ---------------------------------------------------------------------------


class TestInMemorySaverIntegration:
    """CompressedSerializer should work end-to-end with InMemorySaver."""

    def test_save_and_load_checkpoint(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver(serde=CompressedSerializer())

        state = {"messages": [HumanMessage(content="hello")]}
        checkpoint = {
            "v": 2,
            "id": "cp-1",
            "ts": "2025-01-01T00:00:00+00:00",
            "channel_values": state,
            "channel_versions": {"messages": 1},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        metadata = {"source": "input", "step": -1}
        config = {"configurable": {"thread_id": "test-thread"}}

        saved_config = saver.put(config, checkpoint, metadata, {"messages": 1})
        loaded = saver.get_tuple(saved_config)

        assert loaded is not None
        assert len(loaded.checkpoint["channel_values"]["messages"]) == 1
        assert isinstance(
            loaded.checkpoint["channel_values"]["messages"][0], HumanMessage
        )
        assert (
            loaded.checkpoint["channel_values"]["messages"][0].content == "hello"
        )

    def test_checkpoint_storage_smaller_with_compression(self) -> None:
        """Verify that the raw stored bytes are smaller when compression is on."""
        from langgraph.checkpoint.memory import InMemorySaver

        saver_plain = InMemorySaver()
        saver_compressed = InMemorySaver(serde=CompressedSerializer())

        messages = []
        for i in range(10):
            messages.append(HumanMessage(content=f"Turn {i}: what is order {1000+i}?"))
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"call_{i}",
                            "name": "check_order",
                            "args": {"order_id": 1000 + i},
                        }
                    ],
                )
            )
            messages.append(
                ToolMessage(
                    content=json.dumps(
                        {"order_id": 1000 + i, "status": "shipped"}
                    ),
                    tool_call_id=f"call_{i}",
                )
            )
            messages.append(
                AIMessage(content=f"Order {1000 + i} has shipped.")
            )

        state = {"messages": messages}
        checkpoint = {
            "v": 2,
            "id": "cp-1",
            "ts": "2025-01-01T00:00:00+00:00",
            "channel_values": state,
            "channel_versions": {"messages": 1},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        metadata = {"source": "input", "step": -1}

        # Save with plain saver.
        config_plain = {"configurable": {"thread_id": "thread-plain"}}
        saver_plain.put(config_plain, checkpoint, metadata, {"messages": 1})

        # Save with compressed saver.
        config_comp = {"configurable": {"thread_id": "thread-comp"}}
        saver_compressed.put(config_comp, checkpoint, metadata, {"messages": 1})

        # Compare stored blob sizes.
        plain_blob = saver_plain.blobs[("thread-plain", "", "messages", 1)]
        comp_blob = saver_compressed.blobs[("thread-comp", "", "messages", 1)]

        plain_size = len(plain_blob[1])
        comp_size = len(comp_blob[1])

        assert comp_size < plain_size, (
            f"Compressed ({comp_size}) should be smaller than plain ({plain_size})"
        )
