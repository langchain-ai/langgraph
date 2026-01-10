import warnings

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph.types import Interrupt
from langgraph.warnings import LangGraphDeprecatedSinceV10


@pytest.mark.filterwarnings("ignore:LangGraphDeprecatedSinceV10")
def test_interrupt_legacy_ns() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)

        old_interrupt = Interrupt(
            value="abc", resumable=True, when="during", ns=["a:b", "c:d"]
        )

        new_interrupt = Interrupt.from_ns(value="abc", ns="a:b|c:d")
        assert new_interrupt.value == old_interrupt.value
        # Note: IDs are intentionally different now.
        # from_ns includes the interrupt index in the hash to ensure unique IDs
        # for parallel interrupts within the same namespace.
        # Legacy interrupts (from deprecated constructor) preserve their original IDs.
        assert old_interrupt.id  # verify old constructor still generates an ID
        assert new_interrupt.id  # verify from_ns still generates an ID
        assert old_interrupt.id != new_interrupt.id  # they should differ due to idx


def test_from_ns_unique_ids_for_different_indices() -> None:
    """Test that from_ns generates unique IDs for different interrupt indices."""
    interrupt_0 = Interrupt.from_ns(value="test", ns="same_ns", idx=0)
    interrupt_1 = Interrupt.from_ns(value="test", ns="same_ns", idx=1)
    interrupt_2 = Interrupt.from_ns(value="test", ns="same_ns", idx=2)

    # All should have unique IDs
    assert interrupt_0.id != interrupt_1.id
    assert interrupt_1.id != interrupt_2.id
    assert interrupt_0.id != interrupt_2.id

    # Same idx should produce same ID
    interrupt_0_again = Interrupt.from_ns(value="test", ns="same_ns", idx=0)
    assert interrupt_0.id == interrupt_0_again.id


serializer = JsonPlusSerializer(allowed_json_modules=True)


def test_serialization_roundtrip() -> None:
    """Test that the legacy interrupt (pre v1) can be reserialized as the modern interrupt without id corruption."""

    # generated with:
    # JsonPlusSerializer().dumps_typed(Interrupt(value="legacy_test", ns=["legacy_test"], resumable=True, when="during"))
    legacy_interrupt_bytes = b'{"lc": 2, "type": "constructor", "id": ["langgraph", "types", "Interrupt"], "kwargs": {"value": "legacy_test", "resumable": true, "ns": ["legacy_test"], "when": "during"}}'
    legacy_interrupt_id = "f1fa625689ec006a5b32b76863e22a6c"

    interrupt = serializer.loads_typed(("json", legacy_interrupt_bytes))
    assert interrupt.id == legacy_interrupt_id
    assert interrupt.value == "legacy_test"


def test_serialization_roundtrip_complex_ns() -> None:
    """Test that the legacy interrupt (pre v1), with a more complex ns can be reserialized as the modern interrupt without id corruption."""

    # generated with:
    # JsonPlusSerializer().dumps_typed(Interrupt(value="legacy_test", ns=["legacy:test", "with:complex", "name:space"], resumable=True, when="during"))
    legacy_interrupt_bytes = b'{"lc": 2, "type": "constructor", "id": ["langgraph", "types", "Interrupt"], "kwargs": {"value": "legacy_test", "resumable": true, "ns": ["legacy:test", "with:complex", "name:space"], "when": "during"}}'
    legacy_interrupt_id = "e69356a9ee3630ee7f4f597f2693000c"

    interrupt = serializer.loads_typed(("json", legacy_interrupt_bytes))
    assert interrupt.id == legacy_interrupt_id
    assert interrupt.value == "legacy_test"
