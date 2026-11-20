"""Regression tests for Send's hash-once, freeze-on-hash behavior."""

import pytest

from langgraph.types import Send


def test_send_dict_payload_hashable_and_order_independent() -> None:
    left = Send("process", {"x": 1, "nested": {"a": [1, 2], "b": {3, 4}}})
    right = Send("process", {"nested": {"b": {4, 3}, "a": [1, 2]}, "x": 1})
    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1


def test_send_preserves_list_vs_tuple_equality() -> None:
    left = Send("process", {"items": [1, 2]})
    right = Send("process", {"items": (1, 2)})
    assert left != right
    assert hash(left) != hash(right) or left != right


def test_send_mutable_until_hashed() -> None:
    packet = Send("process", {"items": [1]})
    packet.arg["items"].append(2)
    packet.node = "other"
    assert hash(packet) == hash(packet)
    assert packet.arg == {"items": [1, 2]}


@pytest.mark.parametrize("attribute,value", [
    ("node", "other"),
    ("arg", {"new": True}),
    ("timeout", None),
])
def test_send_rejects_field_writes_after_hash(attribute: str, value: object) -> None:
    packet = Send("process", {"x": 1})
    hash(packet)
    with pytest.raises(TypeError, match="immutable once hashed"):
        setattr(packet, attribute, value)


@pytest.mark.parametrize("mutation", [
    lambda arg: arg.__setitem__("x", 2),
    lambda arg: arg["nested"].__setitem__("y", 2),
    lambda arg: arg["list"].append(3),
    lambda arg: arg["list"].__setitem__(0, 3),
    lambda arg: arg["set"].add(4),
])
def test_send_rejects_nested_mutation_after_hash(mutation) -> None:
    packet = Send("process", {"x": 1, "nested": {"y": 1}, "list": [1], "set": {2}})
    hash(packet)
    with pytest.raises(TypeError, match="immutable once hashed"):
        mutation(packet.arg)


def test_send_hash_does_not_freeze_callers_original_state() -> None:
    original = {"items": [1]}
    packet = Send("process", original)
    frozen_hash = hash(packet)
    original["items"].append(2)
    original["other"] = 3
    assert packet.arg == {"items": [1]}
    assert hash(packet) == frozen_hash


def test_send_unsupported_payload_can_still_be_mutated_after_failed_hash() -> None:
    packet = Send("process", {"unhashable": bytearray(b"hello")})
    with pytest.raises(TypeError):
        hash(packet)
    packet.arg["unhashable"] = "supported"
    assert isinstance(hash(packet), int)


def test_send_cyclic_payload_rejected_without_freezing() -> None:
    cyclic = []
    cyclic.append(cyclic)
    packet = Send("process", cyclic)
    with pytest.raises(TypeError, match="cyclic"):
        hash(packet)
    packet.arg.clear()


def test_send_hashable_scalar_unchanged() -> None:
    packet = Send("process", 42)
    assert hash(packet) == hash(("process", 42, None))
