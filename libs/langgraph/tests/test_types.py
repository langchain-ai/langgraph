from collections.abc import Hashable

from langgraph.types import Send


class _Unhashable:
    """A plain object with no __hash__."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Unhashable) and self.value == other.value


def test_send_is_hashable_with_dict_arg():
    # Canonical usage from the docs: Send("node", {"key": "value"}).
    packet = Send("generate_report", {"user_id": 42, "format": "pdf"})
    assert isinstance(packet, Hashable)
    assert hash(packet) == hash(packet)


def test_send_hash_consistent_with_eq():
    a = Send("node", {"a": 1, "b": [2, 3]})
    b = Send("node", {"a": 1, "b": [2, 3]})
    assert a == b
    assert hash(a) == hash(b)


def test_send_hash_ignores_dict_insertion_order():
    a = Send("node", {"a": 1, "b": 2})
    b = Send("node", {"b": 2, "a": 1})
    assert a == b
    assert hash(a) == hash(b)


def test_send_usable_in_set():
    s = {Send("node", {"a": 1}), Send("node", {"a": 1})}
    assert len(s) == 1


def test_send_hash_nested_containers():
    packet = Send("node", {"items": [{"x": 1}, {2, 3}], "flag": True})
    assert isinstance(hash(packet), int)


def test_send_hash_falls_back_on_arbitrary_unhashable_arg():
    packet = Send("node", {"obj": _Unhashable(1)})
    assert isinstance(hash(packet), int)
