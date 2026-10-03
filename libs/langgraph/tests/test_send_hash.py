from collections.abc import Hashable

from langgraph.types import Send


def test_send_hash_dict_payload() -> None:
    s = Send("generate_report", {"user_id": 42})
    assert isinstance(s, Hashable)
    hash(s)  # used to raise TypeError: unhashable type: 'dict'


def test_send_equal_dict_payloads_hash_equal() -> None:
    a = Send("node", {"user_id": 42, "format": "pdf"})
    b = Send("node", {"format": "pdf", "user_id": 42})
    assert a == b
    assert hash(a) == hash(b)


def test_send_dedup_in_set_and_dict() -> None:
    a = Send("node", {"user_id": 42})
    b = Send("node", {"user_id": 42})
    c = Send("node", {"user_id": 43})
    assert len({a, b, c}) == 2
    d = {a: "first"}
    d[b] = "second"
    assert d[a] == "second"


def test_send_hash_nested_payload() -> None:
    a = Send("node", {"items": [1, 2, {"x": [3]}], "tags": {"b", "a"}})
    b = Send("node", {"tags": {"a", "b"}, "items": [1, 2, {"x": [3]}]})
    assert a == b
    assert hash(a) == hash(b)


def test_send_hash_scalar_payload_unchanged() -> None:
    a = Send("node", "hello")
    b = Send("node", "hello")
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_send_hash_with_timeout() -> None:
    a = Send("node", {"user_id": 42}, timeout=5.0)
    b = Send("node", {"user_id": 42}, timeout=5.0)
    assert a == b
    assert hash(a) == hash(b)


def test_send_hash_distinct_payloads() -> None:
    a = Send("node", {"user_id": 42})
    b = Send("other_node", {"user_id": 42})
    assert a != b
    assert len({a, b}) == 2
