"""Unit tests for langgraph.types value objects."""

from collections.abc import Hashable

from langgraph.types import Send


def test_send_hash_supports_dict_arg() -> None:
    """Send must stay hashable when arg is a dict (canonical documented usage).

    Regression test for https://github.com/langchain-ai/langgraph/issues/9178
    """
    send = Send("generate_report", {"user_id": 42, "format": "pdf"})
    assert isinstance(send, Hashable)
    assert hash(send) == hash(Send("generate_report", {"user_id": 42, "format": "pdf"}))
    # usable in hash-based containers (e.g. deduplicating fan-out Sends)
    sends = {Send("a", {"x": 1}), Send("b", {"y": 2}), Send("a", {"x": 1})}
    assert len(sends) == 2


def test_send_hash_supports_nested_containers() -> None:
    """hash must not raise for nested dicts/lists/tuples/sets."""
    assert hash(Send("n", {"a": [1, 2, {"b": (3,)}], "c": {4, 5}}))
    assert hash(Send("n", [1, 2, 3]))
    assert hash(Send("n", "plain"))
    assert hash(Send("n", None))


def test_send_hash_consistent_with_eq() -> None:
    """Equal Sends must hash equal even when dict key order differs."""
    first = Send("a", {"x": 1, "y": [1, 2]})
    second = Send("a", {"y": [1, 2], "x": 1})
    assert first == second
    assert hash(first) == hash(second)
    assert Send("a", {"x": 1}) != Send("a", {"x": 2})
