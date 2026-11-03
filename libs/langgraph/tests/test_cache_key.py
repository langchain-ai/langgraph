from enum import Enum

from langgraph._internal._cache import default_cache_key


class Color(Enum):
    RED = 1
    BLUE = 2


def test_mapping_with_unorderable_keys_returns_a_key():
    """Enum members are hashable but not orderable; the cache key must not raise (#9566)."""
    key = default_cache_key(x={Color.RED: "r", Color.BLUE: "b"})

    assert isinstance(key, bytes)


def test_mapping_with_mixed_key_types_returns_a_key():
    """Mixed key types are valid dict keys but not orderable."""
    assert isinstance(default_cache_key(x={1: "one", "two": 2}), bytes)
    assert isinstance(default_cache_key(x={None: 0, "a": 1}), bytes)


def test_mapping_key_order_does_not_matter():
    """Mappings with the same entries in a different order keep producing the same key."""
    assert default_cache_key(x={Color.RED: "r", Color.BLUE: "b"}) == default_cache_key(
        x={Color.BLUE: "b", Color.RED: "r"}
    )
    assert default_cache_key(x={1: "one", "two": 2}) == default_cache_key(
        x={"two": 2, 1: "one"}
    )


def test_comparable_key_mapping_key_is_unchanged():
    """Mappings whose keys are comparable keep their existing sorted behaviour."""
    assert default_cache_key(x={"b": 2, "a": 1}) == default_cache_key(
        x={"a": 1, "b": 2}
    )
    assert default_cache_key(x={"a": 1}) == default_cache_key(x={"a": 1})
