"""
Unit tests for standard graph state reducers.
"""

from typing import Any

import pytest

from langgraph.graph.reducers import combine_distinct, first_wins, smart_merge_dict


@pytest.mark.parametrize(
    ("current", "update", "expected"),
    [
        # Scenario 1: Basic non-conflicting merge
        ({"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
        # Scenario 2: Conflict upgrading to list
        ({"score": 10}, {"score": 20}, {"score": [10, 20]}),
        # Scenario 3: Mixed types flattening
        (
            {"tags": "python"},
            {"tags": ["rust", "go"]},
            {"tags": ["python", "rust", "go"]},
        ),
        # Scenario 4: List and list extension
        ({"logs": ["start"]}, {"logs": ["end"]}, {"logs": ["start", "end"]}),
        # Scenario 5: Empty state handling
        ({}, {"a": 1}, {"a": 1}),
        ({"a": 1}, {}, {"a": 1}),
    ],
)
def test_smart_merge_dict_flat(
    current: dict[str, Any], update: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Test flat dictionary merging with various conflict resolutions."""
    assert smart_merge_dict(current, update) == expected


def test_smart_merge_dict_nested() -> None:
    """Test recursive deep merge for nested dictionaries."""
    left: dict[str, Any] = {"config": {"retries": 3, "timeout": 10}}
    right: dict[str, Any] = {"config": {"retries": 5, "verbose": True}}

    # 'retries' conflicts -> becomes list [3, 5]
    # 'timeout' preserved -> 10
    # 'verbose' is new -> True added
    expected: dict[str, Any] = {
        "config": {"retries": [3, 5], "timeout": 10, "verbose": True}
    }
    assert smart_merge_dict(left, right) == expected


@pytest.mark.parametrize(
    ("current", "update", "expected"),
    [
        # Basic deduplication and order preservation
        (["a", "b"], ["b", "c", "a"], ["a", "b", "c"]),
        # Empty updates
        (["a", "b"], [], ["a", "b"]),
        # Empty current state
        ([], ["a", "b"], ["a", "b"]),
    ],
)
def test_combine_distinct(
    current: list[str], update: list[str], expected: list[str]
) -> None:
    """Test list merging with deduplication and order preservation."""
    assert combine_distinct(current, update) == expected


def test_combine_distinct_unhashable_dicts():
    # Regression: dict.fromkeys() crashed on list[dict] (e.g. RAG doc lists).
    d1 = [{"id": "a"}, {"id": "b"}]
    d2 = [{"id": "b"}, {"id": "c"}]
    assert combine_distinct(d1, d2) == [{"id": "a"}, {"id": "b"}, {"id": "c"}]


def test_combine_distinct_hashable_order_preserved():
    assert combine_distinct(["x", "y"], ["y", "z", "x"]) == ["x", "y", "z"]


def test_combine_distinct_mixed_hashable_unhashable():
    assert combine_distinct(["a", {"k": 1}], [{"k": 1}, "a", "b"]) == [
        "a",
        {"k": 1},
        "b",
    ]


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (None, ["a"], ["a"]),
        (["a"], None, ["a"]),
        (None, None, []),
    ],
)
def test_combine_distinct_none_inputs(a, b, expected):
    assert combine_distinct(a, b) == expected


def test_combine_distinct_returns_fresh_list():
    # Must not return the input object by reference.
    src = ["a"]
    assert combine_distinct(src, None) is not src


def test_smart_merge_dict_deep_merge_no_mutation():
    a = {"ctx": {"k1": 1}}
    assert smart_merge_dict(a, {"ctx": {"k2": 2}}) == {"ctx": {"k1": 1, "k2": 2}}
    assert a == {"ctx": {"k1": 1}}  # original left untouched


def test_smart_merge_dict_scalar_conflict_upgrades_to_list():
    assert smart_merge_dict({"u": "alice"}, {"u": "bob"}) == {"u": ["alice", "bob"]}


def test_smart_merge_dict_empty_and_none_not_treated_as_missing():
    assert smart_merge_dict({}, {"a": 1}) == {"a": 1}
    assert smart_merge_dict({"a": 1}, {}) == {"a": 1}
    assert smart_merge_dict(None, {"a": 1}) == {"a": 1}
    assert smart_merge_dict({"a": 1}, None) == {"a": 1}


def test_smart_merge_dict_falsy_scalar_values():
    assert smart_merge_dict({"n": 0}, {"n": 1}) == {"n": [0, 1]}


def test_first_wins():
    assert first_wins("first", "second") == "first"
    assert first_wins(None, "second") == "second"
    assert first_wins(0, 5) == 0
