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
        ({"tags": "python"}, {"tags": ["rust", "go"]}, {"tags": ["python", "rust", "go"]}),
        
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
        "config": {
            "retries": [3, 5],
            "timeout": 10,
            "verbose": True
        }
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
def test_combine_distinct(current: list[str], update: list[str], expected: list[str]) -> None:
    """Test list merging with deduplication and order preservation."""
    assert combine_distinct(current, update) == expected


def test_first_wins() -> None:
    """Test first_wins reducer logic."""
    # Ignore update if current exists
    assert first_wins(10, 20) == 10
    assert first_wins("old", "new") == "old"
    
    # Accept update if current is None
    assert first_wins(None, 20) == 20