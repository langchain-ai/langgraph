"""Regression tests for numeric filter operators in PostgresStore.

Fixes https://github.com/langchain-ai/langgraph/issues/7684
PostgresStore numeric filter operators used text comparison (lexicographic)
instead of numeric, causing e.g. '9' >= '10' to be True.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph.store.postgres.base import BasePostgresStore


class _MockPostgresStore(BasePostgresStore):
    """Minimal concrete subclass to access _get_filter_condition."""

    pass


@pytest.fixture()
def store() -> _MockPostgresStore:
    return object.__new__(_MockPostgresStore)  # type: ignore[arg-type,misc]


@pytest.mark.parametrize(
    "op,value,expected_sql_fragment",
    [
        # Numeric values must use CAST AS NUMERIC with jsonb_typeof guard
        (
            "$gt",
            10,
            "jsonb_typeof(value->%s) = 'number') AND (value->>%s::numeric > %s",
        ),
        (
            "$gte",
            10,
            "jsonb_typeof(value->%s) = 'number') AND (value->>%s::numeric >= %s",
        ),
        (
            "$lt",
            10,
            "jsonb_typeof(value->%s) = 'number') AND (value->>%s::numeric < %s",
        ),
        (
            "$lte",
            10,
            "jsonb_typeof(value->%s) = 'number') AND (value->>%s::numeric <= %s",
        ),
        # Float values must also use numeric cast
        (
            "$gt",
            3.14,
            "jsonb_typeof(value->%s) = 'number') AND (value->>%s::numeric > %s",
        ),
        (
            "$lt",
            2.718,
            "jsonb_typeof(value->%s) = 'number') AND (value->>%s::numeric < %s",
        ),
        # String values must NOT use CAST (text comparison is correct for strings)
        ("$gt", "apple", "value->>%s > %s"),
        ("$gte", "banana", "value->>%s >= %s"),
    ],
)
def test_filter_condition_sql_structure(
    store: _MockPostgresStore,
    op: str,
    value: Any,
    expected_sql_fragment: str,
) -> None:
    """Verify filter condition SQL uses correct operator and type for the value."""
    sql, params = store._get_filter_condition("score", op, value)
    assert expected_sql_fragment in sql
    # Numeric values passed as-is (not str()); string values passed as str()
    if isinstance(value, (int, float)):
        # key appears twice: once for jsonb_typeof check, once for CAST
        assert params.count("score") == 2
        assert value in params
    else:
        assert str(value) in params


@pytest.mark.parametrize(
    "op,value",
    [
        ("$gt", 10),
        ("$gte", 10),
        ("$lt", 10),
        ("$lte", 10),
        ("$gt", 3.14),
        ("$gte", 0.0),
        ("$lt", -5),
    ],
)
def test_numeric_filter_params_are_not_stringified(
    store: _MockPostgresStore, op: str, value: Any
) -> None:
    """Numeric values must be in params as Python numbers, not strings.

    Previously all values were passed as str(value), causing '9' >= '10' == True.
    After the fix, numeric comparisons use proper NUMERIC casting.
    """
    _sql, params = store._get_filter_condition("score", op, value)
    # The numeric value must be in params as-is (int/float), not as a string
    assert value in params, (
        f"Numeric value {value!r} should be in params, got {params!r}"
    )
    # Stringified form must NOT be present for numeric inputs
    assert str(value) not in params, (
        f"Stringified value {str(value)!r} should NOT be in params, got {params!r}"
    )


@pytest.mark.parametrize("op", ["$eq", "$ne"])
def test_equality_ops_unchanged(store: _MockPostgresStore, op: str) -> None:
    """Equality operators use jsonb equality and should be unaffected."""
    sql, params = store._get_filter_condition("name", op, "Alice")
    assert "value->" in sql
    assert "jsonb" in sql
    # json.dumps wraps strings in quotes, so check the quoted form
    assert '"Alice"' in params or "Alice" in params


@pytest.mark.parametrize("op", ["$invalid", "$foo", "$unknown_op", "$regex"])
def test_unsupported_operator_raises(store: _MockPostgresStore, op: str) -> None:
    """Unknown operators must raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported operator"):
        store._get_filter_condition("score", op, 10)
