"""Unit tests for PostgresStore._get_filter_condition numeric operators (issue #7684).

These tests run without a live database by calling the method directly on an
object instance created via object.__new__() -- no connection needed.
"""
import pytest

from langgraph.store.postgres.base import BasePostgresStore


def _store() -> BasePostgresStore:
    """Return a bare BasePostgresStore instance (no DB connection required)."""
    return object.__new__(BasePostgresStore)  # type: ignore[abstract]


class TestGetFilterConditionNumericOperators:
    """Numeric values must use (value->>key)::numeric so comparisons are arithmetic."""

    @pytest.mark.parametrize("op,expected_op", [
        ("$gt", ">"),
        ("$gte", ">="),
        ("$lt", "<"),
        ("$lte", "<="),
    ])
    def test_integer_value_uses_numeric_cast(self, op: str, expected_op: str) -> None:
        store = _store()
        sql, params = store._get_filter_condition("score", op, 10)
        assert "::numeric" in sql, (
            f"Expected numeric cast in SQL for {op} with int value, got: {sql!r}"
        )
        assert params == ["score", 10], f"Unexpected params: {params}"

    @pytest.mark.parametrize("op,expected_op", [
        ("$gt", ">"),
        ("$gte", ">="),
        ("$lt", "<"),
        ("$lte", "<="),
    ])
    def test_float_value_uses_numeric_cast(self, op: str, expected_op: str) -> None:
        store = _store()
        sql, params = store._get_filter_condition("score", op, 9.5)
        assert "::numeric" in sql, (
            f"Expected numeric cast in SQL for {op} with float value, got: {sql!r}"
        )
        assert params == ["score", 9.5]

    @pytest.mark.parametrize("op", ["$gt", "$gte", "$lt", "$lte"])
    def test_string_value_keeps_text_comparison(self, op: str) -> None:
        store = _store()
        sql, params = store._get_filter_condition("tag", op, "beta")
        assert "::numeric" not in sql, (
            f"String value must NOT get numeric cast for {op}, got: {sql!r}"
        )
        assert params == ["tag", "beta"]

    def test_numeric_param_is_not_stringified(self) -> None:
        """Regression: str(10) == "10" must NOT appear in params (causes text comparison)."""
        store = _store()
        sql, params = store._get_filter_condition("score", "$gte", 10)
        assert "10" not in params, (
            f"Numeric value was stringified in params: {params}; "
            "this causes lexicographic comparison ('9' >= '10' == True)"
        )
        assert 10 in params
