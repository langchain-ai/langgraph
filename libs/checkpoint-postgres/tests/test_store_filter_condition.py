"""Unit tests for PostgresStore._get_filter_condition — no Postgres required."""
import pytest

from langgraph.store.postgres.base import BasePostgresStore


class _MockStore(BasePostgresStore):
    pass


store = object.__new__(_MockStore)


# Override the autouse conftest fixture so this file doesn't need a Postgres connection.
@pytest.fixture(autouse=True)
async def clear_test_db():
    pass


@pytest.mark.parametrize(
    "op,value,expected_sql,expected_cast",
    [
        ("$gt", 10, "(value->>%s)::numeric > %s", True),
        ("$gte", 10, "(value->>%s)::numeric >= %s", True),
        ("$lt", 10, "(value->>%s)::numeric < %s", True),
        ("$lte", 10, "(value->>%s)::numeric <= %s", True),
        ("$gt", 3.14, "(value->>%s)::numeric > %s", True),
        ("$gte", 3.14, "(value->>%s)::numeric >= %s", True),
        # String values should use text comparison (no cast)
        ("$gt", "b", "value->>%s > %s", False),
        ("$gte", "b", "value->>%s >= %s", False),
        ("$lt", "b", "value->>%s < %s", False),
        ("$lte", "b", "value->>%s <= %s", False),
    ],
)
def test_numeric_operators_use_numeric_cast(op, value, expected_sql, expected_cast):
    sql, params = store._get_filter_condition("score", op, value)
    assert sql == expected_sql, f"{op} with {type(value).__name__}: got {sql!r}"
    if expected_cast:
        # param should be the raw numeric value, not a stringified one
        assert params[1] == value
        assert isinstance(params[1], type(value))
    else:
        assert params[1] == str(value)


def test_numeric_cast_prevents_lexicographic_comparison():
    """9 > 10 lexicographically but not numerically — the cast must be present."""
    sql, params = store._get_filter_condition("score", "$gte", 10)
    assert "::numeric" in sql
    # If the value were stringified, '9' >= '10' would be True in Postgres text ordering.
    # Verify the param is numeric, not a string.
    assert params[1] == 10
    assert not isinstance(params[1], str)


def test_eq_and_ne_use_jsonb():
    sql_eq, _ = store._get_filter_condition("key", "$eq", "val")
    assert "::jsonb" in sql_eq
    sql_ne, _ = store._get_filter_condition("key", "$ne", "val")
    assert "::jsonb" in sql_ne


def test_unsupported_operator_raises():
    with pytest.raises(ValueError, match="Unsupported operator"):
        store._get_filter_condition("key", "$regex", ".*")
