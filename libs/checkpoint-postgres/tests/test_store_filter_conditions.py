from typing import Any

import pytest

from langgraph.store.postgres.base import BasePostgresStore


@pytest.fixture(autouse=True)
def clear_test_db() -> None:
    pass


def test_numeric_filter_conditions_use_numeric_comparison() -> None:
    store = object.__new__(BasePostgresStore)

    cases = [
        ("$gt", ">"),
        ("$gte", ">="),
        ("$lt", "<"),
        ("$lte", "<="),
    ]
    for op, sql_op in cases:
        sql, params = store._get_filter_condition("score", op, 10)
        assert sql == (
            "CASE WHEN jsonb_typeof(value->%s) = 'number' "
            f"THEN (value->>%s)::double precision {sql_op} %s "
            "ELSE false END"
        )
        assert params == ["score", "score", 10.0]


@pytest.mark.parametrize("value", ["10", True])
def test_non_numeric_filter_conditions_keep_text_comparison(value: Any) -> None:
    store = object.__new__(BasePostgresStore)

    sql, params = store._get_filter_condition("score", "$gte", value)

    assert sql == "value->>%s >= %s"
    assert params == ["score", str(value)]
