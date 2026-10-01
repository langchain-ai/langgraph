from collections.abc import AsyncIterator
from typing import cast

import pytest

from langgraph.checkpoint.postgres.base import BasePostgresSaver


class _NoopConn:
    async def execute(self, _query: str) -> None:
        return None


@pytest.fixture
async def conn() -> AsyncIterator[_NoopConn]:
    yield _NoopConn()


def test_search_where_uses_config_context_for_checkpoint_only_before_cursor() -> None:
    predicate, params = BasePostgresSaver._search_where(
        cast(BasePostgresSaver, object()),
        {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}},
        {},
        {"configurable": {"checkpoint_id": "checkpoint-2"}},
    )

    assert "WHERE thread_id = %s" in predicate
    assert "NOT EXISTS (" in predicate
    assert "checkpoint->>'ts' < (" in predicate
    assert params == [
        "thread-1",
        "",
        "thread-1",
        "checkpoint-2",
        "thread-1",
        "checkpoint-2",
        "thread-1",
        "checkpoint-2",
        "checkpoint-2",
    ]
