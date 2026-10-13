import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.conformance import validate
from langgraph.checkpoint.conformance.initializer import checkpointer_test
from langgraph.checkpoint.serde.types import TASKS, _DeltaSnapshot
from psycopg import AsyncConnection

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres.base import (
    PRUNE_DELTA_THREADS_SQL,
    PRUNE_HEADS_SQL,
    PRUNE_REFERENCED_BLOBS_SQL,
)
from tests.conftest import DEFAULT_POSTGRES_URI, DEFAULT_URI
from tests.test_async import _saver as _async_saver
from tests.test_sync import _saver as _sync_saver

SAVERS = ["base", "pool", "pipe", "aio_base", "aio_pool", "aio_pipe"]


@asynccontextmanager
async def _saver(name: str):
    if name.startswith("aio_"):
        async with _async_saver(name.removeprefix("aio_")) as saver:
            yield saver
    else:
        with _sync_saver(name) as saver:
            yield saver


async def _call(saver: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    if isinstance(saver, AsyncPostgresSaver):
        return await getattr(saver, f"a{method}")(*args, **kwargs)
    return getattr(saver, method)(*args, **kwargs)


async def _rows(saver: Any, thread_id: str) -> dict[str, list[tuple]]:
    queries = {
        "checkpoints": "SELECT checkpoint_ns, checkpoint_id FROM checkpoints"
        " WHERE thread_id = %s ORDER BY 1, 2",
        "writes": "SELECT checkpoint_ns, checkpoint_id, task_id, idx"
        " FROM checkpoint_writes WHERE thread_id = %s ORDER BY 1, 2, 3, 4",
        "blobs": "SELECT checkpoint_ns, channel, version FROM checkpoint_blobs"
        " WHERE thread_id = %s ORDER BY 1, 2, 3",
    }
    rows = {}
    for table, sql in queries.items():
        if isinstance(saver, AsyncPostgresSaver):
            async with saver._cursor() as cur:
                await cur.execute(sql, (thread_id,))
                rows[table] = [tuple(r.values()) for r in await cur.fetchall()]
        else:
            with saver._cursor() as cur:
                cur.execute(sql, (thread_id,))
                rows[table] = [tuple(r.values()) for r in cur.fetchall()]
    return rows


async def _put(
    saver: Any,
    thread_id: str,
    parent: dict | None,
    step: int,
    values: dict[str, Any],
    versions: dict[str, str],
    *,
    ns: str = "",
    metadata: dict[str, Any] | None = None,
    v: int = 4,
) -> dict:
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns}}
    if parent is not None:
        config["configurable"]["checkpoint_id"] = parent["configurable"][
            "checkpoint_id"
        ]
    checkpoint = empty_checkpoint()
    checkpoint.update(
        v=v,
        id=str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=versions,
    )
    new_versions = {k: versions[k] for k in values}
    return await _call(
        saver,
        "put",
        config,
        checkpoint,
        {"source": "loop", "step": step, "parents": {}, **(metadata or {})},
        new_versions,
    )


async def _three_checkpoints(saver: Any, thread_id: str, ns: str = "") -> dict:
    parent = None
    steps = [
        ({"a": [1], "s": ["shared"]}, {"a": "1", "s": "1"}),
        ({"a": [1, 2]}, {"a": "2", "s": "1"}),
        ({"a": [1, 2, 3]}, {"a": "3", "s": "1"}),
    ]
    for step, (values, versions) in enumerate(steps):
        parent = await _put(saver, thread_id, parent, step, values, versions, ns=ns)
        await _call(saver, "put_writes", parent, [("a", step)], str(uuid4()))
    return parent


@asynccontextmanager
async def _second_saver(saver: AsyncPostgresSaver):
    assert isinstance(saver.conn, AsyncConnection)
    uri = DEFAULT_POSTGRES_URI + saver.conn.info.dbname
    async with AsyncPostgresSaver.from_conn_string(uri) as other:
        yield other


def _run_once_after(
    saver: AsyncPostgresSaver, sql: str, action: Callable[[], Awaitable[Any]]
) -> None:
    pending = [action]
    cursor = saver._cursor

    class Hooked:
        def __init__(self, cur: Any) -> None:
            self.cur = cur

        def __getattr__(self, name: str) -> Any:
            return getattr(self.cur, name)

        async def execute(self, query: Any, params: Any = None, **kwargs: Any) -> None:
            await self.cur.execute(query, params, **kwargs)
            if query is sql and pending:
                await pending.pop()()

    @asynccontextmanager
    async def hooked(**kwargs: Any):
        async with cursor(**kwargs) as cur:
            yield Hooked(cur)

    saver._cursor = hooked


async def _delta_thread(saver: Any, thread_id: str, evidence: str) -> None:
    if evidence == "counters":
        first = await _put(saver, thread_id, None, 0, {"m": [1]}, {"m": "1"})
        await _put(
            saver,
            thread_id,
            first,
            1,
            {},
            {"m": "2"},
            metadata={"counters_since_delta_snapshot": {"m": [1, 1]}},
        )
    else:
        first = await _put(
            saver, thread_id, None, 0, {"m": _DeltaSnapshot([1])}, {"m": "1"}
        )
        await _put(saver, thread_id, first, 1, {}, {"m": "2"})


@pytest.mark.parametrize("name", SAVERS)
async def test_keep_latest_keeps_each_namespace_head_with_its_writes_and_blobs(
    name: str,
) -> None:
    async with _saver(name) as saver:
        thread_id, other_id = str(uuid4()), str(uuid4())
        heads = {
            ns: await _three_checkpoints(saver, thread_id, ns) for ns in ("", "sub:1")
        }
        await _three_checkpoints(saver, other_id)
        before = {ns: await _call(saver, "get_tuple", h) for ns, h in heads.items()}
        other_before = await _rows(saver, other_id)

        await _call(saver, "prune", [thread_id])

        rows = await _rows(saver, thread_id)
        assert rows["checkpoints"] == [
            (ns, heads[ns]["configurable"]["checkpoint_id"]) for ns in ("", "sub:1")
        ]
        assert [(ns, cid) for ns, cid, *_ in rows["writes"]] == rows["checkpoints"]
        assert rows["blobs"] == [
            (ns, ch, ver)
            for ns in ("", "sub:1")
            for ch, ver in (("a", "3"), ("s", "1"))
        ]
        for ns, head in heads.items():
            assert await _call(saver, "get_tuple", head) == before[ns]
        assert await _rows(saver, other_id) == other_before


@pytest.mark.parametrize("evidence", ["counters", "snapshot"])
@pytest.mark.parametrize("name", SAVERS)
async def test_keep_latest_refuses_a_batch_with_a_delta_channel_thread(
    name: str, evidence: str
) -> None:
    async with _saver(name) as saver:
        ordinary_id, delta_id = str(uuid4()), str(uuid4())
        await _three_checkpoints(saver, ordinary_id)
        await _delta_thread(saver, delta_id, evidence)
        before = [await _rows(saver, t) for t in (ordinary_id, delta_id)]

        with pytest.raises(NotImplementedError, match=delta_id):
            await _call(saver, "prune", [ordinary_id, delta_id])

        assert [await _rows(saver, t) for t in (ordinary_id, delta_id)] == before


@pytest.mark.parametrize("name", SAVERS)
async def test_keep_latest_prunes_a_thread_storing_a_plain_true(name: str) -> None:
    async with _saver(name) as saver:
        thread_id = str(uuid4())
        first = await _put(
            saver, thread_id, None, 0, {"flag": True, "a": [1]}, {"flag": "1", "a": "1"}
        )
        head = await _put(
            saver, thread_id, first, 1, {"flag": True}, {"flag": "2", "a": "1"}
        )

        await _call(saver, "prune", [thread_id])

        assert (await _rows(saver, thread_id))["checkpoints"] == [
            ("", head["configurable"]["checkpoint_id"])
        ]


@pytest.mark.parametrize("name", SAVERS)
async def test_keep_latest_keeps_the_parent_holding_a_pre_v4_heads_sends(
    name: str,
) -> None:
    async with _saver(name) as saver:
        thread_id = str(uuid4())
        first = await _put(saver, thread_id, None, 0, {}, {}, v=1)
        await _call(saver, "put_writes", first, [(TASKS, "send-1")], str(uuid4()))
        second = await _put(saver, thread_id, first, 1, {}, {}, v=1)
        head = await _put(saver, thread_id, second, 2, {}, {}, v=1)
        await _call(saver, "put_writes", second, [(TASKS, "send-2")], str(uuid4()))

        await _call(saver, "prune", [thread_id])

        assert (await _rows(saver, thread_id))["checkpoints"] == [
            ("", second["configurable"]["checkpoint_id"]),
            ("", head["configurable"]["checkpoint_id"]),
        ]
        loaded = await _call(saver, "get_tuple", head)
        assert loaded.checkpoint["channel_values"][TASKS] == ["send-2"]


@pytest.mark.parametrize("name", SAVERS)
async def test_delete_removes_every_row_of_the_threads(name: str) -> None:
    async with _saver(name) as saver:
        ordinary_id, delta_id, other_id = str(uuid4()), str(uuid4()), str(uuid4())
        await _three_checkpoints(saver, ordinary_id)
        await _delta_thread(saver, delta_id, "counters")
        await _three_checkpoints(saver, other_id)
        other_before = await _rows(saver, other_id)

        await _call(saver, "prune", [ordinary_id, delta_id], strategy="delete")

        empty = {"checkpoints": [], "writes": [], "blobs": []}
        assert await _rows(saver, ordinary_id) == empty
        assert await _rows(saver, delta_id) == empty
        assert await _rows(saver, other_id) == other_before


@pytest.mark.parametrize("name", SAVERS)
async def test_unknown_strategy_changes_nothing(name: str) -> None:
    async with _saver(name) as saver:
        thread_id = str(uuid4())
        await _three_checkpoints(saver, thread_id)
        before = await _rows(saver, thread_id)

        with pytest.raises(ValueError, match="keep_all"):
            await _call(saver, "prune", [thread_id], strategy="keep_all")

        assert await _rows(saver, thread_id) == before


async def test_async_saver_prune_runs_from_another_thread_only() -> None:
    async with _saver("aio_base") as saver:
        thread_id = str(uuid4())
        head = await _three_checkpoints(saver, thread_id)

        with pytest.raises(asyncio.InvalidStateError):
            saver.prune([thread_id])
        await asyncio.to_thread(saver.prune, [thread_id])

        assert (await _rows(saver, thread_id))["checkpoints"] == [
            ("", head["configurable"]["checkpoint_id"])
        ]


async def test_prune_conformance() -> None:
    @checkpointer_test(name="AsyncPostgresSaver")
    async def postgres_saver():
        async with AsyncPostgresSaver.from_conn_string(DEFAULT_URI) as saver:
            await saver.setup()
            yield saver

    report = await validate(postgres_saver, capabilities={"prune"})

    prune = report.results["prune"]
    if not prune.passed:
        details = "\n".join(prune.failures)
        pytest.fail(f"Capability prune failed (passed={prune.passed}):\n{details}")


@pytest.mark.parametrize("strategy", ["keep_latest", "delete"])
@pytest.mark.parametrize("name", ["base", "aio_base"])
async def test_prune_rejects_a_single_thread_id_string(
    name: str, strategy: str
) -> None:
    async with _saver(name) as saver:
        for thread_id in ("ab", "a", "b"):
            await _three_checkpoints(saver, thread_id)
        before = [await _rows(saver, t) for t in ("ab", "a", "b")]

        with pytest.raises(TypeError, match="not a str"):
            await _call(saver, "prune", "ab", strategy=strategy)

        assert [await _rows(saver, t) for t in ("ab", "a", "b")] == before


@pytest.mark.parametrize(
    "after", [PRUNE_HEADS_SQL, PRUNE_REFERENCED_BLOBS_SQL], ids=["heads", "blobs"]
)
async def test_keep_latest_keeps_a_checkpoint_saved_while_it_runs(after: str) -> None:
    async with _saver("aio_base") as saver, _second_saver(saver) as other:
        thread_id = str(uuid4())
        head = await _three_checkpoints(saver, thread_id)
        saved: list[dict] = []

        async def save_a_child() -> None:
            saved.append(
                await _put(
                    other, thread_id, head, 3, {"a": [1, 2, 3, 4]}, {"a": "4", "s": "1"}
                )
            )

        _run_once_after(saver, after, save_a_child)
        await saver.aprune([thread_id])

        loaded = await saver.aget_tuple(saved[0])
        assert loaded is not None, "the checkpoint saved during the prune was deleted"
        assert loaded.checkpoint["channel_values"] == {
            "a": [1, 2, 3, 4],
            "s": ["shared"],
        }


async def test_keep_latest_refuses_a_thread_that_turns_delta_after_the_batch_check() -> (
    None
):
    async with _saver("aio_base") as saver, _second_saver(saver) as other:
        thread_id = str(uuid4())
        head = await _three_checkpoints(saver, thread_id)

        async def save_a_delta_checkpoint() -> None:
            await _put(
                other,
                thread_id,
                head,
                3,
                {},
                {"a": "4", "s": "1"},
                metadata={"counters_since_delta_snapshot": {"a": [1, 1]}},
            )

        _run_once_after(saver, PRUNE_DELTA_THREADS_SQL, save_a_delta_checkpoint)
        with pytest.raises(NotImplementedError, match=thread_id):
            await saver.aprune([thread_id])

        assert len((await _rows(saver, thread_id))["checkpoints"]) == 4
