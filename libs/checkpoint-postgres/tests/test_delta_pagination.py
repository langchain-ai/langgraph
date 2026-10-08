from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import (
    Checkpoint,
    DeltaChannelHistory,
    empty_checkpoint,
)
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import _DeltaSnapshot

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres.base import _DELTA_PAGE_SIZE, BasePostgresSaver
from tests.conftest import DEFAULT_URI

CHANNEL = "items"
STEPS = 8
SEED_STEP = 1
SEED_VALUE = [10, 20]
TARGET_STEP = 4

# The real page size is the control; the rest split the walk from the target
# to its seed across pages.
PAGE_SIZES = [_DELTA_PAGE_SIZE, 3, 2, 1]


def _step_args(
    thread_id: str, step: int, parent: dict | None
) -> tuple[dict, Checkpoint, dict[str, Any]]:
    config: dict = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    if parent is not None:
        config["configurable"]["checkpoint_id"] = parent["configurable"][
            "checkpoint_id"
        ]
    checkpoint: Checkpoint = empty_checkpoint()
    checkpoint["id"] = str(uuid6(clock_seq=step))
    checkpoint["channel_versions"][CHANNEL] = f"v{step}"
    if step == SEED_STEP:
        checkpoint["channel_values"][CHANNEL] = _DeltaSnapshot(list(SEED_VALUE))
        return config, checkpoint, {CHANNEL: f"v{step}"}
    return config, checkpoint, {}


async def _abuild_chain(saver: AsyncPostgresSaver) -> list[dict]:
    thread_id = str(uuid4())
    parent: dict | None = None
    configs: list[dict] = []
    for step in range(STEPS):
        config, checkpoint, new_versions = _step_args(thread_id, step, parent)
        parent = await saver.aput(
            config,
            checkpoint,
            {"source": "loop", "step": step, "parents": {}},
            new_versions,
        )
        await saver.aput_writes(parent, [(CHANNEL, f"w{step}")], str(uuid4()))
        configs.append(parent)
    return configs


def _build_chain(saver: PostgresSaver) -> list[dict]:
    thread_id = str(uuid4())
    parent: dict | None = None
    configs: list[dict] = []
    for step in range(STEPS):
        config, checkpoint, new_versions = _step_args(thread_id, step, parent)
        parent = saver.put(
            config,
            checkpoint,
            {"source": "loop", "step": step, "parents": {}},
            new_versions,
        )
        saver.put_writes(parent, [(CHANNEL, f"w{step}")], str(uuid4()))
        configs.append(parent)
    return configs


def _assert_history(entry: DeltaChannelHistory, page_size: int) -> None:
    seed = entry.get("seed")
    assert isinstance(seed, _DeltaSnapshot), (
        f"page_size={page_size}: expected a snapshot seed, "
        f"got {entry.get('seed', '<missing>')!r}"
    )
    assert seed.value == SEED_VALUE
    assert [w[2] for w in entry["writes"]] == ["w1", "w2", "w3"], (
        f"page_size={page_size}: got {[w[2] for w in entry['writes']]}"
    )


@pytest.mark.parametrize("page_size", PAGE_SIZES)
async def test_async_walk_continues_across_pages(
    page_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("langgraph.checkpoint.postgres.aio._DELTA_PAGE_SIZE", page_size)
    async with AsyncPostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        await saver.setup()
        configs = await _abuild_chain(saver)
        result = await saver.aget_delta_channel_history(
            config=configs[TARGET_STEP], channels=[CHANNEL]
        )
        _assert_history(result[CHANNEL], page_size)


@pytest.mark.parametrize("page_size", PAGE_SIZES)
def test_sync_walk_continues_across_pages(
    page_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("langgraph.checkpoint.postgres._DELTA_PAGE_SIZE", page_size)
    with PostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        saver.setup()
        configs = _build_chain(saver)
        result = saver.get_delta_channel_history(
            config=configs[TARGET_STEP], channels=[CHANNEL]
        )
        _assert_history(result[CHANNEL], page_size)


def _record_rows_read(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    read: list[str] = []
    ingest = BasePostgresSaver._ingest_stage1_page

    def record(rows: Sequence[Mapping[str, Any]], *args: Any) -> str | None:
        read.extend(row["checkpoint_id"] for row in rows)
        return ingest(rows, *args)

    monkeypatch.setattr(BasePostgresSaver, "_ingest_stage1_page", staticmethod(record))
    return read


def _ids_from_target_down(configs: list[dict]) -> list[str]:
    return [c["configurable"]["checkpoint_id"] for c in configs[TARGET_STEP::-1]]


async def test_async_walk_reads_nothing_newer_than_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read = _record_rows_read(monkeypatch)
    async with AsyncPostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        await saver.setup()
        configs = await _abuild_chain(saver)
        result = await saver.aget_delta_channel_history(
            config=configs[TARGET_STEP], channels=[CHANNEL]
        )
    _assert_history(result[CHANNEL], _DELTA_PAGE_SIZE)
    assert read == _ids_from_target_down(configs)


def test_sync_walk_reads_nothing_newer_than_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read = _record_rows_read(monkeypatch)
    with PostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        saver.setup()
        configs = _build_chain(saver)
        result = saver.get_delta_channel_history(
            config=configs[TARGET_STEP], channels=[CHANNEL]
        )
    _assert_history(result[CHANNEL], _DELTA_PAGE_SIZE)
    assert read == _ids_from_target_down(configs)


async def test_root_target_has_no_history_and_still_terminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("langgraph.checkpoint.postgres.aio._DELTA_PAGE_SIZE", 1)
    async with AsyncPostgresSaver.from_conn_string(DEFAULT_URI) as saver:
        await saver.setup()
        configs = await _abuild_chain(saver)
        result = await saver.aget_delta_channel_history(
            config=configs[0], channels=[CHANNEL]
        )
        assert result[CHANNEL] == {"writes": []}
