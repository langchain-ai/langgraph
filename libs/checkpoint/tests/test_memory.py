import logging
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from langgraph.checkpoint.base import (
    DELTA_SENTINEL,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import (
    JsonPlusSerializer,
    _warned_blocked_types,
    _warned_unregistered_types,
)


class MemoryPydantic(BaseModel):
    foo: str


@pytest.fixture(autouse=True)
def _reset_warned_types() -> None:
    # Warning dedup state is process-global; reset per-test so each case sees
    # a fresh slate and assertions about warning emission are stable.
    _warned_unregistered_types.clear()
    _warned_blocked_types.clear()


class TestMemorySaver:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.memory_saver = InMemorySaver()

        # objects for test setup
        self.config_1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "1",
            }
        }
        self.config_2: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "checkpoint_id": "2",
            }
        }
        self.config_3: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2-inner",
                "checkpoint_ns": "inner",
            }
        }

        self.chkpnt_1: Checkpoint = empty_checkpoint()
        self.chkpnt_2: Checkpoint = create_checkpoint(self.chkpnt_1, {}, 1)
        self.chkpnt_3: Checkpoint = empty_checkpoint()

        self.metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 2,
            "writes": {},
            "score": 1,
        }
        self.metadata_2: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
            "writes": {"foo": "bar"},
            "score": None,
        }
        self.metadata_3: CheckpointMetadata = {}

    def test_combined_metadata(self) -> None:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "__super_private_key": "super_private_value",
            },
            "metadata": {"run_id": "my_run_id"},
        }
        self.memory_saver.put(
            config, self.chkpnt_2, self.metadata_2, self.chkpnt_2["channel_versions"]
        )
        checkpoint = self.memory_saver.get_tuple(config)
        assert checkpoint is not None
        assert checkpoint.metadata == {
            **self.metadata_2,
            "run_id": "my_run_id",
        }

    async def test_search(self) -> None:
        # set up test
        # save checkpoints
        self.memory_saver.put(
            self.config_1,
            self.chkpnt_1,
            self.metadata_1,
            self.chkpnt_1["channel_versions"],
        )
        self.memory_saver.put(
            self.config_2,
            self.chkpnt_2,
            self.metadata_2,
            self.chkpnt_2["channel_versions"],
        )
        self.memory_saver.put(
            self.config_3,
            self.chkpnt_3,
            self.metadata_3,
            self.chkpnt_3["channel_versions"],
        )

        # call method / assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
        query_4 = {"source": "update", "step": 1}  # no match

        search_results_1 = list(self.memory_saver.list(None, filter=query_1))
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == self.metadata_1

        search_results_2 = list(self.memory_saver.list(None, filter=query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == self.metadata_2

        search_results_3 = list(self.memory_saver.list(None, filter=query_3))
        assert len(search_results_3) == 3

        search_results_4 = list(self.memory_saver.list(None, filter=query_4))
        assert len(search_results_4) == 0

        # search by config (defaults to checkpoints across all namespaces)
        search_results_5 = list(
            self.memory_saver.list({"configurable": {"thread_id": "thread-2"}})
        )
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}

        # TODO: test before and limit params

    async def test_asearch(self) -> None:
        # set up test
        # save checkpoints
        self.memory_saver.put(
            self.config_1,
            self.chkpnt_1,
            self.metadata_1,
            self.chkpnt_1["channel_versions"],
        )
        self.memory_saver.put(
            self.config_2,
            self.chkpnt_2,
            self.metadata_2,
            self.chkpnt_2["channel_versions"],
        )
        self.memory_saver.put(
            self.config_3,
            self.chkpnt_3,
            self.metadata_3,
            self.chkpnt_3["channel_versions"],
        )

        # call method / assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
        query_4 = {"source": "update", "step": 1}  # no match

        search_results_1 = [
            c async for c in self.memory_saver.alist(None, filter=query_1)
        ]
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == self.metadata_1

        search_results_2 = [
            c async for c in self.memory_saver.alist(None, filter=query_2)
        ]
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == self.metadata_2

        search_results_3 = [
            c async for c in self.memory_saver.alist(None, filter=query_3)
        ]
        assert len(search_results_3) == 3

        search_results_4 = [
            c async for c in self.memory_saver.alist(None, filter=query_4)
        ]
        assert len(search_results_4) == 0


async def test_memory_saver() -> None:
    memory_saver = InMemorySaver()
    assert isinstance(memory_saver, InMemorySaver)

    async with memory_saver as async_memory_saver:
        assert async_memory_saver is memory_saver

    with memory_saver as sync_memory_saver:
        assert sync_memory_saver is memory_saver


def test_memory_saver_warns_on_unregistered_msgpack(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = JsonPlusSerializer()
    memory_saver = InMemorySaver(serde=serde)
    obj = MemoryPydantic(foo="bar")

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"foo": obj}
    checkpoint["channel_versions"] = {"foo": 1}

    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }

    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    new_config = memory_saver.put(config, checkpoint, {}, {"foo": 1})
    result = memory_saver.get_tuple(new_config)

    assert result is not None
    assert "unregistered type" in caplog.text.lower()
    assert result.checkpoint["channel_values"]["foo"] == obj


def test_memory_saver_allowlist_silences_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("tests.test_memory", "MemoryPydantic")]
    )
    memory_saver = InMemorySaver(serde=serde)
    obj = MemoryPydantic(foo="bar")

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"foo": obj}
    checkpoint["channel_versions"] = {"foo": 1}

    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }

    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    new_config = memory_saver.put(config, checkpoint, {}, {"foo": 1})
    result = memory_saver.get_tuple(new_config)

    assert result is not None
    assert "unregistered type" not in caplog.text.lower()
    assert result.checkpoint["channel_values"]["foo"] == obj


def test_memory_saver_strict_blocks_unregistered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    memory_saver = InMemorySaver(serde=serde)
    obj = MemoryPydantic(foo="bar")

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"foo": obj}
    checkpoint["channel_versions"] = {"foo": 1}

    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }

    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    new_config = memory_saver.put(config, checkpoint, {}, {"foo": 1})
    result = memory_saver.get_tuple(new_config)

    assert result is not None
    assert "blocked" in caplog.text.lower()
    expected = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
    assert result.checkpoint["channel_values"]["foo"] == expected


def test_memory_saver_with_allowlist_proxy_isolated() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    memory_saver = InMemorySaver(serde=serde)
    proxy = memory_saver.with_allowlist([("tests.test_memory", "MemoryPydantic")])

    obj = MemoryPydantic(foo="bar")

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"foo": obj}
    checkpoint["channel_versions"] = {"foo": 1}

    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }

    new_config = proxy.put(config, checkpoint, {}, {"foo": 1})

    proxied = proxy.get_tuple(new_config)
    assert proxied is not None
    assert proxied.checkpoint["channel_values"]["foo"] == obj

    direct = memory_saver.get_tuple(new_config)
    assert direct is not None
    expected = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
    assert direct.checkpoint["channel_values"]["foo"] == expected


class TestInMemorySaverDeltaChannel:
    def test_load_blobs_returns_sentinel_for_delta_channel(self) -> None:
        """_load_blobs returns DELTA_SENTINEL for delta channels (reconstruction deferred)."""
        saver = InMemorySaver()
        serde = JsonPlusSerializer()

        thread_id, ns, channel = "t1", "", "messages"
        v1 = "00000000000000000000000000000001.0000000000000000"

        saver.blobs[(thread_id, ns, channel, v1)] = serde.dumps_typed(DELTA_SENTINEL)

        cp1 = empty_checkpoint()
        cp1["id"] = "cp1"
        cp1["channel_versions"][channel] = v1
        saver.storage[thread_id][ns] = {
            "cp1": (serde.dumps_typed(cp1), serde.dumps_typed({}), None),
        }

        result = saver._load_blobs(thread_id, ns, {channel: v1})
        assert channel in result
        assert result[channel] is DELTA_SENTINEL

    def test_get_channel_writes_collects_ancestor_writes_only(self) -> None:
        """_get_channel_writes_history collects ancestor writes oldest→newest,
        and excludes writes stored at the target checkpoint itself (those are
        pending writes for the next step, applied separately by pregel)."""
        saver = InMemorySaver()
        serde = JsonPlusSerializer()

        thread_id, ns, channel = "t1", "", "messages"

        cp1 = empty_checkpoint()
        cp1["id"] = "cp1"
        cp2 = empty_checkpoint()
        cp2["id"] = "cp2"
        saver.storage[thread_id][ns] = {
            "cp1": (serde.dumps_typed(cp1), serde.dumps_typed({}), None),
            "cp2": (serde.dumps_typed(cp2), serde.dumps_typed({}), "cp1"),
        }
        # Writes stored at cp1 produced the cp1 snapshot; part of history.
        saver.writes[(thread_id, ns, "cp1")][("task1", 0)] = (
            "task1",
            channel,
            serde.dumps_typed({"content": "hi"}),
            "",
        )
        # Writes stored at cp2 are pending — they will produce cp3 when the
        # step that loaded cp2 completes. They MUST NOT appear in the
        # reconstructed snapshot value at cp2.
        saver.writes[(thread_id, ns, "cp2")][("task2", 0)] = (
            "task2",
            channel,
            serde.dumps_typed({"content": "pending"}),
            "",
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": "cp2",
            }
        }
        result = saver._get_channel_writes_history(config, channel)
        assert result.seed is DELTA_SENTINEL
        values = [v for _, _, v in result.writes]
        assert values == [{"content": "hi"}]

    def test_get_channel_writes_at_root_returns_empty(self) -> None:
        """Reconstructing the root checkpoint's state: no ancestors → []."""
        saver = InMemorySaver()
        serde = JsonPlusSerializer()
        thread_id, ns, channel = "t1", "", "messages"

        cp1 = empty_checkpoint()
        cp1["id"] = "cp1"
        saver.storage[thread_id][ns] = {
            "cp1": (serde.dumps_typed(cp1), serde.dumps_typed({}), None),
        }
        saver.writes[(thread_id, ns, "cp1")][("task1", 0)] = (
            "task1",
            channel,
            serde.dumps_typed({"content": "pending"}),
            "",
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": "cp1",
            }
        }
        result = saver._get_channel_writes_history(config, channel)
        assert result.seed is DELTA_SENTINEL
        assert result.writes == []


class TestBaseFallbackGetChannelWrites:
    """Exercises the `BaseCheckpointSaver._get_channel_writes_history` default
    implementation — the path third-party savers inherit when they don't
    override `_get_channel_writes_history` themselves.

    Regression guard for a bug where the fallback passed the caller's config
    (with `checkpoint_id`) straight to `self.list()`, which most savers
    collapse to a single row — causing the fallback to return `[]`.
    """

    def _build_saver_with_chain(self) -> tuple[InMemorySaver, str, str]:
        """Build an InMemorySaver with a 3-checkpoint chain and per-step writes
        for a `messages` channel.

        Returns `(saver, thread_id, namespace)`. The saver subclass deletes the
        InMemorySaver override so the base class fallback is exercised.
        """

        class _ThirdPartyStyleSaver(InMemorySaver):
            _get_channel_writes_history = (
                InMemorySaver.__mro__[1]._get_channel_writes_history  # type: ignore[attr-defined]
            )
            _aget_channel_writes_history = (
                InMemorySaver.__mro__[1]._aget_channel_writes_history  # type: ignore[attr-defined]
            )

        saver = _ThirdPartyStyleSaver()
        serde = JsonPlusSerializer()
        thread_id, ns, channel = "t1", "", "messages"

        cp0 = empty_checkpoint()
        cp0["id"] = "00000000000000000000000000000001.0000000000000000"
        cp1 = empty_checkpoint()
        cp1["id"] = "00000000000000000000000000000002.0000000000000000"
        cp2 = empty_checkpoint()
        cp2["id"] = "00000000000000000000000000000003.0000000000000000"
        saver.storage[thread_id][ns] = {
            cp0["id"]: (serde.dumps_typed(cp0), serde.dumps_typed({}), None),
            cp1["id"]: (serde.dumps_typed(cp1), serde.dumps_typed({}), cp0["id"]),
            cp2["id"]: (serde.dumps_typed(cp2), serde.dumps_typed({}), cp1["id"]),
        }
        # Writes under cp0 produced cp1's state; writes under cp1 produced cp2's.
        saver.writes[(thread_id, ns, cp0["id"])][("task1", 0)] = (
            "task1",
            channel,
            serde.dumps_typed({"content": "first"}),
            "",
        )
        saver.writes[(thread_id, ns, cp1["id"])][("task2", 0)] = (
            "task2",
            channel,
            serde.dumps_typed({"content": "second"}),
            "",
        )
        return saver, thread_id, ns

    def test_fallback_returns_ancestor_writes_oldest_first(self) -> None:
        saver, thread_id, ns = self._build_saver_with_chain()
        target_id = "00000000000000000000000000000003.0000000000000000"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": target_id,
            }
        }

        result = saver._get_channel_writes_history(config, "messages")

        assert result.seed is DELTA_SENTINEL
        values = [v for _, _, v in result.writes]
        assert values == [{"content": "first"}, {"content": "second"}]

    async def test_async_fallback_returns_ancestor_writes_oldest_first(self) -> None:
        saver, thread_id, ns = self._build_saver_with_chain()
        target_id = "00000000000000000000000000000003.0000000000000000"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": target_id,
            }
        }

        result = await saver._aget_channel_writes_history(config, "messages")

        assert result.seed is DELTA_SENTINEL
        values = [v for _, _, v in result.writes]
        assert values == [{"content": "first"}, {"content": "second"}]

    async def test_async_fallback_concurrent_tasks_do_not_interfere(self) -> None:
        """Regression: the re-entrancy guard must be task-local, not thread-local.

        Two concurrent `_aget_channel_writes_history` calls on the same
        event-loop thread must each see their full reconstructed writes. A
        `threading.local()` guard would let whichever task set it first
        short-circuit the other to `writes=[]`.
        """
        import asyncio

        saver, thread_id, ns = self._build_saver_with_chain()

        # Force the two tasks to interleave across the `set(True)` boundary:
        # each `aget_tuple` yields control, so if the guard were thread-local
        # the second task would observe `active=True` set by the first.
        orig_aget_tuple = saver.aget_tuple

        async def slow_aget_tuple(config: RunnableConfig) -> Any:
            await asyncio.sleep(0)
            return await orig_aget_tuple(config)

        saver.aget_tuple = slow_aget_tuple  # type: ignore[method-assign]

        target_id = "00000000000000000000000000000003.0000000000000000"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": target_id,
            }
        }

        results = await asyncio.gather(
            saver._aget_channel_writes_history(config, "messages"),
            saver._aget_channel_writes_history(config, "messages"),
        )

        expected_values = [{"content": "first"}, {"content": "second"}]
        for result in results:
            assert result.seed is DELTA_SENTINEL
            values = [v for _, _, v in result.writes]
            assert values == expected_values


class TestPreDeltaBlobTerminator:
    """Verify the pre-delta blob terminator: when the ancestor walk hits a
    checkpoint whose blob for the channel is a real value (not
    DELTA_SENTINEL), reconstruction seeds from it and stops. This guards

      * back-compat: a thread written by pre-delta code, then extended under
        delta — reconstruction must return the correct value without walking
        past the last pre-delta ancestor;
      * perf: without the terminator, every reconstruct-after-migration would
        walk all the way to the thread root.
    """

    def _build_mixed_thread(self) -> tuple[InMemorySaver, str, str, str, str]:
        """Three-checkpoint chain: cp1 (pre-delta, blob=[A]), cp2 (delta,
        write=B), cp3 (delta, write=C). Reconstructing at cp3 must yield
        seed=[A] + writes=[B, C].

        Returns `(saver, thread_id, ns, channel, cp3_id)`.
        """
        saver = InMemorySaver()
        serde = JsonPlusSerializer()
        thread_id, ns, channel = "t1", "", "messages"

        v1 = "00000000000000000000000000000001.0"
        v2 = "00000000000000000000000000000002.0"
        v3 = "00000000000000000000000000000003.0"

        # Pre-delta: cp1 stored a real blob for the channel.
        saver.blobs[(thread_id, ns, channel, v1)] = serde.dumps_typed(["A"])
        # Delta-era: cp2 and cp3 store sentinels; real writes in checkpoint_writes.
        saver.blobs[(thread_id, ns, channel, v2)] = serde.dumps_typed(DELTA_SENTINEL)
        saver.blobs[(thread_id, ns, channel, v3)] = serde.dumps_typed(DELTA_SENTINEL)

        cp1 = empty_checkpoint()
        cp1["id"] = "cp1"
        cp1["channel_versions"][channel] = v1
        cp2 = empty_checkpoint()
        cp2["id"] = "cp2"
        cp2["channel_versions"][channel] = v2
        cp3 = empty_checkpoint()
        cp3["id"] = "cp3"
        cp3["channel_versions"][channel] = v3

        saver.storage[thread_id][ns] = {
            "cp1": (serde.dumps_typed(cp1), serde.dumps_typed({}), None),
            "cp2": (serde.dumps_typed(cp2), serde.dumps_typed({}), "cp1"),
            "cp3": (serde.dumps_typed(cp3), serde.dumps_typed({}), "cp2"),
        }
        # Write under cp1 would be from the pre-delta era and MUST be ignored
        # (the blob already captures it). We add one and assert it is not
        # folded into the reconstructed result.
        saver.writes[(thread_id, ns, "cp1")][("task0", 0)] = (
            "task0",
            channel,
            serde.dumps_typed("PRE-DELTA-WRITE"),
            "",
        )
        saver.writes[(thread_id, ns, "cp2")][("task2", 0)] = (
            "task2",
            channel,
            serde.dumps_typed("B"),
            "",
        )
        saver.writes[(thread_id, ns, "cp3")][("task3", 0)] = (
            "task3",
            channel,
            serde.dumps_typed("PENDING-AT-TARGET"),
            "",
        )
        return saver, thread_id, ns, channel, "cp3"

    def test_seed_from_pre_delta_ancestor_blob(self) -> None:
        saver, thread_id, ns, channel, target = self._build_mixed_thread()
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": target,
            }
        }

        result = saver._get_channel_writes_history(config, channel)

        # Seed came from the pre-delta blob at cp1.
        assert result.seed == ["A"]
        # Delta-era writes from cp2 replay through the reducer on top of seed.
        # cp3 is the target — its own write is pending for the NEXT step and
        # must be excluded.
        values = [v for _, _, v in result.writes]
        assert values == ["B"]

    def test_pre_delta_blob_terminates_walk_before_older_writes(self) -> None:
        """Writes stored at the pre-delta ancestor itself must not be replayed
        (the blob subsumes them)."""
        saver, thread_id, ns, channel, target = self._build_mixed_thread()
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": target,
            }
        }

        result = saver._get_channel_writes_history(config, channel)

        values = [v for _, _, v in result.writes]
        # The pre-delta write under cp1 must not appear (the blob subsumes it).
        assert "PRE-DELTA-WRITE" not in values
        # And the pending write at the target is never folded in.
        assert "PENDING-AT-TARGET" not in values


class TestInMemorySaverPrune:
    """Tests for InMemorySaver.prune with DeltaChannel awareness."""

    def _build_chain(
        self,
        saver: InMemorySaver,
        thread_id: str,
        ns: str,
        channel: str,
        n: int,
        *,
        snapshot_at: set[int] | None = None,
    ) -> list[str]:
        """Build a chain of n checkpoints with DELTA_SENTINEL blobs.

        If snapshot_at is provided, writes a _DeltaSnapshot blob at those steps.
        Returns list of checkpoint IDs in order (oldest first).
        """
        from langgraph.checkpoint.serde.types import _DeltaSnapshot

        serde = saver.serde
        cp_ids = []
        parent_id = None
        ver_base = "0000000000000000000000000000000{i}.0000000000000000"

        for i in range(n):
            cp_id = f"cp{i:04d}"
            ver = ver_base.format(i=i)
            cp = empty_checkpoint()
            cp["id"] = cp_id
            cp["channel_versions"][channel] = ver

            if snapshot_at and i in snapshot_at:
                blob = serde.dumps_typed(_DeltaSnapshot(value=[f"msg{i}"]))
            else:
                blob = serde.dumps_typed(DELTA_SENTINEL)

            saver.blobs[(thread_id, ns, channel, ver)] = blob
            saver.storage[thread_id][ns][cp_id] = (
                serde.dumps_typed(cp),
                serde.dumps_typed({}),
                parent_id,
            )
            # Add a dummy write for this checkpoint
            saver.writes[(thread_id, ns, cp_id)][("task", i)] = (
                "task",
                channel,
                serde.dumps_typed(f"write{i}"),
                "",
            )
            cp_ids.append(cp_id)
            parent_id = cp_id

        return cp_ids

    def test_prune_pure_delta_keeps_all(self) -> None:
        """With no snapshots, all checkpoints are required for reconstruction."""
        saver = InMemorySaver()
        thread_id, ns, channel = "t1", "", "messages"
        cp_ids = self._build_chain(saver, thread_id, ns, channel, n=5)

        saver.prune([thread_id], strategy="keep_latest")

        # All checkpoints must be retained (walk needs the full chain)
        remaining = set(saver.storage[thread_id][ns].keys())
        assert remaining == set(cp_ids)

    def test_prune_with_snapshot_removes_pre_snapshot_checkpoints(self) -> None:
        """Checkpoints older than the nearest snapshot can be safely pruned."""
        saver = InMemorySaver()
        thread_id, ns, channel = "t1", "", "messages"
        # Snapshot at step 2; steps 3 and 4 are sentinels
        cp_ids = self._build_chain(saver, thread_id, ns, channel, n=5, snapshot_at={2})

        saver.prune([thread_id], strategy="keep_latest")

        remaining = set(saver.storage[thread_id][ns].keys())
        # cp0, cp1 (before snapshot) must be gone; cp2, cp3, cp4 must remain
        assert cp_ids[0] not in remaining  # pre-snapshot
        assert cp_ids[1] not in remaining  # pre-snapshot
        assert cp_ids[2] in remaining  # the snapshot itself
        assert cp_ids[3] in remaining  # sentinel after snapshot
        assert cp_ids[4] in remaining  # latest

    def test_prune_removes_orphaned_blobs(self) -> None:
        """Blob entries for pruned checkpoints are cleaned up."""
        saver = InMemorySaver()
        thread_id, ns, channel = "t1", "", "messages"
        self._build_chain(saver, thread_id, ns, channel, n=4, snapshot_at={1})
        saver.prune([thread_id], strategy="keep_latest")

        # cp0 blob should be gone (pruned); cp1, cp2, cp3 blobs remain
        assert (
            thread_id,
            ns,
            channel,
            f"0000000000000000000000000000000{0}.0000000000000000",
        ) not in saver.blobs
        for i in range(1, 4):
            ver = f"0000000000000000000000000000000{i}.0000000000000000"
            assert (thread_id, ns, channel, ver) in saver.blobs

    def test_prune_removes_writes_for_pruned_checkpoints(self) -> None:
        """checkpoint_writes for pruned checkpoints are deleted."""
        saver = InMemorySaver()
        thread_id, ns, channel = "t1", "", "messages"
        cp_ids = self._build_chain(saver, thread_id, ns, channel, n=4, snapshot_at={1})

        saver.prune([thread_id], strategy="keep_latest")

        # writes for cp0 must be gone
        assert (thread_id, ns, cp_ids[0]) not in saver.writes
        # writes for cp1+ must remain (they're in the walk chain)
        for cp_id in cp_ids[1:]:
            assert (thread_id, ns, cp_id) in saver.writes

    def test_prune_delete_strategy_removes_everything(self) -> None:
        """strategy='delete' removes all checkpoints for the thread."""
        saver = InMemorySaver()
        thread_id, ns, channel = "t1", "", "messages"
        self._build_chain(saver, thread_id, ns, channel, n=3)

        saver.prune([thread_id], strategy="delete")

        assert not saver.storage.get(thread_id, {}).get(ns)
        assert not any(k[0] == thread_id for k in saver.writes)
        assert not any(k[0] == thread_id for k in saver.blobs)

    def test_prune_non_delta_channel_always_pruneable(self) -> None:
        """A channel with full snapshot blobs (no sentinels) allows full prune."""

        saver = InMemorySaver()
        thread_id, ns, channel = "t1", "", "messages"
        # All snapshots, no sentinels
        cp_ids = self._build_chain(
            saver, thread_id, ns, channel, n=4, snapshot_at={0, 1, 2, 3}
        )

        saver.prune([thread_id], strategy="keep_latest")

        remaining = set(saver.storage[thread_id][ns].keys())
        # Only the latest checkpoint is needed (all blobs are snapshots)
        assert remaining == {cp_ids[-1]}
