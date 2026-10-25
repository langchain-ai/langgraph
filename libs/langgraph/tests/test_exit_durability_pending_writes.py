import asyncio
import concurrent.futures
import threading
import weakref
from collections import Counter
from collections.abc import Sequence
from contextlib import AsyncExitStack, ExitStack
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph._internal._constants import (
    CONFIG_KEY_NODE_FINISHED,
    CONFIG_KEY_RUNNER_SUBMIT,
    ERROR,
)
from langgraph.config import get_stream_writer
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.pregel._executor import AsyncBackgroundExecutor, BackgroundExecutor
from langgraph.pregel._runner import PregelRunner
from langgraph.types import CachePolicy, Durability


@pytest.mark.parametrize("delay_commit", [False, True])
@pytest.mark.parametrize("custom_submit", [False, True], ids=["default", "custom"])
def test_exit_durability_does_not_rerun_late_successful_sibling(
    delay_commit: bool,
    custom_submit: bool,
    durability: Durability,
) -> None:
    """A sibling that succeeds during shutdown must not rerun on resume (#9397)."""
    allow_failure = threading.Event()
    error_saved = threading.Event()
    calls: Counter[str] = Counter()

    class State(TypedDict, total=False):
        a: int
        b: int

    class SignalingSaver(InMemorySaver):
        def put_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[tuple[str, Any]],
            task_id: str,
            task_path: str = "",
        ) -> None:
            super().put_writes(config, writes, task_id, task_path)
            if any(channel == ERROR for channel, _ in writes):
                error_saved.set()

    def flaky(state: State) -> dict[str, int]:
        calls["flaky"] += 1
        assert allow_failure.wait(timeout=5)
        if calls["flaky"] == 1:
            raise RuntimeError("transient failure")
        return {"a": 1}

    def slow(state: State) -> dict[str, int]:
        calls["slow"] += 1
        get_stream_writer()("slow started")
        assert allow_failure.wait(timeout=5)
        # Release after an early exit flush, or finish independently if shutdown
        # correctly waits for this node before persisting the error.
        if not delay_commit:
            error_saved.wait(timeout=1)
        return {"b": 1}

    def node_finished(name: str) -> None:
        if name == "slow" and delay_commit:
            # The Future is done, but its commit callback has not saved writes.
            error_saved.wait(timeout=1)

    builder = StateGraph(State)
    builder.add_node("flaky", flaky)
    builder.add_node("slow", slow)
    builder.add_edge(START, "flaky")
    builder.add_edge(START, "slow")
    builder.add_edge(["flaky", "slow"], END)
    graph = builder.compile(checkpointer=SignalingSaver())
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "late-successful-sibling",
            CONFIG_KEY_NODE_FINISHED: node_finished,
        },
        "max_concurrency": 3,  # Two nodes and the custom stream waiter.
    }

    with pytest.raises(RuntimeError, match="transient failure"):
        with ExitStack() as stack:
            if custom_submit:
                submit = stack.enter_context(BackgroundExecutor(config))
                config["configurable"][CONFIG_KEY_RUNNER_SUBMIT] = weakref.WeakMethod(
                    submit
                )
            for chunk in graph.stream(
                {}, config, durability=durability, stream_mode="custom"
            ):
                # The runner has registered both tasks before yielding this event.
                assert chunk == "slow started"
                allow_failure.set()

    assert calls == {"flaky": 1, "slow": 1}
    assert "b" in [
        ch for _, ch, _ in graph.checkpointer.get_tuple(config).pending_writes
    ]
    config["configurable"].pop(CONFIG_KEY_RUNNER_SUBMIT, None)
    graph = builder.compile(checkpointer=graph.checkpointer)
    assert graph.invoke(None, config, durability=durability) == {"a": 1, "b": 1}
    assert calls == {"flaky": 2, "slow": 1}


@pytest.mark.anyio
@pytest.mark.parametrize("custom_submit", [False, True], ids=["default", "custom"])
@pytest.mark.parametrize("external_cancel", [False, True], ids=["failure", "cancel"])
async def test_exit_durability_retries_cancelled_sibling(
    custom_submit: bool,
    durability: Durability,
    external_cancel: bool,
) -> None:
    """An unfinished sibling must still run again after cancellation."""
    calls: Counter[str] = Counter()
    slow_started = asyncio.Event()

    class State(TypedDict, total=False):
        a: int
        b: int

    async def flaky(state: State) -> dict[str, int]:
        calls["flaky"] += 1
        await asyncio.wait_for(slow_started.wait(), timeout=5)
        if calls["flaky"] == 1:
            if external_cancel:
                await asyncio.Event().wait()
            raise RuntimeError("transient failure")
        return {"a": 1}

    async def slow(state: State) -> dict[str, int]:
        calls["slow"] += 1
        slow_started.set()
        if calls["slow"] == 1:
            await asyncio.Event().wait()
        return {"b": 1}

    builder = StateGraph(State)
    builder.add_node("flaky", flaky)
    builder.add_node("slow", slow)
    builder.add_edge(START, "flaky")
    builder.add_edge(START, "slow")
    builder.add_edge(["flaky", "slow"], END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "cancelled-sibling"}}

    with pytest.raises(
        asyncio.CancelledError if external_cancel else RuntimeError,
        match=None if external_cancel else "transient failure",
    ):
        async with AsyncExitStack() as stack:
            if custom_submit:
                submit = await stack.enter_async_context(
                    AsyncBackgroundExecutor(config)
                )
                config["configurable"][CONFIG_KEY_RUNNER_SUBMIT] = weakref.WeakMethod(
                    submit
                )
            invocation = asyncio.create_task(
                graph.ainvoke({}, config, durability=durability)
            )
            if external_cancel:
                await asyncio.wait_for(slow_started.wait(), timeout=5)
                invocation.cancel()
            await asyncio.wait_for(invocation, timeout=5)

    assert calls == {"flaky": 1, "slow": 1}
    # Both the failure and the cancelled sibling's commit must reach the saver.
    assert (
        len(
            [
                ch
                for _, ch, _ in graph.checkpointer.get_tuple(config).pending_writes
                if ch == ERROR
            ]
        )
        == 2
    )
    config["configurable"].pop(CONFIG_KEY_RUNNER_SUBMIT, None)
    graph = builder.compile(checkpointer=graph.checkpointer)
    assert await graph.ainvoke(None, config, durability=durability) == {"a": 1, "b": 1}
    assert calls == {"flaky": 2, "slow": 2}


@pytest.mark.anyio
@pytest.mark.parametrize("custom_submit", [False, True], ids=["default", "custom"])
@pytest.mark.parametrize("extra_cancels", [1, 3])
async def test_durability_waits_for_commits_after_repeated_cancel(
    custom_submit: bool,
    extra_cancels: int,
    durability: Durability,
) -> None:
    """Repeated cancellation cannot advance the exit stack past unfinished commits."""
    started = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    counts: Counter[str] = Counter()

    class State(TypedDict):
        value: int

    async def node(state: State) -> None:
        counts["started"] += 1
        if counts["started"] == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            counts["cleaning"] += 1
            if counts["cleaning"] == 2:
                cleaning.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            counts["cleaned"] += 1

    builder = StateGraph(State)
    builder.add_node("one", node)
    builder.add_node("two", node)
    builder.add_edge(START, "one")
    builder.add_edge(START, "two")
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": "repeated-cancel"}}

    async with AsyncExitStack() as stack:
        if custom_submit:
            submit = await stack.enter_async_context(AsyncBackgroundExecutor(config))
            config["configurable"][CONFIG_KEY_RUNNER_SUBMIT] = weakref.WeakMethod(
                submit
            )
        invocation = asyncio.create_task(
            graph.ainvoke({"value": 0}, config, durability=durability)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            invocation.cancel()
            await asyncio.wait_for(cleaning.wait(), timeout=5)
            for _ in range(extra_cancels):
                invocation.cancel()
                await asyncio.sleep(0)
            done, _ = await asyncio.wait({invocation}, timeout=0.05)
            returned_before_cleanup = bool(done)
            saved_before_cleanup = saver.get_tuple(config)
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(invocation, timeout=5)

    assert not returned_before_cleanup
    if durability == "exit":
        assert saved_before_cleanup is None
    assert counts == {"started": 2, "cleaning": 2, "cleaned": 2}
    assert sum(ch == ERROR for _, ch, _ in saver.get_tuple(config).pending_writes) == 2


@pytest.mark.anyio
@pytest.mark.parametrize("custom_submit", [False, True], ids=["default", "custom"])
async def test_durability_rejects_child_created_during_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    custom_submit: bool,
    durability: Durability,
) -> None:
    """Node cleanup cannot add an uncancelled child after the shutdown snapshot."""
    closing = asyncio.Event()
    started = asyncio.Event()
    child_started = asyncio.Event()
    release_child = asyncio.Event()
    original_close = PregelRunner.aclose

    async def observe_close(runner: PregelRunner) -> None:
        closing.set()
        await original_close(runner)

    monkeypatch.setattr(PregelRunner, "aclose", observe_close)

    @task
    async def child() -> str:
        child_started.set()
        await release_child.wait()
        return "done"

    @task
    async def parent() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            # The parent can receive another cancellation while entering cleanup.
            while True:
                try:
                    await closing.wait()
                    break
                except asyncio.CancelledError:
                    pass
            await child()

    saver = InMemorySaver()

    @entrypoint(checkpointer=saver)
    async def graph(value: int) -> None:
        await parent()

    config: RunnableConfig = {"configurable": {"thread_id": "late-child"}}
    async with AsyncExitStack() as stack:
        if custom_submit:
            submit = await stack.enter_async_context(AsyncBackgroundExecutor(config))
            config["configurable"][CONFIG_KEY_RUNNER_SUBMIT] = weakref.WeakMethod(
                submit
            )
        invocation = asyncio.create_task(
            graph.ainvoke(0, config, durability=durability)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            invocation.cancel()
            done, _ = await asyncio.wait({invocation}, timeout=1)
            exited = bool(done)
        finally:
            release_child.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(invocation, timeout=5)

    assert exited, "Shutdown waited indefinitely for a late child"
    assert not child_started.is_set()
    assert any(ch == ERROR for _, ch, _ in saver.get_tuple(config).pending_writes)


@pytest.mark.parametrize("custom_submit", [False, True], ids=["default", "custom"])
def test_durability_rejects_sync_child_during_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    custom_submit: bool,
    durability: Durability,
) -> None:
    """A still-running sync node cannot submit new work once its runner closes."""
    allow_failure = threading.Event()
    closing = threading.Event()
    calls: Counter[str] = Counter()
    original_close = PregelRunner.close

    def observe_close(runner: PregelRunner) -> None:
        assert runner._sync_futures is not None
        original_wait = runner._sync_futures.all_done.wait

        def wait_for_commits(timeout: float | None = None) -> bool:
            closing.set()
            return original_wait(timeout)

        monkeypatch.setattr(runner._sync_futures.all_done, "wait", wait_for_commits)
        original_close(runner)

    monkeypatch.setattr(PregelRunner, "close", observe_close)

    class State(TypedDict, total=False):
        value: int

    @task
    def child() -> int:
        calls["child"] += 1
        return 1

    def flaky(state: State) -> None:
        assert allow_failure.wait(timeout=5)
        raise RuntimeError("transient failure")

    def slow(state: State) -> dict[str, int]:
        get_stream_writer()("slow started")
        assert closing.wait(timeout=5)
        with pytest.raises(RuntimeError, match="runner is closing"):
            child().result()
        return {"value": 1}

    builder = StateGraph(State)
    builder.add_node("flaky", flaky)
    builder.add_node("slow", slow)
    builder.add_edge(START, "flaky")
    builder.add_edge(START, "slow")
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)
    config: RunnableConfig = {
        "configurable": {"thread_id": "late-sync-child"},
        "max_concurrency": 3,
    }
    with pytest.raises(RuntimeError, match="transient failure"):
        with ExitStack() as stack:
            if custom_submit:
                submit = stack.enter_context(BackgroundExecutor(config))
                config["configurable"][CONFIG_KEY_RUNNER_SUBMIT] = weakref.WeakMethod(
                    submit
                )
            for chunk in graph.stream(
                {}, config, durability=durability, stream_mode="custom"
            ):
                assert chunk == "slow started"
                # Release only after both node futures have been registered.
                allow_failure.set()

    assert not calls
    pending = saver.get_tuple(config).pending_writes
    assert any(ch == "value" for _, ch, _ in pending), pending


@pytest.mark.anyio
@pytest.mark.parametrize("custom_submit", [False, True], ids=["default", "custom"])
@pytest.mark.parametrize("sync_node", [False, True], ids=["async-node", "sync-node"])
@pytest.mark.parametrize("finish", ["success", "cancel", "error", "child-cancel"])
async def test_durability_owns_pending_task_admission(
    custom_submit: bool,
    sync_node: bool,
    finish: str,
    durability: Durability,
) -> None:
    """Cache lookup is part of task lifetime, before a node Future exists."""
    lookup_started = asyncio.Event()
    handle_ready = asyncio.Event()
    event_loop = asyncio.get_running_loop()
    worker_lookup_started = threading.Event()
    lookup_finished = asyncio.Event()
    release = asyncio.Event()
    cache_tasks: list[asyncio.Task] = []
    calls: list[int] = []
    handles: list[Any] = []

    class DelayedCache(InMemoryCache):
        async def aget(self, keys: Sequence[Any]) -> dict[Any, Any]:
            current = asyncio.current_task()
            assert current is not None
            cache_tasks.append(current)
            lookup_started.set()
            worker_lookup_started.set()
            try:
                await release.wait()
                return {}
            finally:
                lookup_finished.set()

    @task(cache_policy=CachePolicy())
    async def child(value: int) -> int:
        calls.append(value)
        return value

    class State(TypedDict):
        value: int

    async def first(state: State) -> dict[str, int]:
        handles.append(child(1))
        handle_ready.set()
        await lookup_started.wait()
        if finish == "error":
            raise RuntimeError("parent failed during admission")
        return {"value": 1}

    def first_sync(state: State) -> dict[str, int]:
        handles.append(child(1))
        event_loop.call_soon_threadsafe(handle_ready.set)
        assert worker_lookup_started.wait(timeout=5)
        if finish == "error":
            raise RuntimeError("parent failed during admission")
        return {"value": 1}

    async def second(state: State) -> dict[str, int]:
        return {"value": 2}

    builder = StateGraph(State)
    builder.add_node("first", first_sync if sync_node else first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    graph = builder.compile(checkpointer=InMemorySaver(), cache=DelayedCache())
    config: RunnableConfig = {"configurable": {"thread_id": "pending-admission"}}

    async with AsyncExitStack() as stack:
        if custom_submit:
            submit = await stack.enter_async_context(AsyncBackgroundExecutor(config))
            config["configurable"][CONFIG_KEY_RUNNER_SUBMIT] = weakref.WeakMethod(
                submit
            )
        invocation = asyncio.create_task(
            graph.ainvoke({"value": 0}, config, durability=durability)
        )
        try:
            await asyncio.wait_for(lookup_started.wait(), timeout=5)
            await asyncio.wait_for(handle_ready.wait(), timeout=5)
            if finish == "success":
                done, _ = await asyncio.wait({invocation}, timeout=0.05)
                owned = not bool(done)
            elif finish == "child-cancel":
                handles[0].cancel()
                assert await asyncio.wait_for(invocation, timeout=5) == {"value": 2}
                owned = lookup_finished.is_set() and all(t.done() for t in cache_tasks)
            else:
                if finish == "cancel":
                    invocation.cancel()
                with pytest.raises(
                    asyncio.CancelledError if finish == "cancel" else RuntimeError,
                    match=None
                    if finish == "cancel"
                    else "parent failed during admission",
                ):
                    await asyncio.wait_for(invocation, timeout=5)
                owned = lookup_finished.is_set() and all(t.done() for t in cache_tasks)
        finally:
            release.set()
            try:
                await asyncio.wait_for(invocation, timeout=5)
            except (asyncio.CancelledError, RuntimeError):
                pass
            for handle in handles:
                future = (
                    handle if asyncio.isfuture(handle) else asyncio.wrap_future(handle)
                )
                try:
                    await asyncio.wait_for(asyncio.shield(future), timeout=5)
                except asyncio.CancelledError:
                    pass

    assert owned, "Graph exited while task admission was still running"
    assert all(t.done() for t in cache_tasks)
    assert calls == ([1] if finish == "success" else [])


def test_durability_rechecks_sync_admission_after_cache_lookup(
    monkeypatch: pytest.MonkeyPatch,
    durability: Durability,
) -> None:
    """A blocking cache lookup can overlap the start of synchronous shutdown."""
    closing = threading.Event()
    allow_failure = threading.Event()
    calls: list[int] = []
    original_close = PregelRunner.close

    def observe_close(runner: PregelRunner) -> None:
        assert runner._sync_futures is not None
        original_wait = runner._sync_futures.all_done.wait

        def wait(timeout: float | None = None) -> bool:
            closing.set()
            return original_wait(timeout)

        monkeypatch.setattr(runner._sync_futures.all_done, "wait", wait)
        original_close(runner)

    monkeypatch.setattr(PregelRunner, "close", observe_close)

    class DelayedCache(InMemoryCache):
        def get(self, keys: Sequence[Any]) -> dict[Any, Any]:
            get_stream_writer()("cache lookup")
            assert closing.wait(timeout=5)
            return {}

    class ImmediateChildExecutor(BackgroundExecutor):
        def submit(
            self, fn: Any, *args: Any, **kwargs: Any
        ) -> concurrent.futures.Future:
            if args and getattr(args[0], "name", None) == "child":
                # A custom submitter may finish a short task before returning
                # its Future, so cancelling at registration would be too late.
                for key in (
                    "__name__",
                    "__cancel_on_exit__",
                    "__reraise_on_exit__",
                    "__next_tick__",
                ):
                    kwargs.pop(key, None)
                future: concurrent.futures.Future = concurrent.futures.Future()
                try:
                    future.set_result(fn(*args, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)
                return future
            return super().submit(fn, *args, **kwargs)

    @task(cache_policy=CachePolicy())
    def child(value: int) -> int:
        calls.append(value)
        return value

    class State(TypedDict, total=False):
        value: int

    def flaky(state: State) -> None:
        assert allow_failure.wait(timeout=5)
        raise RuntimeError("transient failure")

    def slow(state: State) -> dict[str, int]:
        with pytest.raises(RuntimeError, match="runner is closing"):
            child(1).result()
        return {"value": 1}

    builder = StateGraph(State)
    builder.add_node("flaky", flaky)
    builder.add_node("slow", slow)
    builder.add_edge(START, "flaky")
    builder.add_edge(START, "slow")
    graph = builder.compile(checkpointer=InMemorySaver(), cache=DelayedCache())
    config: RunnableConfig = {
        "configurable": {"thread_id": "sync-cache-shutdown"},
        "max_concurrency": 3,
    }
    with pytest.raises(RuntimeError, match="transient failure"):
        with ImmediateChildExecutor(config) as submit:
            config["configurable"][CONFIG_KEY_RUNNER_SUBMIT] = weakref.WeakMethod(
                submit
            )
            for chunk in graph.stream(
                {}, config, durability=durability, stream_mode="custom"
            ):
                assert chunk == "cache lookup"
                allow_failure.set()
    assert not calls
