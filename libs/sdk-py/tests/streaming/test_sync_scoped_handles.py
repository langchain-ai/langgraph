"""Sync mirror of test_scoped_handles.py for SyncScopedStreamHandle."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, wait

import httpx

from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.stream import SyncScopedStreamHandle
from langgraph_sdk._sync.threads import SyncThreadsClient
from streaming._events import (
    lifecycle_completed_event,
    lifecycle_started_event,
    message_finish_event,
    message_start_event,
    message_text_delta_event,
    message_text_finish_event,
    tasks_result_event,
    tasks_start_event,
    tool_finished_event,
    tool_output_delta_event,
    tool_started_event,
)
from streaming._sync_fake_server import SyncFakeServer

# ---------------------------------------------------------------------------
# Task 11.1: _finish idempotency — double-finish must not double-close inboxes
# ---------------------------------------------------------------------------


def test_sync_scoped_handle_finish_is_idempotent():
    """Calling _finish twice must not enqueue a second sentinel on each inbox."""
    handle = SyncScopedStreamHandle(
        thread=None,  # ty: ignore[invalid-argument-type]
        path=("worker:abc",),
        graph_name="worker",
        trigger_call_id="abc",
    )
    # Mark all three inboxes so _close_inboxes sends sentinels to each.
    handle._mark_iterated("messages")
    handle._mark_iterated("tools")
    handle._mark_iterated("tasks")
    handle._finish("completed")
    handle._finish("completed")  # second call must be a no-op

    # Each inbox should have exactly one None sentinel from the first _finish.
    assert handle._messages_inbox.qsize() == 1
    assert handle._tools_inbox.qsize() == 1
    assert handle._tasks_inbox.qsize() == 1
    assert handle._messages_inbox.get_nowait() is None
    assert handle._tools_inbox.get_nowait() is None
    assert handle._tasks_inbox.get_nowait() is None


def test_sync_scoped_handle_finish_concurrent_only_one_wins():
    """Concurrent _finish calls from two threads: only the first must close inboxes."""
    handle = SyncScopedStreamHandle(
        thread=None,  # ty: ignore[invalid-argument-type]
        path=("worker:abc",),
        graph_name="worker",
        trigger_call_id="abc",
    )
    # Mark all three inboxes so _close_inboxes sends sentinels to each.
    handle._mark_iterated("messages")
    handle._mark_iterated("tools")
    handle._mark_iterated("tasks")
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def _call_finish(status: str) -> None:
        try:
            barrier.wait()
            handle._finish(status)  # ty: ignore[invalid-argument-type]
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_call_finish, args=("completed",))
    t2 = threading.Thread(target=_call_finish, args=("failed",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors
    # Each inbox must have exactly one None sentinel regardless of which thread won.
    assert handle._messages_inbox.qsize() == 1
    assert handle._tools_inbox.qsize() == 1
    assert handle._tasks_inbox.qsize() == 1
    assert handle.status in ("completed", "failed")


# ---------------------------------------------------------------------------
# Task 11.2: subgraphs yields handle with correct metadata and completion status
# ---------------------------------------------------------------------------


def test_sync_subgraphs_yields_handle_and_completes_status():
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            tasks_result_event(seq=2, namespace=[], task_id="abc", name="worker"),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            handles = list(thread.subgraphs)

    assert len(handles) == 1
    handle = handles[0]
    assert handle.path == ("worker:abc",)
    assert handle.namespace == ["worker:abc"]
    assert handle.graph_name == "worker"
    assert handle.trigger_call_id == "abc"
    assert handle.status == "completed"
    assert handle.error is None


def test_sync_subgraphs_failed_and_interrupted_statuses():
    failed_fake = SyncFakeServer()
    failed_fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            tasks_result_event(
                seq=2, namespace=[], task_id="abc", name="worker", error="boom"
            ),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(transport=failed_fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            [failed_handle] = list(thread.subgraphs)

    interrupted_fake = SyncFakeServer()
    interrupted_fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:def"], task_id="t-child"),
            tasks_result_event(
                seq=2,
                namespace=[],
                task_id="def",
                name="worker",
                interrupts=[{"value": "pause"}],
            ),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(
        transport=interrupted_fake.transport, base_url="http://test"
    ) as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            [interrupted_handle] = list(thread.subgraphs)

    assert failed_handle.status == "failed"
    assert failed_handle.error == "boom"
    assert interrupted_handle.status == "interrupted"
    assert interrupted_handle.error is None


# ---------------------------------------------------------------------------
# Task 11.3: grandchild routing via _route_sibling_inboxes_to_grandchildren
# ---------------------------------------------------------------------------


def test_sync_subgraph_handles_are_recursive_for_grandchildren():
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            tasks_start_event(
                seq=2,
                namespace=["worker:abc", "tool:def"],
                task_id="t-grandchild",
            ),
            tasks_result_event(
                seq=3, namespace=["worker:abc"], task_id="def", name="tool"
            ),
            tasks_result_event(seq=4, namespace=[], task_id="abc", name="worker"),
            lifecycle_completed_event(seq=5),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            [child] = list(thread.subgraphs)
            [grandchild] = list(child.subgraphs)

    assert child.path == ("worker:abc",)
    assert grandchild.path == ("worker:abc", "tool:def")
    assert grandchild.graph_name == "tool"
    assert grandchild.trigger_call_id == "def"
    assert grandchild.status == "completed"


def test_sync_subgraph_messages_are_scoped_to_child_namespace():
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            message_start_event(
                seq=2,
                namespace=["worker:abc"],
                message_id="msg-child",
                run_id="run-child",
            ),
            message_text_delta_event(seq=3, namespace=["worker:abc"], text="child"),
            message_text_finish_event(seq=4, namespace=["worker:abc"], text="child"),
            message_finish_event(seq=5, namespace=["worker:abc"]),
            tasks_result_event(seq=6, namespace=[], task_id="abc", name="worker"),
            lifecycle_completed_event(seq=7),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            [handle] = list(thread.subgraphs)
            child_messages = list(handle.messages)
            root_messages = list(thread.messages)

    assert [m.message_id for m in child_messages] == ["msg-child"]
    assert [str(m.text) for m in child_messages] == ["child"]
    assert root_messages == []


def test_sync_subgraph_tool_calls_are_scoped_to_child_namespace():
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            tool_started_event(
                seq=2,
                namespace=["worker:abc"],
                tool_call_id="call-child",
                tool_name="search",
            ),
            tool_output_delta_event(
                seq=3,
                namespace=["worker:abc"],
                tool_call_id="call-child",
                delta="child-delta",
            ),
            tool_finished_event(
                seq=4,
                namespace=["worker:abc"],
                tool_call_id="call-child",
                output={"ok": True},
            ),
            tasks_result_event(seq=5, namespace=[], task_id="abc", name="worker"),
            lifecycle_completed_event(seq=6),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            [handle] = list(thread.subgraphs)
            child_calls = list(handle.tool_calls)
            root_calls = list(thread.tool_calls)

    assert [call.tool_call_id for call in child_calls] == ["call-child"]
    assert list(child_calls[0].deltas) == ["child-delta"]
    assert child_calls[0].output == {"ok": True}
    assert root_calls == []


# ---------------------------------------------------------------------------
# Task 11.4: root/child cross-talk regression
# ---------------------------------------------------------------------------


def test_sync_root_and_child_projections_do_not_cross_talk():
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            message_start_event(seq=2, message_id="root-msg", run_id="root-run"),
            message_text_delta_event(seq=3, text="root"),
            message_text_finish_event(seq=4, text="root"),
            message_finish_event(seq=5),
            message_start_event(
                seq=6,
                namespace=["worker:abc"],
                message_id="child-msg",
                run_id="child-run",
            ),
            message_text_delta_event(seq=7, namespace=["worker:abc"], text="child"),
            message_text_finish_event(seq=8, namespace=["worker:abc"], text="child"),
            message_finish_event(seq=9, namespace=["worker:abc"]),
            tasks_result_event(seq=10, namespace=[], task_id="abc", name="worker"),
            lifecycle_completed_event(seq=11),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            [handle] = list(thread.subgraphs)
            root_messages = list(thread.messages)
            child_messages = list(handle.messages)

    assert [m.message_id for m in root_messages] == ["root-msg"]
    assert [str(m.text) for m in root_messages] == ["root"]
    assert [m.message_id for m in child_messages] == ["child-msg"]
    assert [str(m.text) for m in child_messages] == ["child"]


def test_sync_subagents_aliases_subgraphs():
    fake = SyncFakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            assert thread.subagents is thread.subgraphs
            thread.run.start(input={})


# ---------------------------------------------------------------------------
# Fix A: sibling routing dispatches at push time, preserves order, fans out
# ---------------------------------------------------------------------------


def test_sync_scoped_handle_has_descendant_handles_dict():
    """SyncScopedStreamHandle must expose _descendant_handles so push-time
    fan-out can be registered without drain-and-replay."""
    handle = SyncScopedStreamHandle(
        thread=None,  # ty: ignore[invalid-argument-type]
        path=("worker:abc",),
        graph_name="worker",
        trigger_call_id="abc",
    )
    # Must exist and be empty at construction.
    assert hasattr(handle, "_descendant_handles")
    assert isinstance(handle._descendant_handles, dict)
    assert len(handle._descendant_handles) == 0


def test_sync_register_descendant_forwards_buffered_events_in_order():
    """_register_descendant must drain already-buffered events whose namespace
    matches the new grandchild, push them into the grandchild, and preserve
    the original arrival order in the parent inbox."""
    from langchain_protocol import Event

    parent = SyncScopedStreamHandle(
        thread=None,  # ty: ignore[invalid-argument-type]
        path=("worker:abc",),
        graph_name="worker",
        trigger_call_id="abc",
    )
    child_path = ("worker:abc", "tool:gc1")
    grandchild = SyncScopedStreamHandle(
        thread=None,  # ty: ignore[invalid-argument-type]
        path=child_path,
        graph_name="tool",
        trigger_call_id="gc1",
    )
    # Put two grandchild-scoped events and one parent-scoped event into parent inbox.
    evt_gc1: Event = message_start_event(
        seq=1, namespace=list(child_path), message_id="m1", run_id="r1"
    )  # ty: ignore[invalid-assignment]
    evt_gc2: Event = message_start_event(
        seq=2, namespace=list(child_path), message_id="m2", run_id="r2"
    )  # ty: ignore[invalid-assignment]
    evt_parent: Event = message_start_event(
        seq=3, namespace=list(parent.path), message_id="m3", run_id="r3"
    )  # ty: ignore[invalid-assignment]
    parent._messages_inbox.put_nowait(evt_gc1)
    parent._messages_inbox.put_nowait(evt_parent)
    parent._messages_inbox.put_nowait(evt_gc2)

    parent._register_descendant(grandchild)

    # Grandchild inbox must have exactly the two grandchild events.
    assert grandchild._messages_inbox.qsize() == 2
    first = grandchild._messages_inbox.get_nowait()
    second = grandchild._messages_inbox.get_nowait()
    assert isinstance(first, dict) and isinstance(second, dict)
    assert (first.get("params") or {}).get("data", {}).get("id") == "m1"
    assert (second.get("params") or {}).get("data", {}).get("id") == "m2"

    # Parent inbox must still contain all 3 events in original order.
    assert parent._messages_inbox.qsize() == 3
    items = [parent._messages_inbox.get_nowait() for _ in range(3)]
    ids = [
        (i.get("params") or {}).get("data", {}).get("id")
        if isinstance(i, dict)
        else None
        for i in items
    ]
    assert ids == ["m1", "m3", "m2"]


def test_sync_grandchild_sibling_routing_preserves_event_order():
    """Events enqueued in a child handle's _messages_inbox before a grandchild
    is discovered via child.subgraphs must be delivered in arrival order."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            # Grandchild messages arrive *before* grandchild tasks-start.
            message_start_event(
                seq=2,
                namespace=["worker:abc", "tool:gc1"],
                message_id="msg-E1",
                run_id="r1",
            ),
            message_text_delta_event(
                seq=3, namespace=["worker:abc", "tool:gc1"], text="E1"
            ),
            message_text_finish_event(
                seq=4, namespace=["worker:abc", "tool:gc1"], text="E1"
            ),
            message_finish_event(seq=5, namespace=["worker:abc", "tool:gc1"]),
            message_start_event(
                seq=6,
                namespace=["worker:abc", "tool:gc1"],
                message_id="msg-E2",
                run_id="r2",
            ),
            message_text_delta_event(
                seq=7, namespace=["worker:abc", "tool:gc1"], text="E2"
            ),
            message_text_finish_event(
                seq=8, namespace=["worker:abc", "tool:gc1"], text="E2"
            ),
            message_finish_event(seq=9, namespace=["worker:abc", "tool:gc1"]),
            message_start_event(
                seq=10,
                namespace=["worker:abc", "tool:gc1"],
                message_id="msg-E3",
                run_id="r3",
            ),
            message_text_delta_event(
                seq=11, namespace=["worker:abc", "tool:gc1"], text="E3"
            ),
            message_text_finish_event(
                seq=12, namespace=["worker:abc", "tool:gc1"], text="E3"
            ),
            message_finish_event(seq=13, namespace=["worker:abc", "tool:gc1"]),
            # Grandchild tasks-start arrives *after* its messages.
            tasks_start_event(
                seq=14,
                namespace=["worker:abc", "tool:gc1"],
                task_id="t-grandchild",
            ),
            tasks_result_event(
                seq=15, namespace=["worker:abc"], task_id="gc1", name="tool"
            ),
            tasks_result_event(seq=16, namespace=[], task_id="abc", name="worker"),
            lifecycle_completed_event(seq=17),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            [child] = list(thread.subgraphs)
            [grandchild] = list(child.subgraphs)
            gc_messages = list(grandchild.messages)

    assert [m.message_id for m in gc_messages] == ["msg-E1", "msg-E2", "msg-E3"]
    texts = [str(m.text) for m in gc_messages]
    assert texts == ["E1", "E2", "E3"]


def test_sync_grandchild_events_dispatched_to_correct_sibling():
    """When two sibling grandchildren exist, messages scoped to one grandchild
    must not bleed into the other grandchild's inbox."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            tasks_start_event(
                seq=2, namespace=["worker:abc", "tool:gc1"], task_id="t-gc1"
            ),
            tasks_start_event(
                seq=3, namespace=["worker:abc", "tool:gc2"], task_id="t-gc2"
            ),
            message_start_event(
                seq=4,
                namespace=["worker:abc", "tool:gc1"],
                message_id="msg-gc1",
                run_id="ra",
            ),
            message_text_delta_event(
                seq=5, namespace=["worker:abc", "tool:gc1"], text="GC1"
            ),
            message_text_finish_event(
                seq=6, namespace=["worker:abc", "tool:gc1"], text="GC1"
            ),
            message_finish_event(seq=7, namespace=["worker:abc", "tool:gc1"]),
            message_start_event(
                seq=8,
                namespace=["worker:abc", "tool:gc2"],
                message_id="msg-gc2",
                run_id="rb",
            ),
            message_text_delta_event(
                seq=9, namespace=["worker:abc", "tool:gc2"], text="GC2"
            ),
            message_text_finish_event(
                seq=10, namespace=["worker:abc", "tool:gc2"], text="GC2"
            ),
            message_finish_event(seq=11, namespace=["worker:abc", "tool:gc2"]),
            tasks_result_event(
                seq=12, namespace=["worker:abc"], task_id="gc1", name="tool"
            ),
            tasks_result_event(
                seq=13, namespace=["worker:abc"], task_id="gc2", name="tool"
            ),
            tasks_result_event(seq=14, namespace=[], task_id="abc", name="worker"),
            lifecycle_completed_event(seq=15),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            [child] = list(thread.subgraphs)
            grandchildren = list(child.subgraphs)
            by_path = {h.path: h for h in grandchildren}
            gc1_messages = list(by_path[("worker:abc", "tool:gc1")].messages)
            gc2_messages = list(by_path[("worker:abc", "tool:gc2")].messages)

    assert [m.message_id for m in gc1_messages] == ["msg-gc1"]
    assert [m.message_id for m in gc2_messages] == ["msg-gc2"]


# ---------------------------------------------------------------------------
# Task 11.5: _finish thread-safe status transition via threading.Lock
# ---------------------------------------------------------------------------


def _finish_one(
    idx: int,
    handle: SyncScopedStreamHandle,
    barrier: threading.Barrier,
    errors: list[Exception],
    statuses: list[str],
) -> None:
    try:
        barrier.wait()
        status = statuses[idx % len(statuses)]
        handle._finish(status)  # ty: ignore[invalid-argument-type]
    except Exception as exc:
        errors.append(exc)


def test_sync_scoped_handle_finish_thread_safe_with_20_concurrent_calls():
    """20 concurrent _finish calls must yield exactly one terminal status and one
    sentinel per inbox (deterministic even under high contention).
    """
    n_workers = 20
    statuses = ["completed", "failed", "interrupted"]

    for _ in range(50):  # repeat to increase chance of catching races
        handle = SyncScopedStreamHandle(
            thread=None,  # ty: ignore[invalid-argument-type]
            path=("worker:abc",),
            graph_name="worker",
            trigger_call_id="abc",
        )
        # Mark all inboxes so _close_inboxes sends sentinels to each.
        handle._mark_iterated("messages")
        handle._mark_iterated("tools")
        handle._mark_iterated("tasks")
        barrier = threading.Barrier(n_workers)
        errors: list[Exception] = []

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(_finish_one, i, handle, barrier, errors, statuses)
                for i in range(n_workers)
            ]
            wait(futures)

        assert not errors, f"Unexpected exception(s): {errors}"
        # Status must be one of the valid terminal values (not "started").
        assert handle.status in ("completed", "failed", "interrupted"), (
            f"Unexpected status: {handle.status!r}"
        )
        # Each inbox must have exactly one None sentinel — the lock must ensure
        # that _close_inboxes() is called exactly once.
        assert handle._messages_inbox.qsize() == 1, (
            f"messages_inbox has {handle._messages_inbox.qsize()} items (expected 1)"
        )
        assert handle._tools_inbox.qsize() == 1, (
            f"tools_inbox has {handle._tools_inbox.qsize()} items (expected 1)"
        )
        assert handle._tasks_inbox.qsize() == 1, (
            f"tasks_inbox has {handle._tasks_inbox.qsize()} items (expected 1)"
        )
        assert handle._messages_inbox.get_nowait() is None
        assert handle._tools_inbox.get_nowait() is None
        assert handle._tasks_inbox.get_nowait() is None


# ---------------------------------------------------------------------------
# Fix B: bound SyncScopedStreamHandle inboxes via max_queue_size
# ---------------------------------------------------------------------------


def test_sync_scoped_handle_inboxes_bounded_by_max_queue_size():
    """SyncScopedStreamHandle with max_queue_size=N creates queues with maxsize=N."""
    handle = SyncScopedStreamHandle(
        thread=None,  # ty: ignore[invalid-argument-type]
        path=("worker:1",),
        graph_name="worker",
        trigger_call_id="1",
        max_queue_size=16,
    )
    assert handle._messages_inbox.maxsize == 16
    assert handle._tools_inbox.maxsize == 16
    assert handle._tasks_inbox.maxsize == 16


def test_sync_child_handle_inherits_max_queue_size_from_parent():
    """Grandchild SyncScopedStreamHandles created by _SyncHandleSubgraphsProjection
    inherit the parent's max_queue_size so all queues are consistently bounded."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            tasks_start_event(
                seq=2, namespace=["worker:abc", "tool:gc1"], task_id="t-gc1"
            ),
            tasks_result_event(
                seq=3, namespace=["worker:abc"], task_id="gc1", name="tool"
            ),
            tasks_result_event(seq=4, namespace=[], task_id="abc", name="worker"),
            lifecycle_completed_event(seq=5),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            [child] = list(thread.subgraphs)
            [grandchild] = list(child.subgraphs)

    # Grandchild queues must have the same maxsize as the parent queues.
    assert grandchild._messages_inbox.maxsize == child._messages_inbox.maxsize
    assert grandchild._tools_inbox.maxsize == child._tools_inbox.maxsize
    assert grandchild._tasks_inbox.maxsize == child._tasks_inbox.maxsize


# ---------------------------------------------------------------------------
# Fix C: force-complete uses parent terminal status
# ---------------------------------------------------------------------------


def test_sync_force_complete_uses_failed_when_run_errored():
    """If the lifecycle signals an errored run, scoped children that are still
    'started' when the subgraphs iterator's finally block runs must be
    force-finished as 'failed', not 'completed'."""
    from streaming._events import lifecycle_errored_event

    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            # No tasks_result — run errored before the child task finished.
            lifecycle_errored_event(seq=2, error="boom"),
        ]
    )
    handles: list[SyncScopedStreamHandle] = []
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            for handle in thread.subgraphs:
                handles.append(handle)

    assert len(handles) == 1
    child = handles[0]
    assert child.status == "failed"


def test_sync_force_complete_uses_completed_when_run_completed():
    """If the lifecycle signals a completed run, any subgraph child still
    'started' at finally time is force-finished as 'completed' (normal case)."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            # No tasks_result — but lifecycle completed normally.
            lifecycle_completed_event(seq=2),
        ]
    )
    handles: list[SyncScopedStreamHandle] = []
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            for handle in thread.subgraphs:
                handles.append(handle)

    assert len(handles) == 1
    child = handles[0]
    assert child.status == "completed"


# ---------------------------------------------------------------------------
# Fix D: only enqueue close sentinel on inboxes that had a consumer
# ---------------------------------------------------------------------------


def test_sync_close_inboxes_does_not_enqueue_on_uniterated_inboxes():
    """_close_inboxes must not push a sentinel on inboxes that had no consumer."""
    handle = SyncScopedStreamHandle(
        thread=None,  # ty: ignore[invalid-argument-type]
        path=("worker:1",),
        graph_name="worker",
        trigger_call_id="1",
    )
    # No projection iterated — _close_inboxes should leave all queues empty.
    handle._close_inboxes()
    assert handle._messages_inbox.qsize() == 0
    assert handle._tools_inbox.qsize() == 0
    assert handle._tasks_inbox.qsize() == 0


def test_sync_close_inboxes_enqueues_sentinel_on_iterated_inboxes():
    """_close_inboxes must push a None sentinel only on inboxes that had a consumer,
    so projection iterators see the EOF signal."""
    handle = SyncScopedStreamHandle(
        thread=None,  # ty: ignore[invalid-argument-type]
        path=("worker:1",),
        graph_name="worker",
        trigger_call_id="1",
    )
    handle._mark_iterated("messages")
    handle._close_inboxes()
    # Only the messages inbox should have a sentinel.
    assert handle._messages_inbox.qsize() == 1
    assert handle._messages_inbox.get_nowait() is None
    assert handle._tools_inbox.qsize() == 0
    assert handle._tasks_inbox.qsize() == 0
