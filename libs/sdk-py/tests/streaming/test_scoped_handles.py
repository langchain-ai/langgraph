"""Tests for nested scoped stream handles."""

from __future__ import annotations

import httpx

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
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
from streaming._fake_server import FakeServer, _StreamScript


async def test_subgraphs_subscribes_to_tasks_channel():
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            _ = [handle async for handle in thread.subgraphs]

    assert any(
        "tasks" in body.get("channels", []) for body in fake.stream_request_bodies
    )


async def test_subgraphs_yields_handle_and_completes_status():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            tasks_result_event(seq=2, namespace=[], task_id="abc", name="worker"),
            lifecycle_completed_event(seq=3),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            handles = [handle async for handle in thread.subgraphs]

    assert len(handles) == 1
    handle = handles[0]
    assert handle.path == ("worker:abc",)
    assert handle.namespace == ["worker:abc"]
    assert handle.graph_name == "worker"
    assert handle.trigger_call_id == "abc"
    assert handle.status == "completed"
    assert handle.error is None


async def test_subgraphs_failed_and_interrupted_statuses():
    failed = FakeServer()
    failed.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            tasks_result_event(
                seq=2,
                namespace=[],
                task_id="abc",
                name="worker",
                error="boom",
            ),
            lifecycle_completed_event(seq=3),
        ]
    )
    failed_asgi = httpx.ASGITransport(app=failed.app)
    async with httpx.AsyncClient(transport=failed_asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            [failed_handle] = [handle async for handle in thread.subgraphs]

    interrupted = FakeServer()
    interrupted.script(
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
    interrupted_asgi = httpx.ASGITransport(app=interrupted.app)
    async with httpx.AsyncClient(
        transport=interrupted_asgi,
        base_url="http://test",
    ) as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            [interrupted_handle] = [handle async for handle in thread.subgraphs]

    assert failed_handle.status == "failed"
    assert failed_handle.error == "boom"
    assert interrupted_handle.status == "interrupted"
    assert interrupted_handle.error is None


async def test_subgraph_messages_are_scoped_to_child_namespace():
    fake = FakeServer()
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
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            [handle] = [h async for h in thread.subgraphs]
            child_messages = [message async for message in handle.messages]
            root_messages = [message async for message in thread.messages]

    assert [message.message_id for message in child_messages] == ["msg-child"]
    assert [await message.text for message in child_messages] == ["child"]
    assert root_messages == []


async def test_subgraph_tool_calls_are_scoped_to_child_namespace():
    fake = FakeServer()
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
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            [handle] = [h async for h in thread.subgraphs]
            child_calls = [call async for call in handle.tool_calls]
            root_calls = [call async for call in thread.tool_calls]

    assert [call.tool_call_id for call in child_calls] == ["call-child"]
    assert [delta async for delta in child_calls[0].deltas] == ["child-delta"]
    assert await child_calls[0].output == {"ok": True}
    assert root_calls == []


async def test_subgraph_handles_are_recursive_for_grandchildren():
    fake = FakeServer()
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
                seq=3,
                namespace=["worker:abc"],
                task_id="def",
                name="tool",
            ),
            tasks_result_event(seq=4, namespace=[], task_id="abc", name="worker"),
            lifecycle_completed_event(seq=5),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            [child] = [handle async for handle in thread.subgraphs]
            [grandchild] = [handle async for handle in child.subgraphs]

    assert child.path == ("worker:abc",)
    assert grandchild.path == ("worker:abc", "tool:def")
    assert grandchild.graph_name == "tool"
    assert grandchild.trigger_call_id == "def"
    assert grandchild.status == "completed"


async def test_subagents_aliases_subgraphs_until_protocol_distinguishes_them():
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            assert thread.subagents is thread.subgraphs
            await thread.run.start(input={})


async def test_root_and_child_projections_do_not_cross_talk():
    fake = FakeServer()
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
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            [handle] = [h async for h in thread.subgraphs]
            root_messages = [message async for message in thread.messages]
            child_messages = [message async for message in handle.messages]

    assert [message.message_id for message in root_messages] == ["root-msg"]
    assert [await message.text for message in root_messages] == ["root"]
    assert [message.message_id for message in child_messages] == ["child-msg"]
    assert [await message.text for message in child_messages] == ["child"]


async def test_grandchild_sibling_routing_preserves_event_order():
    """Events enqueued in a child handle's _messages_inbox before a grandchild
    is discovered via child.subgraphs must be delivered in arrival order, not
    reordered by the drain-and-replay path in _route_sibling_inboxes_to_grandchildren."""
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            # Child discovered by thread.subgraphs.
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
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            [child] = [h async for h in thread.subgraphs]
            [grandchild] = [h async for h in child.subgraphs]
            gc_messages = [m async for m in grandchild.messages]

    assert [m.message_id for m in gc_messages] == ["msg-E1", "msg-E2", "msg-E3"]
    texts = [await m.text for m in gc_messages]
    assert texts == ["E1", "E2", "E3"]


async def test_grandchild_events_dispatched_to_correct_sibling_not_first_match():
    """When two sibling grandchildren exist, messages scoped to one grandchild
    must not bleed into the other grandchild's inbox."""
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            # Two grandchildren discovered.
            tasks_start_event(
                seq=2, namespace=["worker:abc", "tool:gc1"], task_id="t-gc1"
            ),
            tasks_start_event(
                seq=3, namespace=["worker:abc", "tool:gc2"], task_id="t-gc2"
            ),
            # Messages for gc1.
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
            # Messages for gc2.
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
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            [child] = [h async for h in thread.subgraphs]
            grandchildren = [h async for h in child.subgraphs]
            by_path = {h.path: h for h in grandchildren}
            gc1_messages = [
                m async for m in by_path[("worker:abc", "tool:gc1")].messages
            ]
            gc2_messages = [
                m async for m in by_path[("worker:abc", "tool:gc2")].messages
            ]

    assert [m.message_id for m in gc1_messages] == ["msg-gc1"]
    assert [m.message_id for m in gc2_messages] == ["msg-gc2"]


def test_scoped_handle_inboxes_bounded_by_max_queue_size():
    """ScopedStreamHandle with max_queue_size=N creates queues with maxsize=N."""
    from unittest.mock import MagicMock

    from langgraph_sdk._async.stream import ScopedStreamHandle

    fake_thread = MagicMock()
    handle = ScopedStreamHandle(
        thread=fake_thread,
        path=("worker:1",),
        graph_name="worker",
        trigger_call_id="1",
        max_queue_size=16,
    )
    assert handle._messages_inbox.maxsize == 16
    assert handle._tools_inbox.maxsize == 16
    assert handle._tasks_inbox.maxsize == 16


async def test_child_handle_inherits_max_queue_size_from_parent():
    """Grandchild ScopedStreamHandles created by _HandleSubgraphsProjection
    inherit the parent's max_queue_size so all queues are consistently bounded."""
    fake = FakeServer()
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
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            [child] = [h async for h in thread.subgraphs]
            [grandchild] = [h async for h in child.subgraphs]

    # Grandchild handles created by _HandleSubgraphsProjection must use the
    # parent handle's max_queue_size (default 0 = unbounded in asyncio.Queue).
    assert grandchild._messages_inbox.maxsize == child._messages_inbox.maxsize
    assert grandchild._tools_inbox.maxsize == child._tools_inbox.maxsize
    assert grandchild._tasks_inbox.maxsize == child._tasks_inbox.maxsize


async def test_force_complete_uses_failed_when_run_errored():
    """If the lifecycle signals an errored run, scoped children that are still
    'started' when the subgraphs projection's finally block runs must be
    force-finished as 'failed', not 'completed'."""
    from streaming._events import lifecycle_errored_event

    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            # No tasks_result — run errored before the child task finished.
            lifecycle_errored_event(seq=2, error="boom"),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    handles: list = []
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            async for handle in thread.subgraphs:
                handles.append(handle)

    assert len(handles) == 1
    child = handles[0]
    # The run errored, so the child should be force-finished as "failed".
    assert child.status == "failed"


async def test_force_complete_uses_completed_when_run_completed():
    """If the lifecycle signals a completed run, any subgraph child still
    'started' at finally time is force-finished as 'completed' (normal case)."""
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tasks_start_event(seq=1, namespace=["worker:abc"], task_id="t-child"),
            # No tasks_result — but lifecycle completed normally.
            lifecycle_completed_event(seq=2),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    handles: list = []
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            async for handle in thread.subgraphs:
                handles.append(handle)

    assert len(handles) == 1
    child = handles[0]
    assert child.status == "completed"


def test_close_inboxes_does_not_enqueue_on_uniterated_inboxes():
    """_close_inboxes must not push a sentinel on inboxes that had no consumer."""
    from unittest.mock import MagicMock

    from langgraph_sdk._async.stream import ScopedStreamHandle

    fake_thread = MagicMock()
    handle = ScopedStreamHandle(
        thread=fake_thread,
        path=("worker:1",),
        graph_name="worker",
        trigger_call_id="1",
    )
    # No projection iterated — _close_inboxes should leave all queues empty.
    handle._close_inboxes()
    assert handle._messages_inbox.qsize() == 0
    assert handle._tools_inbox.qsize() == 0
    assert handle._tasks_inbox.qsize() == 0


def test_close_inboxes_enqueues_sentinel_on_iterated_inboxes():
    """_close_inboxes must push a None sentinel only on inboxes that had a consumer,
    so projection iterators see the EOF signal."""
    from unittest.mock import MagicMock

    from langgraph_sdk._async.stream import ScopedStreamHandle

    fake_thread = MagicMock()
    handle = ScopedStreamHandle(
        thread=fake_thread,
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


async def test_subgraph_scoped_messages_survive_shared_stream_reconnect():
    fake = FakeServer()
    # Connection order (async): shared stream opens first (via _reconcile_stream),
    # then the lifecycle watcher task runs. Three scripts are needed:
    # 1. shared stream initial (fail_after=2 to trigger reconnect)
    # 2. lifecycle watcher
    # 3. shared stream reconnect (carries the message events)
    fake.script_sequence(
        [
            _StreamScript(
                events=[
                    lifecycle_started_event(seq=0),
                    tasks_start_event(
                        seq=1, namespace=["worker:abc"], task_id="t-child"
                    ),
                ],
                fail_after=2,
            ),
            _StreamScript(
                events=[lifecycle_completed_event(seq=7)],
            ),
            _StreamScript(
                events=[
                    message_start_event(
                        seq=2,
                        namespace=["worker:abc"],
                        message_id="child-msg",
                        run_id="child-run",
                    ),
                    message_text_delta_event(
                        seq=3, namespace=["worker:abc"], text="child"
                    ),
                    message_text_finish_event(
                        seq=4, namespace=["worker:abc"], text="child"
                    ),
                    message_finish_event(seq=5, namespace=["worker:abc"]),
                    tasks_result_event(
                        seq=6, namespace=[], task_id="abc", name="worker"
                    ),
                    lifecycle_completed_event(seq=7),
                ]
            ),
        ]
    )
    async with httpx.AsyncClient(
        transport=fake.transport, base_url="http://test"
    ) as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            handles = [handle async for handle in thread.subgraphs]
            child = handles[0]
            chunks = [chunk async for chunk in child.messages]

    assert child.path == ("worker:abc",)
    assert [await chunk.text for chunk in chunks] == ["child"]
    assert fake.stream_request_bodies[-1]["since"] >= 1
