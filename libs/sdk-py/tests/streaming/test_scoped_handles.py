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
from streaming._fake_server import FakeServer


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
