"""Sync v3 streaming projection tests."""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from langchain_core.language_models.chat_model_stream import ChatModelStream
from langchain_protocol import Event

from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.threads import SyncThreadsClient
from streaming._events import (
    lifecycle_completed_event,
    lifecycle_errored_event,
    lifecycle_started_event,
    message_error_event,
    message_finish_event,
    message_start_event,
    message_text_delta_event,
    message_text_finish_event,
    tasks_result_event,
    tasks_start_event,
    tool_error_event,
    tool_finished_event,
    tool_output_delta_event,
    tool_started_event,
)
from streaming._sync_fake_server import SyncFakeServer, SyncStreamScript


def test_sync_values_first_yield_is_rest_state_and_output_returns_final_state():
    fake = SyncFakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    fake.set_state({"answer": 42})
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            first = next(iter(thread.values))
            output = thread.output

    assert first == {"answer": 42}
    assert output == {"answer": 42}


def test_sync_messages_yield_chat_model_stream():
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-1"),
            message_text_delta_event(seq=2, text="hi", message_id="msg-1"),
            message_text_finish_event(seq=3, text="hi", message_id="msg-1"),
            message_finish_event(seq=4, message_id="msg-1"),
            lifecycle_completed_event(seq=5),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            messages = list(thread.messages)

    assert len(messages) == 1
    assert isinstance(messages[0], ChatModelStream)
    assert messages[0].message_id == "msg-1"
    assert str(messages[0].text) == "hi"
    assert messages[0].output.id == "msg-1"


def test_sync_tool_calls_yield_handle_deltas_and_output():
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1", tool_name="search"),
            tool_output_delta_event(seq=2, tool_call_id="call-1", delta="a"),
            tool_output_delta_event(seq=3, tool_call_id="call-1", delta="b"),
            tool_finished_event(seq=4, tool_call_id="call-1", output={"ok": True}),
            lifecycle_completed_event(seq=5),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            calls = list(thread.tool_calls)

    assert [call.tool_call_id for call in calls] == ["call-1"]
    assert list(calls[0].deltas) == ["a", "b"]
    assert calls[0].output == {"ok": True}


# ---------------------------------------------------------------------------
# Task 10.6 — messages pre-dispatch inner loop must filter by namespace
# ---------------------------------------------------------------------------


def test_sync_messages_ignores_nested_namespace_for_root_projection():
    """Root projection must not yield messages from child namespaces."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, namespace=["child:1"], message_id="nested"),
            message_text_delta_event(seq=2, namespace=["child:1"], text="nested"),
            message_finish_event(seq=3, namespace=["child:1"]),
            lifecycle_completed_event(seq=4),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            messages = list(thread.messages)

    assert messages == []


# ---------------------------------------------------------------------------
# Task 10.7 — comprehensive sync messages projection tests
# ---------------------------------------------------------------------------


def test_sync_messages_multiple_messages_are_distinct_streams():
    """Two sequential messages produce two separate ChatModelStream objects."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-1"),
            message_text_delta_event(seq=2, text="one", message_id="msg-1"),
            message_text_finish_event(seq=3, text="one", message_id="msg-1"),
            message_finish_event(seq=4, message_id="msg-1"),
            message_start_event(seq=5, message_id="msg-2"),
            message_text_delta_event(seq=6, text="two", message_id="msg-2"),
            message_text_finish_event(seq=7, text="two", message_id="msg-2"),
            message_finish_event(seq=8, message_id="msg-2"),
            lifecycle_completed_event(seq=9),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            streams = list(thread.messages)

    assert [s.message_id for s in streams] == ["msg-1", "msg-2"]
    assert [str(s.text) for s in streams] == ["one", "two"]


def test_sync_messages_error_event_fails_active_stream():
    """A messages `error` event marks the stream as failed."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-1"),
            message_error_event(seq=2, message="model failed", message_id="msg-1"),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            streams = list(thread.messages)

    assert len(streams) == 1
    with pytest.raises(RuntimeError, match="model failed"):
        _ = streams[0].output


def test_sync_subgraph_scoped_messages_and_tool_calls():
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
            tool_started_event(
                seq=6,
                namespace=["worker:abc"],
                tool_call_id="call-child",
                tool_name="search",
            ),
            tool_output_delta_event(
                seq=7,
                namespace=["worker:abc"],
                tool_call_id="call-child",
                delta="delta",
            ),
            tool_finished_event(
                seq=8,
                namespace=["worker:abc"],
                tool_call_id="call-child",
                output={"child": True},
            ),
            tasks_result_event(seq=9, namespace=[], task_id="abc", name="worker"),
            lifecycle_completed_event(seq=10),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            [handle] = list(thread.subgraphs)
            child_messages = list(handle.messages)
            child_calls = list(handle.tool_calls)

    assert handle.status == "completed"
    assert handle.path == ("worker:abc",)
    assert [message.message_id for message in child_messages] == ["msg-child"]
    assert [str(message.text) for message in child_messages] == ["child"]
    assert [call.tool_call_id for call in child_calls] == ["call-child"]
    assert list(child_calls[0].deltas) == ["delta"]
    assert child_calls[0].output == {"child": True}


def test_sync_subgraph_scoped_messages_survive_shared_stream_reconnect():
    fake = SyncFakeServer()
    # Connection order (sync): lifecycle watcher thread opens first (it is started
    # in __enter__ and races ahead during run.start), then the main thread opens
    # the shared stream. Three scripts are needed:
    # 1. lifecycle watcher
    # 2. shared stream initial (fail_after=2 to trigger reconnect)
    # 3. shared stream reconnect (carries the message events)
    fake.script_sequence(
        [
            SyncStreamScript(
                events=[
                    lifecycle_started_event(seq=0),
                    lifecycle_completed_event(seq=7),
                ]
            ),
            SyncStreamScript(
                events=[
                    lifecycle_started_event(seq=0),
                    tasks_start_event(
                        seq=1, namespace=["worker:abc"], task_id="t-child"
                    ),
                ],
                fail_after=2,
            ),
            SyncStreamScript(
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
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            handles = list(thread.subgraphs)
            child = handles[0]
            chunks = list(child.messages)

    assert child.path == ("worker:abc",)
    assert [str(chunk.text) for chunk in chunks] == ["child"]
    assert fake.stream_request_bodies[-1]["since"] >= 1


# ---------------------------------------------------------------------------
# Task 10.7 — comprehensive sync tool_calls projection tests
# ---------------------------------------------------------------------------


def test_sync_tool_calls_multiple_concurrent_calls_route_by_id():
    """Two interleaved tool calls each resolve to the correct output."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-a", tool_name="alpha"),
            tool_output_delta_event(seq=2, tool_call_id="call-a", delta="a1"),
            tool_finished_event(seq=3, tool_call_id="call-a", output="A"),
            tool_started_event(seq=4, tool_call_id="call-b", tool_name="beta"),
            tool_output_delta_event(seq=5, tool_call_id="call-b", delta="b1"),
            tool_finished_event(seq=6, tool_call_id="call-b", output="B"),
            lifecycle_completed_event(seq=7),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            calls = list(thread.tool_calls)

    by_id = {call.tool_call_id: call for call in calls}
    assert set(by_id) == {"call-a", "call-b"}
    assert list(by_id["call-a"].deltas) == ["a1"]
    assert list(by_id["call-b"].deltas) == ["b1"]
    assert by_id["call-a"].output == "A"
    assert by_id["call-b"].output == "B"


def test_sync_tool_calls_ignores_nested_namespace_for_root_projection():
    """Root tool_calls projection must not yield handles from child namespaces."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, namespace=["child:1"], tool_call_id="nested"),
            tool_finished_event(seq=2, namespace=["child:1"], tool_call_id="nested"),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            calls = list(thread.tool_calls)

    assert calls == []


def test_sync_tool_calls_error_event_fails_output():
    """A `tool-error` event fails the handle's output property."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1"),
            tool_output_delta_event(seq=2, tool_call_id="call-1", delta="before"),
            tool_error_event(seq=3, tool_call_id="call-1", message="boom"),
            lifecycle_completed_event(seq=4),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            calls = list(thread.tool_calls)

    assert len(calls) == 1
    assert list(calls[0].deltas) == ["before"]
    with pytest.raises(RuntimeError, match="boom"):
        _ = calls[0].output


# ---------------------------------------------------------------------------
# Task 10.10 — run-error propagates to active tool-call handle
# ---------------------------------------------------------------------------


def test_sync_tool_calls_run_error_fails_active_handle():
    """A lifecycle errored event propagates its message to the active handle."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1"),
            tool_output_delta_event(seq=2, tool_call_id="call-1", delta="partial"),
            lifecycle_errored_event(seq=3, error="run failed"),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            calls = list(thread.tool_calls)

    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="Run errored: run failed"):
        _ = calls[0].output


# ---------------------------------------------------------------------------
# Task 10.9 — _drain_messages_inbox must pre-dispatch before yielding
# ---------------------------------------------------------------------------


def test_sync_drain_messages_inbox_pre_dispatches_before_yield():
    """When draining the root inbox, str(message.text) must work immediately on yield."""
    fake = SyncFakeServer()
    fake.script([lifecycle_completed_event(seq=10)])
    fake.set_state({})
    collected_texts: list[str] = []
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            # Activate the root inbox and populate it directly (simulating what
            # a sync _SubgraphsProjection would do).
            inbox = thread._activate_root_messages_inbox()
            inbox.put(
                cast(
                    Event,
                    message_start_event(seq=1, message_id="msg-x"),
                )
            )
            inbox.put(
                cast(
                    Event,
                    message_text_delta_event(seq=2, text="hello", message_id="msg-x"),
                )
            )
            inbox.put(
                cast(
                    Event,
                    message_text_finish_event(seq=3, text="hello", message_id="msg-x"),
                )
            )
            inbox.put(cast(Event, message_finish_event(seq=4, message_id="msg-x")))
            inbox.put(None)  # EOF sentinel

            # Read text immediately inside the for loop — this requires pre-dispatch.
            for stream in thread.messages:
                collected_texts.append(str(stream.text))

    assert collected_texts == ["hello"]


# ---------------------------------------------------------------------------
# Fix A — remove blocking 1s wait in tool_calls iterator finally
# ---------------------------------------------------------------------------


def test_sync_tool_calls_explicit_close_does_not_block_1s():
    """Closing the tool_calls iterator must return in <200ms even without a terminal event.

    The prior finally block called `run_done.result(timeout=1.0)` unconditionally,
    causing a mandatory 1-second stall whenever the caller breaks out of the
    iterator before a lifecycle terminal event arrives.
    """
    fake = SyncFakeServer()
    # Script has a started lifecycle and one tool, but NO terminal lifecycle event.
    # If the blocking wait is present, the iterator's finally will stall for 1s.
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1"),
        ]
    )
    import time
    from collections.abc import Generator
    from typing import cast

    from langgraph_sdk._sync.stream import SyncToolCallHandle

    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            gen = cast(
                Generator[SyncToolCallHandle, None, None],
                thread.tool_calls._tool_calls_iter(),
            )
            _handle = next(gen)  # receive the one tool-started handle
            start = time.monotonic()
            # Explicitly close the generator — must not stall 1s.
            gen.close()
            elapsed = time.monotonic() - start

    assert elapsed < 0.2, f"tool_calls close() took {elapsed:.3f}s (expected <0.2s)"


# ---------------------------------------------------------------------------
# Fix B — drop len(active)==1 silent message routing fallback
# ---------------------------------------------------------------------------


def test_sync_messages_orphan_delta_without_matching_key_is_dropped():
    """A delta whose message_id doesn't match any active stream must be dropped.

    The old code fell back to routing the event to the only active stream when
    `len(active) == 1`, causing orphan/mismatched deltas to silently corrupt
    an unrelated stream's content.
    """
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-A"),
            # Delta with a mismatched message_id — must be dropped, not routed to msg-A.
            message_text_delta_event(seq=2, text="orphan", message_id="msg-UNKNOWN"),
            message_text_delta_event(seq=3, text="real", message_id="msg-A"),
            message_finish_event(seq=4, message_id="msg-A"),
            lifecycle_completed_event(seq=5),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            streams = list(thread.messages)

    assert len(streams) == 1
    # Only the correctly-keyed delta "real" must appear; "orphan" must be dropped.
    assert str(streams[0].text) == "real"


# ---------------------------------------------------------------------------
# Fix C — bound SyncToolCallHandle._deltas via max_queue_size
# ---------------------------------------------------------------------------


def test_sync_tool_call_handle_deltas_queue_is_bounded():
    """SyncToolCallHandle._deltas must be a bounded queue.

    Unbounded queues allow producers to enqueue indefinitely, causing memory
    growth when consumers are slow.
    """
    from langgraph_sdk._sync.stream import SyncToolCallHandle

    handle_default = SyncToolCallHandle(tool_call_id="tc1", name="foo")
    assert handle_default._deltas.maxsize > 0, (
        "default maxsize must be positive (bounded)"
    )

    handle_custom = SyncToolCallHandle(tool_call_id="tc2", name="bar", max_queue_size=8)
    assert handle_custom._deltas.maxsize == 8


# ---------------------------------------------------------------------------
# Fix D — enforce single consumer on SyncToolCallHandle.deltas
# ---------------------------------------------------------------------------


def test_sync_tool_call_handle_deltas_single_consumer_guard():
    """Accessing `handle.deltas` a second time must raise immediately.

    `_deltas` is a single-consumer queue; fanning out to multiple consumers
    would cause each consumer to miss events already consumed by the other.
    The property must raise before returning the iterator so the caller
    sees the error even without iterating.
    """
    from langgraph_sdk._sync.stream import SyncToolCallHandle

    handle = SyncToolCallHandle(tool_call_id="tc1", name="foo")

    # First access: fine — returns the iterator.
    _iter_1 = handle.deltas

    # Second access: must raise immediately (before any iteration).
    with pytest.raises(RuntimeError, match="single consumer"):
        _ = handle.deltas


def test_sync_messages_subscription_pre_dispatches_before_yield():
    """Over a live subscription, str(stream.text) must work immediately on yield."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-1"),
            message_text_delta_event(seq=2, text="hello", message_id="msg-1"),
            message_text_finish_event(seq=3, text="hello", message_id="msg-1"),
            message_finish_event(seq=4, message_id="msg-1"),
            lifecycle_completed_event(seq=5),
        ]
    )
    fake.set_state({})
    collected: list[str] = []
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            for stream in thread.messages:
                collected.append(str(stream.text))
    assert collected == ["hello"]


def test_sync_tool_calls_subscription_resolves_output_before_yield():
    """Over a live subscription, call.output is resolved when the handle is yielded."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1", tool_name="search"),
            tool_finished_event(seq=2, tool_call_id="call-1", output={"ok": True}),
            lifecycle_completed_event(seq=3),
        ]
    )
    fake.set_state({})
    outputs: list[Any] = []
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            for call in thread.tool_calls:
                outputs.append(call.output)  # resolved (blocking) on yield
    assert outputs == [{"ok": True}]


# ---------------------------------------------------------------------------
# Regression: interleaved-concurrent tool calls must BOTH surface
# ---------------------------------------------------------------------------


def test_sync_tool_calls_interleaved_concurrent_calls_both_surface():
    """Two tool calls whose events interleave must BOTH be yielded (regression:
    the pre-decoder read-ahead silently dropped the second concurrent call)."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-a", tool_name="search"),
            tool_started_event(seq=2, tool_call_id="call-b", tool_name="lookup"),
            tool_finished_event(seq=3, tool_call_id="call-a", output={"a": 1}),
            tool_finished_event(seq=4, tool_call_id="call-b", output={"b": 2}),
            lifecycle_completed_event(seq=5),
        ]
    )
    fake.set_state({})
    seen = []
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            for call in thread.tool_calls:
                seen.append(call.tool_call_id)
    assert sorted(seen) == ["call-a", "call-b"]
