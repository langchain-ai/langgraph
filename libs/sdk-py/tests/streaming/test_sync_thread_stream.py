"""Tests for SyncThreadStream — Tasks 9.1 through 9.6."""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Iterator

import httpx
import pytest

from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.threads import SyncThreadsClient
from langgraph_sdk.stream.transport.sync_http import (
    SyncEventStreamHandle,
    SyncProtocolSseTransport,
)
from streaming._sync_fake_server import SyncFakeServer, SyncStreamScript

# ---------------------------------------------------------------------------
# Task 9.1 — run_start_gate
# ---------------------------------------------------------------------------


def test_sync_subscribe_before_run_start_waits_on_gate():
    """A subscribe issued before run.start completes must block until the
    gate is set, mirroring async behavior."""
    fake = SyncFakeServer()
    # Lifecycle + fanout streams: empty so threads terminate cleanly.
    fake.script_sequence(
        [
            SyncStreamScript(events=[]),  # lifecycle watcher
            SyncStreamScript(events=[]),  # first subscribe
        ]
    )

    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            controller = thread._controller
            assert controller is not None

            started = threading.Event()

            def slow_subscriber() -> None:
                started.set()
                list(thread.subscribe(["values"]))

            t = threading.Thread(target=slow_subscriber)
            t.start()
            started.wait(timeout=0.5)

            # Set the gate manually (simulating run.start completing)
            time.sleep(0.05)
            assert controller._run_start_gate is not None
            controller._run_start_gate.set()

            t.join(timeout=2.0)

    # The subscriber should have unblocked and terminated cleanly.
    assert not t.is_alive(), "subscriber thread should have terminated"


# ---------------------------------------------------------------------------
# Task 9.2 — reconnect backoff + ready check
# ---------------------------------------------------------------------------


def test_sync_reconnect_uses_backoff_between_attempts(monkeypatch):
    """_reconnect_shared_stream sleeps between retry attempts with exp+jitter
    backoff, mirroring the async reconnect behavior."""
    import langgraph_sdk.stream.sync_controller as _ctrl_mod

    sleeps: list[float] = []
    monkeypatch.setattr(_ctrl_mod.time, "sleep", lambda d: sleeps.append(d))

    from langgraph_sdk.stream.sync_controller import SyncStreamController
    from langgraph_sdk.stream.transport.sync_http import SyncProtocolSseTransport

    class _FailingTransport(SyncProtocolSseTransport):
        """Transport that always raises on open_event_stream."""

        def open_event_stream(self, params: dict) -> SyncEventStreamHandle:  # noqa: ARG002
            raise RuntimeError("scripted transport failure")

    with httpx.Client(base_url="http://test") as raw:
        transport = _FailingTransport(client=raw, thread_id="t-1")
        controller = SyncStreamController(transport, max_reconnect_attempts=5)
        controller._shared_stream_filter = {"channels": ["values"]}
        result = controller._reconnect_shared_stream()

    assert result is False, "all attempts should have failed"
    # Attempts 0..4 → sleeps before attempts 1..4 → 4 sleeps
    assert len(sleeps) == 4, f"Expected 4 sleeps, got {sleeps}"
    # Backoff should grow (each delay is larger than previous, ignoring jitter)
    delays_without_jitter = [0.1 * (2**i) for i in range(4)]
    for i, (sleep, expected_base) in enumerate(
        zip(sleeps, delays_without_jitter, strict=False)
    ):
        assert sleep >= expected_base, (
            f"sleep[{i}]={sleep} < expected base {expected_base}"
        )


# ---------------------------------------------------------------------------
# Task 9.3 — rotation drains buffered events from old stream
# ---------------------------------------------------------------------------


def test_sync_rotation_does_not_lose_buffered_events():
    """When the shared stream rotates, old-stream events already in the queue
    are not dropped.  _drain_and_close dispatches remaining events from the
    old handle to subscribers before closing it."""
    import queue
    from typing import Any

    from langgraph_sdk.stream.sync_controller import SyncStreamController
    from langgraph_sdk.stream.transport.sync_http import (
        SyncEventStreamHandle,
        SyncProtocolSseTransport,
    )
    from streaming._events import values_event

    event_a = values_event(seq=1, counter=1)

    class _ScriptedTransport(SyncProtocolSseTransport):
        """First call produces event_a; second call produces an empty stream."""

        def open_event_stream(self, params: dict) -> SyncEventStreamHandle:  # noqa: ARG002
            def _gen_a() -> Iterator[Any]:
                yield event_a

            def _gen_empty() -> Iterator[Any]:
                return
                yield  # pragma: no cover

            # Alternate: first call → a, second → empty.
            if not hasattr(self, "_call_count"):
                self._call_count = 0
            self._call_count += 1
            events_gen: Iterator[Any] = (
                _gen_a() if self._call_count == 1 else _gen_empty()
            )
            return SyncEventStreamHandle(
                events=events_gen,
                error=lambda: None,
                close=lambda: None,
            )

    with httpx.Client(base_url="http://test") as raw:
        transport = _ScriptedTransport(client=raw, thread_id="t-1")
        controller = SyncStreamController(transport)
        sub = controller.register_subscription({"channels": ["values"]})

        # First reconcile — opens old stream (event_a available immediately).
        controller.reconcile_stream({"channels": ["values"]})
        # Do NOT start fanout; let reconcile_stream cause a rotation directly.

        # Second reconcile: rotates to empty stream; drain thread handles old.
        controller.reconcile_stream({"channels": ["values", "updates"]})

        # Start fanout AFTER rotation (picks up the new empty stream).
        controller.ensure_fanout_running()

        # Allow drain thread to finish before collecting results.
        controller.close()

        received = []
        while True:
            try:
                item = sub.queue.get_nowait()
                if item is None:
                    continue
                received.append(item)
            except queue.Empty:
                break

    seqs = [e.get("seq") for e in received]
    assert 1 in seqs, f"event_a (seq=1) not received via drain; got seqs={seqs}"


# ---------------------------------------------------------------------------
# Task 9.4 — _next_command_id lock
# ---------------------------------------------------------------------------


def test_sync_concurrent_commands_do_not_share_command_id():
    """50 concurrent threads calling _send_command must each get a unique id."""
    from concurrent.futures import ThreadPoolExecutor
    from typing import Any

    captured_ids: list[int] = []
    ids_lock = threading.Lock()

    class _CapturingTransport(SyncProtocolSseTransport):
        """Captures command ids; always returns success."""

        def send_command(self, command: dict) -> dict:
            with ids_lock:
                captured_ids.append(command["id"])
            return {"type": "success", "id": command["id"], "result": {}}

        def open_event_stream(self, params: dict) -> SyncEventStreamHandle:  # noqa: ARG002
            def _gen() -> Iterator[Any]:
                return
                yield

            return SyncEventStreamHandle(
                events=_gen(), error=lambda: None, close=lambda: None
            )

    fake = SyncFakeServer()
    fake.script_sequence([SyncStreamScript(events=[])])

    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads_client = SyncThreadsClient(SyncHttpClient(raw))
        with threads_client.stream(thread_id="t-cmd", assistant_id="agent") as stream:
            # Pre-set gate so _send_command doesn't wait.
            if stream._controller and stream._controller._run_start_gate:
                stream._controller._run_start_gate.set()
            # Replace transport with capturing transport.
            capture_transport = _CapturingTransport(client=raw, thread_id="t-cmd")
            stream._transport = capture_transport

            with ThreadPoolExecutor(max_workers=50) as ex:
                futures = [
                    ex.submit(stream._send_command, "noop", {}) for _ in range(50)
                ]
                for f in futures:
                    f.result()

    assert len(set(captured_ids)) == 50, (
        f"Expected 50 unique command ids, got {len(set(captured_ids))} unique "
        f"out of {len(captured_ids)} total: {sorted(captured_ids)}"
    )


# ---------------------------------------------------------------------------
# Task 9.5 — sync events returns fresh iterator per access
# ---------------------------------------------------------------------------


def test_sync_events_returns_fresh_iterator_each_access():
    """Two accesses of `thread.events` yield independent subscriptions,
    mirroring the async semantics where each access opens a new subscriber."""
    fake = SyncFakeServer()
    from streaming._events import values_event

    event_1 = values_event(seq=1, counter=1)
    fake.script_sequence(
        [
            SyncStreamScript(events=[]),  # lifecycle watcher
            SyncStreamScript(events=[event_1]),  # first events access
            SyncStreamScript(events=[event_1]),  # second events access
        ]
    )

    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads_client = SyncThreadsClient(SyncHttpClient(raw))
        with threads_client.stream(thread_id="t-5", assistant_id="agent") as thread:
            # Pre-set gate.
            if thread._controller and thread._controller._run_start_gate:
                thread._controller._run_start_gate.set()

            iter1 = thread.events
            iter2 = thread.events

            # They must be independent objects (different subscription iterators).
            assert iter1 is not iter2


# ---------------------------------------------------------------------------
# Task 9.6 — close ordering: fail active streams before controller close
# ---------------------------------------------------------------------------


def test_close_unblocks_active_subscription_before_lifecycle_join():
    """close() must send None to active subscriptions BEFORE joining the
    lifecycle watcher thread, so callers wake quickly even if the watcher
    thread blocks for up to 1s."""
    import queue

    # Gate that keeps the lifecycle watcher thread alive for 0.4s.
    lifecycle_block = threading.Event()
    unblock_times: list[float] = []
    close_times: list[float] = []

    class _BlockingFakeServer(SyncFakeServer):
        """Lifecycle stream blocks until gate set; subscribe stream is empty."""

        def _handle(self, request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/stream/events"):
                import orjson

                body = orjson.loads(request.content)
                channels = body.get("channels", [])
                if "lifecycle" in channels:
                    # Block lifecycle watcher for 0.4s.
                    lifecycle_block.wait(timeout=0.4)
            return super()._handle(request)

    fake = _BlockingFakeServer()
    fake.script_sequence(
        [
            SyncStreamScript(events=[]),  # lifecycle watcher
            SyncStreamScript(events=[]),  # subscribe fanout stream
        ]
    )

    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads_client = SyncThreadsClient(SyncHttpClient(raw))
        with threads_client.stream(thread_id="t-6", assistant_id="agent") as thread:
            if thread._controller and thread._controller._run_start_gate:
                thread._controller._run_start_gate.set()

            assert thread._controller is not None
            sub = thread._controller.register_subscription({"channels": ["values"]})
            thread._controller.reconcile_stream({"channels": ["values"]})
            thread._controller.ensure_fanout_running()

            consumer_ready = threading.Event()

            def _consume() -> None:
                consumer_ready.set()
                while True:
                    try:
                        item = sub.queue.get(timeout=2.0)
                        if item is None:
                            unblock_times.append(time.monotonic())
                            return
                    except queue.Empty:
                        return

            t = threading.Thread(target=_consume)
            t.start()
            consumer_ready.wait(timeout=1.0)
            time.sleep(0.02)

            close_times.append(time.monotonic())
            # __exit__ calls close() here.

    lifecycle_block.set()  # Unblock watcher so test can finish.
    t.join(timeout=2.0)
    assert not t.is_alive(), "consumer thread should have unblocked"
    assert unblock_times, "consumer never received sentinel"
    elapsed = unblock_times[0] - close_times[0]
    # With controller closed BEFORE lifecycle join, sentinel arrives fast.
    # Lifecycle watcher blocks for 0.4s but that should not delay the sentinel.
    assert elapsed < 0.3, (
        f"consumer woke {elapsed:.3f}s after close() — "
        "controller.close() should precede the lifecycle thread join"
    )


def test_sync_thread_agent_get_tree_fetches_assistant_graph():
    fake = SyncFakeServer()
    fake.set_graph(
        {
            "nodes": [{"id": "agent", "type": "runnable", "data": {"name": "agent"}}],
            "edges": [{"source": "agent", "target": "__end__"}],
        }
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(
            thread_id="t-1",
            assistant_id="agent",
            headers={"X-Custom-Header": "my-value"},
        ) as thread:
            graph = thread.agent.get_tree(xray=True)

    assert graph["nodes"][0]["id"] == "agent"
    assert graph["edges"] == [{"source": "agent", "target": "__end__"}]
    assert fake.graph_request_params == [{"xray": "true"}]
    assert fake.graph_request_headers[0].get("x-custom-header") == "my-value"


def test_sync_thread_agent_get_tree_raises_after_close():
    with httpx.Client(base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        stream = threads.stream(thread_id="t-1", assistant_id="agent")
        stream.close()
        with pytest.raises(RuntimeError, match="closed"):
            stream.agent.get_tree()


def test_sync_extensions_projection_empty_name_raises():
    with httpx.Client(base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        stream = threads.stream(thread_id="t-1", assistant_id="agent")
        with pytest.raises(ValueError, match="non-empty"):
            stream.extensions[""]


def test_sync_extensions_projection_closed_stream_yields_nothing():
    with httpx.Client(base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        # Enter and immediately exit so _controller is set but _closed is True.
        with threads.stream(thread_id="t-1", assistant_id="agent") as stream:
            pass
        payloads = list(stream.extensions["progress"])
    assert payloads == []


def test_sync_threads_stream_mints_uuid4_when_thread_id_none():
    with httpx.Client(base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        stream = threads.stream(assistant_id="agent")
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            stream.thread_id,
        )
        assert uuid.UUID(stream.thread_id).version == 4


def test_sync_run_start_sends_command():
    from streaming._events import lifecycle_completed_event

    fake = SyncFakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            result = thread.run.start(input={"x": 1})

    assert result == {"run_id": "run-1"}
    assert fake.received_commands[0]["method"] == "run.start"
    assert fake.received_commands[0]["params"]["assistant_id"] == "agent"


def test_sync_events_iterates_raw_events():
    from streaming._events import values_event

    fake = SyncFakeServer()
    fake.script([values_event(seq=1, counter=1)])
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            events = list(thread.subscribe(["values"]))

    assert events == [values_event(seq=1, counter=1)]


def test_sync_lifecycle_watcher_reconnects_with_since_after_transport_drop():
    from streaming._events import lifecycle_completed_event, lifecycle_event

    fake = SyncFakeServer()
    fake.set_state({"ok": True})
    fake.script_sequence(
        [
            SyncStreamScript(
                events=[lifecycle_event(seq=1, phase="running")],
                fail_after=1,
            ),
            SyncStreamScript(events=[lifecycle_completed_event(seq=2)]),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="existing", assistant_id="agent") as thread:
            terminal = thread._wait_for_run_done()

    assert terminal.status == "completed"
    assert terminal.error is None
    assert fake.stream_request_bodies[1]["since"] == 1


def test_sync_threads_stream_accepts_websocket_transport_option():
    with httpx.Client(base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        stream = threads.stream(
            thread_id="t-1",
            assistant_id="agent",
            transport="websocket",
        )
    assert stream._transport_kind == "websocket"


def test_sync_threads_stream_rejects_unknown_transport_option():
    import pytest

    with httpx.Client(base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with pytest.raises(ValueError, match="transport"):
            threads.stream(
                thread_id="t-1",
                assistant_id="agent",
                transport="bogus",  # ty: ignore[invalid-argument-type]
            )


def test_v3_streaming_sync_surface_smoke():
    from streaming._events import (
        custom_event,
        lifecycle_completed_event,
        message_finish_event,
        message_start_event,
        message_text_delta_event,
        message_text_finish_event,
        tool_finished_event,
        tool_started_event,
        values_event,
    )

    fake = SyncFakeServer()
    fake.set_state({"final": True})
    # Single script — projections consume events in parallel threads so all
    # subscriptions are registered before SSE rotation could drop events.
    # Mirrors the async smoke test's `asyncio.gather` pattern.
    fake.script(
        [
            values_event(seq=1, values={"step": 1}),
            message_start_event(seq=2, message_id="msg-1"),
            message_text_delta_event(seq=3, text="hi", message_id="msg-1"),
            message_text_finish_event(seq=4, text="hi", message_id="msg-1"),
            message_finish_event(seq=5, message_id="msg-1"),
            tool_started_event(seq=6, tool_call_id="call-1", tool_name="search"),
            tool_finished_event(seq=7, tool_call_id="call-1", output={"ok": True}),
            custom_event(seq=8, name="progress", step=1),
            lifecycle_completed_event(seq=9),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            start = thread.run.start(
                input={"messages": [{"role": "user", "content": "hi"}]}
            )

            # Gate every reconcile_stream call on a barrier so that all four
            # projection threads register their subscriptions before any
            # reconcile widens (or rotates) the shared SSE. This mirrors the
            # async smoke test's `asyncio.gather` pattern: every subscription
            # is registered before the first SSE opens; one SSE covers all
            # consumers and `_seen_event_ids` covers any subsequent reconnect.
            controller = thread._controller
            assert controller is not None
            barrier = threading.Barrier(4)
            real_reconcile = controller.reconcile_stream

            def _gated_reconcile(candidate_filter):
                barrier.wait(timeout=10)
                return real_reconcile(candidate_filter)

            controller.reconcile_stream = _gated_reconcile  # ty: ignore[invalid-assignment]

            results: dict[str, object] = {}
            errors: list[BaseException] = []

            def _run_values() -> None:
                try:
                    for v in thread.values:
                        results["values"] = v
                        return
                except BaseException as err:  # pragma: no cover - propagated
                    errors.append(err)

            def _run_messages() -> None:
                try:
                    results["messages"] = list(thread.messages)
                except BaseException as err:  # pragma: no cover - propagated
                    errors.append(err)

            def _run_tools() -> None:
                try:
                    results["tools"] = list(thread.tool_calls)
                except BaseException as err:  # pragma: no cover - propagated
                    errors.append(err)

            def _run_progress() -> None:
                try:
                    results["progress"] = list(thread.extensions["progress"])
                except BaseException as err:  # pragma: no cover - propagated
                    errors.append(err)

            workers = [
                threading.Thread(target=_run_values),
                threading.Thread(target=_run_messages),
                threading.Thread(target=_run_tools),
                threading.Thread(target=_run_progress),
            ]
            for w in workers:
                w.start()
            for w in workers:
                w.join(timeout=10)
                assert not w.is_alive(), "smoke worker thread hung"
            controller.reconcile_stream = real_reconcile  # ty: ignore[invalid-assignment]
            assert not errors, errors
            final = thread.output

    assert start == {"run_id": "run-1"}
    assert results["values"] == fake.state["values"]
    messages_result = results["messages"]
    assert isinstance(messages_result, list)
    assert [str(m.text) for m in messages_result] == ["hi"]  # ty: ignore[unresolved-attribute]
    tools_result = results["tools"]
    assert isinstance(tools_result, list)
    assert tools_result[0].name == "search"  # ty: ignore[unresolved-attribute]
    assert results["progress"] == [{"name": "progress", "step": 1}]
    assert final == {"final": True}


# ---------------------------------------------------------------------------
# interleave_projections tests
# ---------------------------------------------------------------------------


def test_interleave_projections_single_channel_values():
    from streaming._events import (
        lifecycle_completed_event,
        lifecycle_started_event,
        values_event,
    )

    fake = SyncFakeServer()
    fake.set_state({"counter": 0})
    fake.script(
        [
            lifecycle_started_event(seq=0),
            values_event(seq=1, counter=1),
            values_event(seq=2, counter=2),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            items = []
            for ch, item in thread.interleave_projections(["values"]):
                items.append((ch, item))
    assert ("values", {"counter": 1}) in items
    assert ("values", {"counter": 2}) in items
    assert all(ch == "values" for ch, _ in items)


def test_interleave_projections_values_and_messages_arrival_order():
    from streaming._events import (
        lifecycle_completed_event,
        lifecycle_started_event,
        message_finish_event,
        message_start_event,
        values_event,
    )

    fake = SyncFakeServer()
    fake.set_state({"counter": 0})
    fake.script(
        [
            lifecycle_started_event(seq=0),
            values_event(seq=1, counter=1),
            message_start_event(seq=2, message_id="m-1"),
            values_event(seq=3, counter=2),
            message_finish_event(seq=4, message_id="m-1"),
            lifecycle_completed_event(seq=5),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            order = []
            for ch, _ in thread.interleave_projections(["values", "messages"]):
                order.append(ch)
                if len(order) >= 3:
                    break
    assert order[:3] == ["values", "messages", "values"]


def test_interleave_projections_mixes_builtin_and_extension():
    from streaming._events import (
        custom_event,
        lifecycle_completed_event,
        lifecycle_started_event,
        values_event,
    )

    fake = SyncFakeServer()
    fake.set_state({"counter": 0})
    fake.script(
        [
            lifecycle_started_event(seq=0),
            values_event(seq=1, counter=1),
            custom_event(seq=2, name="foo", hello="world"),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            items = []
            for ch, item in thread.interleave_projections(["values", "foo"]):
                items.append((ch, item))
    assert ("values", {"counter": 1}) in items
    assert ("foo", {"name": "foo", "hello": "world"}) in items


def test_interleave_projections_tool_calls_uses_public_name():
    from streaming._events import (
        lifecycle_completed_event,
        lifecycle_started_event,
        tool_finished_event,
        tool_started_event,
    )

    fake = SyncFakeServer()
    fake.set_state({})
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1", tool_name="search"),
            tool_finished_event(seq=2, tool_call_id="call-1", output={"ok": True}),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            names = []
            handle = None
            for ch, item in thread.interleave_projections(["tool_calls"]):
                names.append(ch)
                if handle is None:
                    handle = item
                break
    assert names == ["tool_calls"]
    assert handle is not None
    assert handle.tool_call_id == "call-1"


def test_interleave_projections_subgraphs_discovers_child():
    from streaming._events import (
        lifecycle_completed_event,
        lifecycle_started_event,
    )

    fake = SyncFakeServer()
    fake.set_state({})
    fake.script(
        [
            lifecycle_started_event(seq=0),
            lifecycle_started_event(seq=1, namespace=["child"]),
            lifecycle_completed_event(seq=2),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            discovered = []
            for ch, handle in thread.interleave_projections(["subgraphs"]):
                discovered.append((ch, handle.path))
    assert ("subgraphs", ("child",)) in discovered


def test_interleave_projections_inflight_tool_call_failed_on_break():
    """A tool handle held past an early break is failed in teardown, never left hanging."""
    from streaming._events import (
        lifecycle_completed_event,
        lifecycle_started_event,
        tool_started_event,
    )

    fake = SyncFakeServer()
    fake.set_state({})
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1", tool_name="search"),
            # no tool-finished: the call is still in flight when the consumer breaks
            lifecycle_completed_event(seq=2),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            handle = None
            for _, item in thread.interleave_projections(["tool_calls"]):
                handle = item
                break
            assert handle is not None
            # Without teardown finalization this blocks forever; the bounded
            # timeout turns a regression into a TimeoutError, not a RuntimeError.
            with pytest.raises(RuntimeError):
                handle._result.result(timeout=2)


def test_interleave_projections_inflight_subgraph_finished_on_terminal():
    """A discovered subgraph child with no terminal tasks-result is force-completed."""
    from streaming._events import (
        lifecycle_completed_event,
        lifecycle_started_event,
    )

    fake = SyncFakeServer()
    fake.set_state({})
    fake.script(
        [
            lifecycle_started_event(seq=0),
            lifecycle_started_event(seq=1, namespace=["child"]),
            # no tasks-result for the child: it is still "started" at run end
            lifecycle_completed_event(seq=2),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            child = None
            for _, handle in thread.interleave_projections(["subgraphs"]):
                child = handle
            assert child is not None
            assert child.status == "completed"


@pytest.mark.parametrize("channel", ["lifecycle", "tools", "input"])
def test_interleave_projections_rejects_reserved_channel(channel):
    """Reserved protocol channel names raise instead of silently no-op'ing.

    `infer_channel` treats these as first-class methods, but they have no
    interleave decoder, so routing them to the extension/`custom:` fallback
    would subscribe to a channel that never matches and yield nothing. Fail
    closed. (`updates`/`checkpoints`/`tasks` are supported and tested below.)
    """
    from streaming._events import (
        lifecycle_completed_event,
        lifecycle_started_event,
    )

    fake = SyncFakeServer()
    fake.set_state({})
    fake.script([lifecycle_started_event(seq=0), lifecycle_completed_event(seq=1)])
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with (
            threads.stream(thread_id="t-1", assistant_id="agent") as thread,
            pytest.raises(ValueError, match=channel),
        ):
            for _ in thread.interleave_projections([channel]):
                pass


def test_interleave_projections_data_channels_yield_payloads():
    """`updates`/`checkpoints`/`tasks` yield their raw `params.data` payloads."""
    from streaming._events import (
        checkpoints_event,
        lifecycle_completed_event,
        lifecycle_started_event,
        tasks_start_event,
        updates_event,
    )

    fake = SyncFakeServer()
    fake.set_state({})
    fake.script(
        [
            lifecycle_started_event(seq=0),
            updates_event(seq=1, node={"v": 1}),
            checkpoints_event(seq=2, ts="t-0", v=4),
            tasks_start_event(seq=3, task_id="task-9"),
            lifecycle_completed_event(seq=4),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            items = list(
                thread.interleave_projections(["updates", "checkpoints", "tasks"])
            )
    assert ("updates", {"node": {"v": 1}}) in items
    assert ("checkpoints", {"ts": "t-0", "v": 4}) in items
    assert any(ch == "tasks" and item.get("id") == "task-9" for ch, item in items)


def test_interleave_projections_data_channel_scoped_to_root_namespace():
    """A child-namespace checkpoint must not leak into a root interleave."""
    from streaming._events import (
        checkpoints_event,
        lifecycle_completed_event,
        lifecycle_started_event,
    )

    fake = SyncFakeServer()
    fake.set_state({"counter": 0})
    fake.script(
        [
            lifecycle_started_event(seq=0),
            checkpoints_event(seq=1, namespace=["child"], scope="child"),
            checkpoints_event(seq=2, scope="root"),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            checkpoints = [
                item
                for ch, item in thread.interleave_projections(["values", "checkpoints"])
                if ch == "checkpoints"
            ]
    assert {"scope": "root"} in checkpoints
    assert {"scope": "child"} not in checkpoints
