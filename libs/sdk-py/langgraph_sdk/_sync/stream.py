"""Synchronous thread-centric streaming surface for the v3 protocol.

`SyncThreadStream` is a synchronous context manager that owns a
`SyncProtocolSseTransport` for one thread, dispatches commands (`run.start`,
`run.respond`), exposes subscriptions over a single shared SSE, and surfaces
lifecycle state (`interrupted`, `interrupts`) via an always-on lifecycle watcher
thread.

Sync mirror of `libs/sdk-py/langgraph_sdk/_async/stream.py`.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_core.language_models.chat_model_stream import ChatModelStream
from langchain_protocol import Event, SubscribeParams

from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk.stream.sync_controller import SyncStreamController, _SyncSubscription
from langgraph_sdk.stream.transport.sync_http import (
    SyncEventStreamHandle,
    SyncProtocolSseTransport,
)


class InterruptPayload(TypedDict):
    """Payload surfaced when the server requests human input for a thread."""

    interrupt_id: str
    value: Any
    namespace: list[str]


@dataclass
class _RunTerminal:
    """Terminal state record resolved into `_run_done` on lifecycle completion."""

    status: Literal["completed", "errored"]
    error: BaseException | None = None


_ALL_CHANNELS: list[str] = [
    "values",
    "updates",
    "messages",
    "tools",
    "lifecycle",
    "input",
    "checkpoints",
    "tasks",
    "custom",
]


def _exact_namespace_params(
    channels: list[str],
    namespace: list[str],
) -> SubscribeParams:
    return {
        "channels": channels,
        "namespaces": [list(namespace)],
        "depth": 0,
    }


def _event_namespace(params_field: Any) -> list[str]:
    if not isinstance(params_field, dict):
        return []
    namespace = params_field.get("namespace") or []
    return list(namespace) if isinstance(namespace, list) else []


def _message_event_id(data: dict[str, Any]) -> str | None:
    message_id = data.get("id") or data.get("message_id")
    return str(message_id) if message_id is not None else None


def _message_route_key(data: dict[str, Any], fallback: str | None = None) -> str:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    run_id = metadata.get("run_id") if metadata else None
    message_id = _message_event_id(data)
    if run_id is not None:
        return f"run:{run_id}"
    if message_id is not None:
        return f"message:{message_id}"
    if fallback is not None:
        return f"message:{fallback}"
    return "__single__"


class _BlockingResult:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._value: Any = None
        self._error: BaseException | None = None

    def set_result(self, value: Any) -> None:
        if self._event.is_set():
            return
        self._value = value
        self._event.set()

    def set_exception(self, error: BaseException) -> None:
        if self._event.is_set():
            return
        self._error = error
        self._event.set()

    def result(self, timeout: float | None = None) -> Any:
        if not self._event.wait(timeout):
            raise TimeoutError("Result was not set before timeout.")
        if self._error is not None:
            raise self._error
        return self._value

    def done(self) -> bool:
        return self._event.is_set()


class SyncRunModule:
    """Command dispatcher for `run.start`.

    Bound to one `SyncThreadStream`; accesses its transport and id allocator.
    """

    def __init__(self, owner: SyncThreadStream) -> None:
        self._owner = owner

    def start(
        self,
        *,
        input: Any = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send `run.start` to the server. Returns the result (`{"run_id": ...}`)."""
        params: dict[str, Any] = {"assistant_id": self._owner.assistant_id}
        if input is not None:
            params["input"] = input
        if config is not None:
            params["config"] = config
        if metadata is not None:
            params["metadata"] = metadata
        result = self._owner._send_command("run.start", params)
        self._owner._run_seen = True
        controller = self._owner._controller
        if controller is not None and controller._run_start_gate is not None:
            controller._run_start_gate.set()
        return result

    def respond(
        self,
        response: Any,
        *,
        interrupt_id: str | None = None,
    ) -> dict[str, Any]:
        """Reply to a server-side interrupt and resume the run.

        Args:
            response: the response value forwarded as `params.response` on the wire.
            interrupt_id: optional explicit id. When omitted, requires exactly one
                outstanding interrupt.

        Raises:
            RuntimeError: no outstanding interrupts; `interrupt_id` is None but
                multiple interrupts are outstanding; or the explicit `interrupt_id`
                doesn't match any outstanding interrupt.
        """
        outstanding = self._owner.interrupts
        if interrupt_id is None:
            if len(outstanding) == 0:
                raise RuntimeError(
                    "thread.run.respond: no outstanding interrupt. Provide an "
                    "explicit `interrupt_id` or wait for `thread.interrupted`."
                )
            if len(outstanding) > 1:
                ids = [p["interrupt_id"] for p in outstanding]
                raise RuntimeError(
                    f"thread.run.respond: ambiguous — {len(outstanding)} "
                    f"outstanding interrupts ({ids!r}). Provide an explicit "
                    "`interrupt_id`."
                )
            match = outstanding[0]
        else:
            match = next(
                (p for p in outstanding if p["interrupt_id"] == interrupt_id),
                None,
            )
            if match is None:
                raise RuntimeError(
                    f"thread.run.respond: interrupt_id {interrupt_id!r} does not "
                    "match any outstanding interrupt in `thread.interrupts`."
                )
        params = {
            "interrupt_id": match["interrupt_id"],
            "namespace": match["namespace"],
            "response": response,
        }
        return self._owner._send_command("input.respond", params)


class _SyncValuesProjection:
    """Typed projection for `thread.values`.

    Supports `for snapshot in thread.values` (REST snapshot then live stream
    events) and `thread.values.get()` (delegates to `thread.output`).
    """

    def __init__(self, thread: SyncThreadStream) -> None:
        self._thread = thread

    def __iter__(self) -> Iterator[Any]:
        """Iterate over state snapshots: REST state first, then live values events."""
        return self._values_iter()

    def get(self) -> Any:
        """Return terminal state values; equivalent to `thread.output`."""
        return self._thread.output

    def _values_iter(self) -> Iterator[Any]:
        if self._thread._transport is None:
            raise RuntimeError("SyncThreadStream not entered — use `with`.")
        params: SubscribeParams = {"channels": ["values"]}
        sub = self._thread._register_subscription(params)
        try:
            self._thread._reconcile_stream(params)
            self._thread._ensure_fanout_running()
            state = self._thread._fetch_state()
            yield state["values"]
            while True:
                item = sub.queue.get()
                if item is None:
                    return
                params_field = item.get("params") or {}
                data = (
                    params_field.get("data") if isinstance(params_field, dict) else None
                )
                if data is not None:
                    yield data
        finally:
            self._thread._unregister_subscription(sub.id)


class _SyncMessagesProjection:
    """Typed projection for root-scope `thread.messages`.

    Iterating yields one `ChatModelStream` per message-start event. Each
    stream is fully dispatched before being yielded so that `str(message.text)`
    works immediately inside a `for` loop.
    """

    def __init__(
        self, thread: SyncThreadStream, namespace: list[str] | None = None
    ) -> None:
        self._thread = thread
        self._namespace = list(namespace or [])

    def __iter__(self) -> Iterator[ChatModelStream]:
        return self._messages_iter()

    def _messages_iter(self) -> Iterator[ChatModelStream]:
        if self._thread._transport is None:
            raise RuntimeError("SyncThreadStream not entered — use `with`.")
        root_inbox = self._thread._root_messages_inbox if not self._namespace else None
        if root_inbox is not None:
            yield from _drain_messages_inbox(root_inbox, self._namespace, self._thread)
            return
        params = _exact_namespace_params(["messages"], self._namespace)
        sub = self._thread._register_subscription(params)
        active: dict[str, ChatModelStream] = {}
        try:
            self._thread._reconcile_stream(params)
            self._thread._ensure_fanout_running()
            while True:
                item = sub.queue.get()
                if item is None:
                    return
                params_field = item.get("params") or {}
                if _event_namespace(params_field) != self._namespace:
                    continue
                data = (
                    params_field.get("data") if isinstance(params_field, dict) else None
                )
                if not isinstance(data, dict):
                    continue
                event_type = data.get("event")
                if event_type == "message-start":
                    message_id = _message_event_id(data)
                    key = _message_route_key(data, fallback=message_id)
                    metadata = (
                        data.get("metadata")
                        if isinstance(data.get("metadata"), dict)
                        else {}
                    )
                    stream = ChatModelStream(
                        namespace=list(self._namespace),
                        node=metadata.get("langgraph_node") if metadata else None,
                        message_id=message_id,
                    )
                    active[key] = stream
                    self._thread._register_active_message_stream(stream)
                    stream.dispatch(data)
                    # Pre-dispatch all remaining events for this message so the
                    # caller can access str(message.text) inside a for loop.
                    while not stream._done:
                        next_item = sub.queue.get()
                        if next_item is None:
                            sub.queue.put(None)
                            break
                        next_params = next_item.get("params") or {}
                        next_data = (
                            next_params.get("data")
                            if isinstance(next_params, dict)
                            else None
                        )
                        if not isinstance(next_data, dict):
                            continue
                        next_event_type = next_data.get("event")
                        next_key = _message_route_key(next_data)
                        target = active.get(next_key)
                        if target is None and len(active) == 1:
                            target = next(iter(active.values()))
                        if target is not None:
                            target.dispatch(next_data)
                            if next_event_type in ("message-finish", "error"):
                                self._thread._unregister_active_message_stream(target)
                                for rk, cand in list(active.items()):
                                    if cand is target:
                                        del active[rk]
                    yield stream
                else:
                    key = _message_route_key(data)
                    stream = active.get(key)
                    if stream is None and len(active) == 1:
                        stream = next(iter(active.values()))
                    if stream is None:
                        continue
                    stream.dispatch(data)
                    if event_type in ("message-finish", "error"):
                        self._thread._unregister_active_message_stream(stream)
                        for route_key, candidate in list(active.items()):
                            if candidate is stream:
                                del active[route_key]
        finally:
            for s in active.values():
                self._thread._unregister_active_message_stream(s)
            self._thread._unregister_subscription(sub.id)


def _drain_messages_inbox(
    inbox: queue.Queue[Event | None],
    namespace: list[str],
    thread: SyncThreadStream,
) -> Iterator[ChatModelStream]:
    """Drain a pre-filled inbox of messages events, yielding one stream per message."""
    active: dict[str, ChatModelStream] = {}
    try:
        while True:
            item = inbox.get()
            if item is None:
                return
            params_field = item.get("params") or {}
            data = params_field.get("data") if isinstance(params_field, dict) else None
            if not isinstance(data, dict):
                continue
            event_type = data.get("event")
            if event_type == "message-start":
                message_id = _message_event_id(data)
                key = _message_route_key(data, fallback=message_id)
                metadata = (
                    data.get("metadata")
                    if isinstance(data.get("metadata"), dict)
                    else {}
                )
                stream = ChatModelStream(
                    namespace=list(namespace),
                    node=metadata.get("langgraph_node") if metadata else None,
                    message_id=message_id,
                )
                active[key] = stream
                thread._register_active_message_stream(stream)
                stream.dispatch(data)
                yield stream
            else:
                key = _message_route_key(data)
                stream = active.get(key)
                if stream is None and len(active) == 1:
                    stream = next(iter(active.values()))
                if stream is None:
                    continue
                stream.dispatch(data)
                if event_type in ("message-finish", "error"):
                    thread._unregister_active_message_stream(stream)
                    for route_key, candidate in list(active.items()):
                        if candidate is stream:
                            del active[route_key]
    finally:
        for s in active.values():
            thread._unregister_active_message_stream(s)


class SyncToolCallHandle:
    """Sync handle for one root-scope tool call."""

    def __init__(
        self,
        *,
        tool_call_id: str,
        name: str,
        input: Any = None,
        namespace: list[str] | None = None,
    ) -> None:
        self.tool_call_id = tool_call_id
        self.name = name
        self.input = input
        self.namespace = list(namespace or [])
        self.done = False
        self.error: BaseException | None = None
        self._result: _BlockingResult = _BlockingResult()
        self._deltas: queue.Queue[str | None] = queue.Queue()

    @property
    def output(self) -> Any:
        """Block until the tool call completes and return its output."""
        return self._result.result()

    @property
    def deltas(self) -> Iterator[str]:
        """Iterate over tool output deltas emitted before the terminal event."""
        return self._delta_iter()

    def _delta_iter(self) -> Iterator[str]:
        while True:
            item = self._deltas.get()
            if item is None:
                return
            yield item

    def _push_delta(self, delta: str) -> None:
        if self.done:
            return
        self._deltas.put_nowait(delta)

    def _finish(self, output: Any) -> None:
        if self.done:
            return
        self.done = True
        self._result.set_result(output)
        self._deltas.put_nowait(None)

    def _fail(self, err: BaseException) -> None:
        if self.done:
            return
        self.done = True
        self.error = err
        self._result.set_exception(err)
        self._deltas.put_nowait(None)


class _SyncToolCallsProjection:
    """Typed projection for root-scope `thread.tool_calls`."""

    def __init__(
        self, thread: SyncThreadStream, namespace: list[str] | None = None
    ) -> None:
        self._thread = thread
        self._namespace = list(namespace or [])

    def __iter__(self) -> Iterator[SyncToolCallHandle]:
        return self._tool_calls_iter()

    def _tool_calls_iter(self) -> Iterator[SyncToolCallHandle]:
        if self._thread._transport is None:
            raise RuntimeError("SyncThreadStream not entered — use `with`.")
        params = _exact_namespace_params(["tools"], self._namespace)
        sub = self._thread._register_subscription(params)
        active: dict[str, SyncToolCallHandle] = {}
        try:
            self._thread._reconcile_stream(params)
            self._thread._ensure_fanout_running()
            while True:
                item = sub.queue.get()
                if item is None:
                    return
                params_field = item.get("params") or {}
                if _event_namespace(params_field) != self._namespace:
                    continue
                data = (
                    params_field.get("data") if isinstance(params_field, dict) else None
                )
                if not isinstance(data, dict):
                    continue
                event_type = data.get("event")
                tool_call_id = data.get("tool_call_id")
                if not isinstance(tool_call_id, str):
                    continue
                if event_type == "tool-started":
                    tool_name = data.get("tool_name")
                    if not isinstance(tool_name, str):
                        tool_name = ""
                    handle = SyncToolCallHandle(
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        input=data.get("input"),
                        namespace=list(self._namespace),
                    )
                    active[tool_call_id] = handle
                    self._thread._register_active_tool_call(handle)
                    # Pre-dispatch events until this tool call completes so that
                    # `call.output` is resolved when the caller receives the handle.
                    while not handle.done:
                        next_item = sub.queue.get()
                        if next_item is None:
                            sub.queue.put(None)
                            break
                        next_params = next_item.get("params") or {}
                        if _event_namespace(next_params) != self._namespace:
                            continue
                        next_data = (
                            next_params.get("data")
                            if isinstance(next_params, dict)
                            else None
                        )
                        if not isinstance(next_data, dict):
                            continue
                        next_event_type = next_data.get("event")
                        next_tcid = next_data.get("tool_call_id")
                        if not isinstance(next_tcid, str):
                            continue
                        if next_event_type == "tool-output-delta":
                            h = active.get(next_tcid)
                            delta = next_data.get("delta")
                            if h is not None and isinstance(delta, str):
                                h._push_delta(delta)
                        elif next_event_type == "tool-finished":
                            h = active.pop(next_tcid, None)
                            if h is not None:
                                self._thread._unregister_active_tool_call(h)
                                h._finish(next_data.get("output"))
                        elif next_event_type == "tool-error":
                            h = active.pop(next_tcid, None)
                            if h is not None:
                                self._thread._unregister_active_tool_call(h)
                                message = next_data.get("message")
                                h._fail(
                                    RuntimeError(
                                        str(message) if message else "Tool call errored"
                                    )
                                )
                    yield handle
                elif event_type == "tool-output-delta":
                    h = active.get(tool_call_id)
                    delta = data.get("delta")
                    if h is not None and isinstance(delta, str):
                        h._push_delta(delta)
                elif event_type == "tool-finished":
                    h = active.pop(tool_call_id, None)
                    if h is not None:
                        self._thread._unregister_active_tool_call(h)
                        h._finish(data.get("output"))
                elif event_type == "tool-error":
                    h = active.pop(tool_call_id, None)
                    if h is not None:
                        self._thread._unregister_active_tool_call(h)
                        message = data.get("message")
                        h._fail(
                            RuntimeError(
                                str(message) if message else "Tool call errored"
                            )
                        )
        finally:
            err = RuntimeError("Tool call stream closed before terminal tool event.")
            for h in active.values():
                self._thread._unregister_active_tool_call(h)
                h._fail(err)
            self._thread._unregister_subscription(sub.id)


class SyncThreadStream:
    """Synchronous context manager for one thread's v3 streaming session.

    Construct via `client.threads.stream(thread_id=None, *, assistant_id, ...)`
    rather than instantiating directly.
    """

    def __init__(
        self,
        *,
        http: SyncHttpClient,
        thread_id: str,
        assistant_id: str,
        headers: Mapping[str, str] | None = None,
        run_start_timeout: float | None = None,
        explicit_thread_id: bool = False,
    ) -> None:
        self._http = http
        self._headers = dict(headers or {})
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self._run_start_timeout = run_start_timeout
        self._explicit_thread_id = explicit_thread_id
        self._closed = False
        self._transport: SyncProtocolSseTransport | None = None
        self._controller: SyncStreamController | None = None
        self._command_id_lock = threading.Lock()
        self._next_command_id = 1
        self.interrupted: bool = False
        self.interrupts: list[InterruptPayload] = []
        self._lifecycle_watcher_thread: threading.Thread | None = None
        self._lifecycle_watcher_handle: SyncEventStreamHandle | None = None
        self._run_seen: bool = False
        self._run_done: _BlockingResult | None = None
        self._active_message_streams: set[ChatModelStream] = set()
        self._active_tool_calls: set[SyncToolCallHandle] = set()
        self._root_messages_inbox: queue.Queue[Event | None] | None = None
        self.run = SyncRunModule(self)
        self.values = _SyncValuesProjection(self)
        self.messages = _SyncMessagesProjection(self, namespace=[])
        self.tool_calls = _SyncToolCallsProjection(self, namespace=[])

    def __enter__(self) -> SyncThreadStream:
        if self._closed:
            raise RuntimeError("SyncThreadStream is closed and cannot be re-entered.")
        self._transport = SyncProtocolSseTransport(
            client=self._http.client,
            thread_id=self.thread_id,
            headers=self._headers,
        )
        # Gate is unset; SyncRunModule.start (or an explicit set) clears it so
        # that subscriptions opening before run.start block until the server
        # has accepted the run command.
        run_start_gate = threading.Event()
        self._controller = SyncStreamController(
            self._transport, run_start_gate=run_start_gate
        )
        self._run_done = _BlockingResult()
        self._ensure_lifecycle_watcher_running()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def output(self) -> Any:
        """Fetch terminal thread state (blocking); waits for lifecycle completion.

        Raises:
            RuntimeError: stream not entered, or no run started and no explicit
                thread_id was provided.
        """
        if self._can_return_existing_state_immediately():
            state = self._fetch_state()
            if self._state_is_terminal(state):
                return state["values"]
        terminal = self._wait_for_run_done()
        if terminal.error is not None:
            raise terminal.error
        state = self._fetch_state()
        return state["values"]

    @property
    def events(self) -> Iterator[Event]:
        """Raw iterator of every `Event` the server emits for this thread."""
        if self._transport is None:
            raise RuntimeError("SyncThreadStream not entered — use `with`.")
        return self.subscribe(_ALL_CHANNELS)

    def close(self) -> None:
        """Tear down the thread stream. Idempotent."""
        if self._closed:
            return
        self._closed = True
        run_done = self._run_done
        if run_done is not None and not run_done.done():
            run_done.set_exception(RuntimeError("SyncThreadStream closed"))
        # Close the controller first so active subscription iterators receive
        # their None sentinel immediately, before the lifecycle watcher join
        # (which may block for up to 1s).
        if self._controller is not None:
            self._controller.close()
        handle = self._lifecycle_watcher_handle
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.close()
        thread = self._lifecycle_watcher_thread
        if thread is not None and thread.is_alive():
            with contextlib.suppress(RuntimeError):
                thread.join(timeout=1.0)
        self._fail_active_message_streams(RuntimeError("SyncThreadStream closed"))
        self._fail_active_tool_calls(RuntimeError("SyncThreadStream closed"))
        if self._transport is not None:
            self._transport.close()

    # ------------------------------------------------------------------
    # Delegation to SyncStreamController
    # ------------------------------------------------------------------

    def _register_subscription(self, params: SubscribeParams) -> _SyncSubscription:
        if self._controller is None:
            raise RuntimeError("SyncThreadStream not entered — use `with`.")
        return self._controller.register_subscription(params)

    def _unregister_subscription(self, subscription_id: int) -> None:
        if self._controller is not None:
            self._controller.unregister_subscription(subscription_id)

    def _ensure_fanout_running(self) -> None:
        if self._controller is not None:
            self._controller.ensure_fanout_running()

    def _reconcile_stream(self, candidate_filter: SubscribeParams) -> None:
        if self._controller is None:
            raise RuntimeError("SyncThreadStream not entered — use `with`.")
        self._controller.reconcile_stream(candidate_filter)

    def _activate_root_messages_inbox(self) -> queue.Queue[Event | None]:
        if self._root_messages_inbox is None:
            self._root_messages_inbox = queue.Queue()
        return self._root_messages_inbox

    def _register_active_message_stream(self, stream: ChatModelStream) -> None:
        self._active_message_streams.add(stream)

    def _unregister_active_message_stream(self, stream: ChatModelStream) -> None:
        self._active_message_streams.discard(stream)

    def _fail_active_message_streams(self, err: BaseException) -> None:
        for stream in list(self._active_message_streams):
            stream.fail(err)
        self._active_message_streams.clear()

    def _register_active_tool_call(self, handle: SyncToolCallHandle) -> None:
        self._active_tool_calls.add(handle)

    def _unregister_active_tool_call(self, handle: SyncToolCallHandle) -> None:
        self._active_tool_calls.discard(handle)

    def _fail_active_tool_calls(self, err: BaseException) -> None:
        for handle in list(self._active_tool_calls):
            handle._fail(err)
        self._active_tool_calls.clear()

    def subscribe(
        self,
        channels: list[str],
        *,
        namespaces: list[list[str]] | None = None,
        depth: int | None = None,
    ) -> Iterator[Event]:
        """Open a typed subscription against the shared SSE.

        Returns an iterator that yields raw `Event` dicts matching the given
        filter. Multiple concurrent subscribes share one HTTP connection whose
        union expands or rotates as subscriptions come and go.
        """
        if self._transport is None:
            raise RuntimeError("SyncThreadStream not entered — use `with`.")
        params: SubscribeParams = {"channels": list(channels)}
        if namespaces is not None:
            params["namespaces"] = namespaces
        if depth is not None:
            params["depth"] = depth
        return self._subscription_iter(params)

    def _subscription_iter(self, params: SubscribeParams) -> Iterator[Event]:
        sub = self._register_subscription(params)
        try:
            if self._closed:
                return
            self._reconcile_stream(params)
            self._ensure_fanout_running()
            while True:
                item = sub.queue.get()
                if item is None:
                    return
                yield item
        finally:
            self._unregister_subscription(sub.id)

    def _send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a protocol command and return the `result` payload."""
        if self._transport is None:
            raise RuntimeError("SyncThreadStream not entered — use `with`.")
        with self._command_id_lock:
            command_id = self._next_command_id
            self._next_command_id += 1
        response = self._transport.send_command(
            {"id": command_id, "method": method, "params": params}
        )
        if response is None:
            return {}
        if response.get("type") == "error":
            code = response.get("error", "unknown")
            message = response.get("message", "")
            raise RuntimeError(f"Protocol error [{code}]: {message}")
        meta = response.get("meta")
        if isinstance(meta, dict):
            applied_through_seq = meta.get("applied_through_seq")
            if self._controller is not None:
                self._controller.observe_applied_through_seq(applied_through_seq)
        return response.get("result", {})

    def _ensure_lifecycle_watcher_running(self) -> None:
        if self._lifecycle_watcher_thread is not None:
            return
        self._lifecycle_watcher_thread = threading.Thread(
            target=self._run_lifecycle_watcher,
            name="langgraph-sdk-sync-lifecycle",
            daemon=True,
        )
        self._lifecycle_watcher_thread.start()

    def _run_lifecycle_watcher(self) -> None:
        """Always-on thread consuming lifecycle + input channels."""
        if self._transport is None:
            return
        try:
            handle = self._transport.open_event_stream(
                {"channels": ["lifecycle", "input"]}
            )
            self._lifecycle_watcher_handle = handle
            for event in handle.events:
                if self._closed:
                    return
                self._apply_lifecycle_event(event)
        except Exception as exc:
            run_done = self._run_done
            if run_done is not None and not run_done.done():
                run_done.set_result(
                    _RunTerminal(
                        status="errored",
                        error=RuntimeError(f"Lifecycle transport failed: {exc}"),
                    )
                )

    def _fetch_state(self) -> dict[str, Any]:
        """Fetch the current thread state from the REST endpoint."""
        return self._http.get(
            f"/threads/{self.thread_id}/state",
            headers=self._headers or None,
        )

    def _state_is_terminal(self, state: dict[str, Any]) -> bool:
        """Return `True` if the thread state has no pending tasks or next nodes."""
        return not state.get("next") and not state.get("tasks")

    def _can_return_existing_state_immediately(self) -> bool:
        """Return `True` if we can try the REST state before waiting on the lifecycle."""
        return self._explicit_thread_id and not self._run_seen

    def _wait_for_run_done(self) -> _RunTerminal:
        """Block until lifecycle completion.

        Raises:
            RuntimeError: stream not entered, or no run started and no explicit
                thread_id was provided.
        """
        if self._run_done is None:
            raise RuntimeError("SyncThreadStream not entered — use `with`.")
        if not self._run_seen and not self._explicit_thread_id:
            raise RuntimeError(
                "thread.output: no run has been started and no explicit thread_id "
                "was provided. Call thread.run.start() first."
            )
        return self._run_done.result()

    def _apply_lifecycle_event(self, event: Event) -> None:
        """Update `interrupted` / `interrupts` / `_run_done` from a lifecycle or input event."""
        method = event.get("method")
        if method == "input.requested":
            params = event.get("params") or {}
            data = params.get("data") if isinstance(params, dict) else None
            interrupt_id = data.get("interrupt_id") if isinstance(data, dict) else None
            if isinstance(interrupt_id, str):
                payload: InterruptPayload = {
                    "interrupt_id": interrupt_id,
                    "value": data.get("value") if isinstance(data, dict) else None,
                    "namespace": params.get("namespace") or []
                    if isinstance(params, dict)
                    else [],
                }
                self.interrupts.append(payload)
                self.interrupted = True
        elif method == "lifecycle":
            params = event.get("params") or {}
            data = params.get("data") if isinstance(params, dict) else None
            phase = data.get("phase") if isinstance(data, dict) else None
            if phase in ("started", "running"):
                self._run_seen = True
            elif phase in ("completed", "errored"):
                self.interrupted = False
                self.interrupts = []
                run_done = self._run_done
                if run_done is not None and not run_done.done():
                    if phase == "errored":
                        error_msg = (
                            data.get("error") if isinstance(data, dict) else None
                        )
                        error = RuntimeError(
                            f"Run errored: {error_msg}" if error_msg else "Run errored"
                        )
                        self._fail_active_message_streams(error)
                        self._fail_active_tool_calls(error)
                        run_done.set_result(_RunTerminal(status="errored", error=error))
                    else:
                        run_done.set_result(_RunTerminal(status="completed"))
