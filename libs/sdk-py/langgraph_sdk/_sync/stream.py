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
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

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
        explicit_thread_id: bool = False,
    ) -> None:
        self._http = http
        self._headers = dict(headers or {})
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self._explicit_thread_id = explicit_thread_id
        self._closed = False
        self._transport: SyncProtocolSseTransport | None = None
        self._controller: SyncStreamController | None = None
        self._next_command_id = 1
        self.interrupted: bool = False
        self.interrupts: list[InterruptPayload] = []
        self._lifecycle_watcher_thread: threading.Thread | None = None
        self._lifecycle_watcher_handle: SyncEventStreamHandle | None = None
        self._run_seen: bool = False
        self._run_done: _BlockingResult | None = None
        self.run = SyncRunModule(self)
        self.values = _SyncValuesProjection(self)

    def __enter__(self) -> SyncThreadStream:
        if self._closed:
            raise RuntimeError("SyncThreadStream is closed and cannot be re-entered.")
        self._transport = SyncProtocolSseTransport(
            client=self._http.client,
            thread_id=self.thread_id,
            headers=self._headers,
        )
        self._controller = SyncStreamController(self._transport)
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
        handle = self._lifecycle_watcher_handle
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.close()
        thread = self._lifecycle_watcher_thread
        if thread is not None and thread.is_alive():
            with contextlib.suppress(RuntimeError):
                thread.join(timeout=1.0)
        if self._controller is not None:
            self._controller.close()
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
                        run_done.set_result(_RunTerminal(status="errored", error=error))
                    else:
                        run_done.set_result(_RunTerminal(status="completed"))
