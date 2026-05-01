from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Any, ClassVar, Literal

from typing_extensions import NotRequired, TypedDict

_logger = logging.getLogger(__name__)


class _ProtocolEventParams(TypedDict):
    """Parameters for a protocol event.

    `timestamp` is wall-clock milliseconds since the epoch and can go
    backwards across NTP adjustments — use `ProtocolEvent.seq` for
    ordering.
    """

    namespace: list[str]
    timestamp: int
    data: Any
    interrupts: NotRequired[tuple[Any, ...]]


class ProtocolEvent(TypedDict):
    """A protocol event emitted by the streaming infrastructure.

    Wraps a raw stream part (values, messages, custom, etc.) in a uniform
    envelope with a monotonic sequence number assigned by the root StreamMux.
    Consumers that need a total order across root events should use `seq`, not
    `params.timestamp` (which is wall-clock and not monotonic).
    """

    type: Literal["event"]
    eventId: NotRequired[str]
    seq: NotRequired[int]
    method: str  # StreamMode value: "values", "messages", "custom", etc.
    params: _ProtocolEventParams


class StreamTransformer(ABC):
    """Extension point for custom stream projections.

    Transformers observe protocol events flowing through the StreamMux and
    build typed derived projections (StreamChannels, promises, etc.).

    Set `_native = True` on a transformer to have its projection keys
    exposed as direct attributes on the run stream (in addition to
    appearing in `run.extensions`).

    Subclasses must implement `init` and override at least one of
    `process` / `aprocess`. The `finalize` / `afinalize` and `fail` /
    `afail` hooks are optional — the default implementations are no-ops.
    StreamChannel instances in the projection dict are auto-closed /
    auto-failed by the mux, so most transformers don't need `finalize`
    or `fail` at all.

    Transformers that need async work pick the async lane by:

    1. Overriding `aprocess` (and optionally `afinalize` / `afail`), or
    2. Calling `self.schedule(coro)` from inside a sync `process`, or
    3. Setting `requires_async = True` explicitly.

    The mux detects these cases at registration and raises if they're
    used under sync `stream()` — they only work under `astream()`.

    Use `aprocess` when the pump must wait for async work before the
    next transformer sees the event (e.g. PII redaction that mutates
    `event` in place). Use `schedule()` for decoupled async work whose
    result lands on an independent projection (e.g. async moderation
    scoring, cost lookup, external tracing).

    Attributes:
        scope: Namespace the transformer operates within — `()` for the
            root mux. Set at construction from the mux's scope (each
            factory is called as `factory(scope)`).
        requires_async: Explicit opt-in for transformers that need a
            running event loop but don't override any async method (for
            example, transformers that call `schedule()` from a sync
            `process`). The mux also auto-detects the async lane when
            `aprocess`, `afinalize`, or `afail` is overridden.
        supports_sync: Set True only for transformers that override
            async-lane hooks while still fully supporting the sync lane.
            Such transformers may be registered under `stream()`.
        required_stream_modes: Stream modes the graph must emit for
            this transformer to have anything to process. Computed as
            the union across all registered transformers to determine
            which modes a `stream_events(version="v3")` run requests from the graph.
            Empty tuple means the transformer consumes only synthetic
            events (or is purely passive).
    """

    requires_async: ClassVar[bool] = False
    supports_sync: ClassVar[bool] = False
    required_stream_modes: ClassVar[tuple[str, ...]] = ()

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        """Initialize the transformer with its mux's scope.

        Args:
            scope: The namespace tuple the owning mux is scoped to.
                `()` for the root. Factories receive this at
                construction time (`factory(scope)` in `StreamMux`).
        """
        self.scope: tuple[str, ...] = scope

    @abstractmethod
    def init(self) -> dict[str, Any]:
        """Return the projection dict.

        Keys become entries in `run.extensions`. If the transformer has
        `_native = True`, keys are also set as direct attributes on the
        run stream.

        StreamChannel instances in the return value are automatically
        wired by the StreamMux for protocol event auto-forwarding.
        """
        ...

    def _on_register(self, mux: Any) -> None:
        """Called by `StreamMux._register` after this transformer is wired in.

        Default is a no-op. Override to capture a reference to the
        owning mux — needed for transformers that build mini-muxes
        via `mux._make_child(...)` (e.g. `SubgraphTransformer`).
        """

    def process(self, event: ProtocolEvent) -> bool:
        """Handle an event on the sync lane.

        Called for every event before it is appended to the main event
        log. Subclasses must override either `process` or `aprocess`.
        The default raises so a missing override fails loudly rather
        than silently passing every event through.

        Args:
            event: The protocol event to observe.

        Returns:
            True to keep the event in the main log, False to suppress it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override process() or aprocess()"
        )

    async def aprocess(self, event: ProtocolEvent) -> bool:
        """Handle an event on the async lane.

        The mux awaits this before dispatching to the next transformer,
        so a slow `aprocess` serializes the pipeline. Use it only when
        a later transformer — or a consumer reading the event
        synchronously — must see the result of the async work (e.g.
        PII redaction that mutates `event` in place).

        The default delegates to `process`, so purely-sync transformers
        run unchanged under `astream()`.

        Args:
            event: The protocol event to observe.

        Returns:
            True to keep the event in the main log, False to suppress it.
        """
        return self.process(event)

    def finalize(self) -> None:
        """Called when the run ends normally (sync lane).

        Override to close StreamChannels, resolve promises, or perform
        other teardown. StreamChannel instances in the projection dict
        are auto-closed by the mux.
        """

    async def afinalize(self) -> None:
        """Called when the run ends normally (async lane).

        By the time this runs, the mux has already awaited every task
        started via `schedule()`, so StreamChannels can be closed here
        without a last-task-wins race.

        The default delegates to `finalize`.
        """
        self.finalize()

    def fail(self, err: BaseException) -> None:
        """Called when the run ends with an error (sync lane).

        Override to fail StreamChannels, reject promises, or perform
        other teardown. StreamChannel instances in the projection dict
        are auto-failed by the mux.

        Args:
            err: The exception that ended the run.
        """

    async def afail(self, err: BaseException) -> None:
        """Called when the run ends with an error (async lane).

        The mux cancels and awaits every task started via `schedule()`
        before calling this, so cleanup doesn't race with in-flight work.

        The default delegates to `fail`.

        Args:
            err: The exception that ended the run.
        """
        self.fail(err)

    # ------------------------------------------------------------------
    # Scheduled async work
    # ------------------------------------------------------------------

    def schedule(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        on_error: Literal["log", "raise"] = "log",
    ) -> asyncio.Task[Any]:
        """Schedule a coroutine tied to this transformer's lifecycle.

        The mux holds the task reference, awaits all scheduled tasks
        during `aclose()` before calling `afinalize()`, and cancels
        them on `afail()`. Authors don't need to track tasks or
        implement the last-task-closes-the-log dance.

        Requires a running event loop — call only under `astream()`.
        Set `requires_async = True` on the class so registration under
        sync `stream()` fails fast with a clear message.

        Args:
            coro: The coroutine to run. Its lifecycle is owned by the
                mux from this point on.
            on_error: `"log"` (default) catches and logs any exception
                the coroutine raises, so a single failure doesn't tear
                down the run. `"raise"` lets the exception propagate
                when the mux joins pendings, converting the close path
                into the fail path.

        Returns:
            The asyncio Task. Authors rarely need to await it directly
            — consumers read results from whatever projection the
            coroutine pushes into.

        Raises:
            RuntimeError: If called without a running event loop (i.e.
                under sync `stream()` rather than `astream()`).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                f"{type(self).__name__}.schedule() requires a running "
                "event loop; this transformer must run under astream(), "
                "not stream(). Set requires_async=True on the class so "
                "this fails at registration rather than at first event."
            ) from None

        wrapped = self._wrap_scheduled(coro) if on_error == "log" else coro
        task = asyncio.create_task(wrapped)
        tasks = self._scheduled_task_set()
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return task

    @staticmethod
    async def _wrap_scheduled(coro: Coroutine[Any, Any, Any]) -> Any:
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except BaseException:
            _logger.exception("Scheduled StreamTransformer task failed")

    def _scheduled_task_set(self) -> set[asyncio.Task[Any]]:
        """Return the lazily-allocated task set.

        Avoids requiring subclasses to call `super().__init__()`.
        """
        tasks: set[asyncio.Task[Any]] | None = getattr(
            self, "_stream_scheduled_tasks", None
        )
        if tasks is None:
            tasks = set()
            self._stream_scheduled_tasks = tasks
        return tasks


def transformer_requires_async(transformer: StreamTransformer) -> bool:
    """Return True if the transformer needs a running event loop.

    A transformer requires async if it explicitly opts in
    (`requires_async = True`) or overrides any of the async-lane methods
    (`aprocess`, `afinalize`, `afail`) without also declaring that it
    supports the sync lane.

    Args:
        transformer: The transformer to inspect.

    Returns:
        True if the transformer cannot run under sync `stream()`.
    """
    if transformer.requires_async:
        return True
    if transformer.supports_sync:
        return False
    cls = type(transformer)
    for name in ("aprocess", "afinalize", "afail"):
        if getattr(cls, name) is not getattr(StreamTransformer, name):
            return True
    return False
