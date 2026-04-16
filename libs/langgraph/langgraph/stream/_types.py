from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Any, ClassVar, Literal

from typing_extensions import NotRequired, TypedDict

_logger = logging.getLogger(__name__)


class _ProtocolEventParams(TypedDict):
    """Parameters for a protocol event."""

    namespace: list[str]
    timestamp: int
    data: Any
    interrupts: NotRequired[tuple[Any, ...]]


class ProtocolEvent(TypedDict):
    """A protocol event emitted by the streaming infrastructure.

    Wraps a raw stream part (values, messages, custom, etc.) in a uniform
    envelope with a monotonic sequence number assigned by the StreamMux.
    """

    type: Literal["event"]
    eventId: NotRequired[str]
    seq: NotRequired[int]
    method: str  # StreamMode value: "values", "messages", "custom", etc.
    params: _ProtocolEventParams


class StreamTransformer(ABC):
    """Extension point for custom stream projections.

    Transformers observe protocol events flowing through the StreamMux and
    build typed derived projections (EventLogs, StreamChannels, promises, etc.).

    Set `_native = True` on a transformer to have its projection keys
    exposed as direct attributes on the run stream (in addition to
    appearing in `run.extensions`).

    Subclasses must implement `init` and override at least one of
    `process` / `aprocess`. The `finalize`/`afinalize` and `fail`/`afail`
    hooks are optional — the default implementations are no-ops.
    EventLog and StreamChannel instances in the projection dict are
    auto-closed/failed by the mux, so most transformers don't need
    ``finalize`` or ``fail`` at all.

    Async lane
    ----------
    Transformers that need async work pick the async lane by:

    1. Overriding ``aprocess`` (and optionally ``afinalize``/``afail``), or
    2. Calling ``self.schedule(coro)`` from inside a sync ``process``, or
    3. Setting ``requires_async = True`` explicitly.

    The mux detects these cases at registration and raises if they're
    used under sync ``stream()`` — they only work under ``astream()``.

    Use ``aprocess`` when the pump must *wait* for async work before
    the next transformer sees the event (e.g. PII redaction that
    mutates ``event`` in place). Use ``schedule()`` for decoupled async
    work whose result lands on an independent projection (e.g. async
    moderation scoring, cost lookup, external tracing).
    """

    #: Explicit opt-in for transformers that need a running event loop but
    #: don't override any async method (for example, transformers that call
    #: ``schedule()`` from a sync ``process``). The mux also auto-detects the
    #: async lane when ``aprocess``/``afinalize``/``afail`` is overridden.
    requires_async: ClassVar[bool] = False

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

    def process(self, event: ProtocolEvent) -> bool:
        """Sync event handler. Override for the sync lane.

        Called for every event before it is appended to the main event log.
        Return False to suppress the event from the main log.

        Subclasses must override either ``process`` or ``aprocess``. The
        default raises so a missing override fails loudly rather than
        silently passing every event through.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override process() or aprocess()"
        )

    async def aprocess(self, event: ProtocolEvent) -> bool:
        """Async event handler. Override for the async lane.

        The mux awaits this before dispatching to the next transformer,
        so a slow ``aprocess`` serializes the pipeline. Use this only
        when a later transformer — or a consumer reading the event
        synchronously — must see the result of the async work (e.g.
        PII redaction that mutates ``event`` in place).

        The default delegates to ``process``, so purely-sync transformers
        run unchanged under ``astream()``.
        """
        return self.process(event)

    def finalize(self) -> None:
        """Called when the run ends normally (sync lane).

        Override to close EventLogs, resolve promises, or perform other
        teardown. StreamChannel instances are auto-closed by the mux.
        """

    async def afinalize(self) -> None:
        """Called when the run ends normally (async lane).

        By the time this runs, the mux has already awaited every task
        started via ``schedule()``, so EventLogs can be closed here
        without a last-task-wins race.

        The default delegates to ``finalize``.
        """
        self.finalize()

    def fail(self, err: BaseException) -> None:
        """Called when the run ends with an error (sync lane).

        Override to fail EventLogs, reject promises, or perform other
        teardown. StreamChannel instances are auto-failed by the mux.
        """

    async def afail(self, err: BaseException) -> None:
        """Called when the run ends with an error (async lane).

        The mux cancels and awaits every task started via ``schedule()``
        before calling this, so cleanup doesn't race with in-flight work.

        The default delegates to ``fail``.
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
        during ``aclose()`` before calling ``afinalize()``, and cancels
        them on ``afail()``. Authors don't need to track tasks or
        implement the last-task-closes-the-log dance.

        ``on_error="log"`` (default): exceptions are caught and logged;
        a single failed call doesn't tear down the run.

        ``on_error="raise"``: exceptions propagate when the mux joins
        pendings, converting the close path into the fail path.

        Requires a running event loop — call only under ``astream()``.
        Set ``requires_async = True`` on the class so registration under
        sync ``stream()`` fails fast with a clear message.
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
        """Lazily-allocated task set. Avoids requiring super().__init__()."""
        tasks: set[asyncio.Task[Any]] | None = getattr(
            self, "_stream_scheduled_tasks", None
        )
        if tasks is None:
            tasks = set()
            self._stream_scheduled_tasks = tasks
        return tasks


def transformer_requires_async(transformer: StreamTransformer) -> bool:
    """True if the transformer needs a running event loop.

    A transformer requires async if it explicitly opts in
    (``requires_async = True``) or overrides any of the async-lane methods
    (``aprocess``, ``afinalize``, ``afail``).
    """
    if transformer.requires_async:
        return True
    cls = type(transformer)
    for name in ("aprocess", "afinalize", "afail"):
        if getattr(cls, name) is not getattr(StreamTransformer, name):
            return True
    return False
