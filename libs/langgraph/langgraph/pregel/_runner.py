from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import inspect
import logging
import re
import threading
import time
import uuid
import weakref
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from functools import partial
from typing import (
    Any,
    Generic,
    TypeVar,
    cast,
)

from langchain_core.callbacks import Callbacks

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CALL,
    CONFIG_KEY_SCRATCHPAD,
    ERROR,
    INTERRUPT,
    NO_WRITES,
    RESUME,
    RETURN,
)
from langgraph._internal._future import chain_future, run_coroutine_threadsafe
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph._internal._typing import MISSING
from langgraph.constants import TAG_HIDDEN
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.pregel._algo import Call
from langgraph.pregel._executor import Submit
from langgraph.pregel._retry import arun_with_retry, run_with_retry
from langgraph.types import (
    CachePolicy,
    PregelExecutableTask,
    RetryPolicy,
    TimeoutPolicy,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", concurrent.futures.Future, asyncio.Future)
E = TypeVar("E", threading.Event, asyncio.Event)

# List of filenames to exclude from exception traceback
# Note: Frames will be removed if they are the last frame in traceback, recursively
EXCLUDED_FRAME_FNAMES = (
    "langgraph/pregel/retry.py",
    "langgraph/pregel/runner.py",
    "langgraph/pregel/executor.py",
    "langgraph/utils/runnable.py",
    "langchain_core/runnables/config.py",
    "concurrent/futures/thread.py",
    "concurrent/futures/_base.py",
)

SKIP_RERAISE_SET: weakref.WeakSet[concurrent.futures.Future | asyncio.Future] = (
    weakref.WeakSet()
)

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

# Patterns used to detect potentially malicious prompt content (Instruction 3).
_MALICIOUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(ignore\s+(previous|prior|above)\s+instructions?)"),
    re.compile(r"(?i)(system\s*prompt|you\s+are\s+now|act\s+as\s+)"),
    re.compile(r"(?i)(exec\s*\(|eval\s*\(|__import__\s*\()"),
    re.compile(r"(?i)(subprocess|os\.system|shell\s*=\s*True)"),
    re.compile(r"(?i)(base64\.b64decode|base64\.decodebytes)"),
    re.compile(r"(?i)(rm\s+-rf|del\s+/[sq]|format\s+c:)"),
]

# Compiled pattern for base64-encoded blobs that may hide malicious content.
_BASE64_BLOB_PATTERN = re.compile(r"[A-Za-z0-9+/]{60,}={0,2}")

# Global tool allow list.  When non-empty only tasks whose name appears in
# this set are permitted to execute (Instruction 6).  An empty set means
# "allow all" so that the default behaviour is preserved when the caller has
# not configured an explicit allow list.
TOOL_ALLOW_LIST: set[str] = set()

# Global inter-agent authentication token (Instruction 5).  When non-empty
# every inter-agent call must present this token.  An empty string disables
# the check so that the default behaviour is preserved.
AGENT_AUTH_TOKEN: str = ""


def _sanitize_task_input(task: PregelExecutableTask) -> None:
    """Validate and sanitize task inputs before they reach the LLM (Instructions 2 & 3).

    Raises ``ValueError`` if the input contains content that violates the
    input-sanitization or prompt-injection policies.
    """
    # Collect all string values from the task's input for inspection.
    candidates: list[str] = []
    if isinstance(task.input, str):
        candidates.append(task.input)
    elif isinstance(task.input, dict):
        for v in task.input.values():
            if isinstance(v, str):
                candidates.append(v)
    elif isinstance(task.input, (list, tuple)):
        for item in task.input:
            if isinstance(item, str):
                candidates.append(item)

    for text in candidates:
        # Check for known malicious patterns.
        for pattern in _MALICIOUS_PATTERNS:
            if pattern.search(text):
                logger.warning(
                    "task_input_rejected",
                    extra={
                        "task_id": task.id,
                        "task_name": task.name,
                        "pattern": pattern.pattern,
                    },
                )
                raise ValueError(
                    f"Task input for '{task.name}' contains disallowed content "
                    f"matching pattern: {pattern.pattern}"
                )
        # Check for suspicious base64 blobs.
        for blob in _BASE64_BLOB_PATTERN.findall(text):
            try:
                decoded = base64.b64decode(blob + "==").decode("utf-8", errors="replace")
                for pattern in _MALICIOUS_PATTERNS:
                    if pattern.search(decoded):
                        logger.warning(
                            "task_input_rejected_base64",
                            extra={
                                "task_id": task.id,
                                "task_name": task.name,
                            },
                        )
                        raise ValueError(
                            f"Task input for '{task.name}' contains disallowed "
                            "base64-encoded content."
                        )
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise
                # Ignore decode errors for non-base64 strings.


def _check_tool_allow_list(task: PregelExecutableTask) -> None:
    """Enforce the tool allow list policy (Instruction 6).

    Raises ``PermissionError`` when ``TOOL_ALLOW_LIST`` is non-empty and the
    task's name is not present in it.
    """
    if TOOL_ALLOW_LIST and task.name not in TOOL_ALLOW_LIST:
        logger.error(
            "task_blocked_not_in_allow_list",
            extra={"task_id": task.id, "task_name": task.name},
        )
        raise PermissionError(
            f"Task '{task.name}' is not in the tool allow list and cannot be executed."
        )


def _authenticate_agent_call(task: PregelExecutableTask) -> None:
    """Verify inter-agent call authentication (Instruction 5).

    When ``AGENT_AUTH_TOKEN`` is set the task's config must carry a matching
    ``agent_auth_token`` key inside its ``configurable`` mapping.
    """
    if not AGENT_AUTH_TOKEN:
        return
    conf = (task.config or {}).get("configurable", {})
    token = conf.get("agent_auth_token", "")
    if token != AGENT_AUTH_TOKEN:
        logger.error(
            "inter_agent_auth_failed",
            extra={"task_id": task.id, "task_name": task.name},
        )
        raise PermissionError(
            f"Inter-agent call to '{task.name}' failed authentication."
        )


def _validate_task(task: PregelExecutableTask) -> None:
    """Run all pre-execution security checks for a task."""
    _check_tool_allow_list(task)
    _authenticate_agent_call(task)
    _sanitize_task_input(task)


def _log_task_start(task: PregelExecutableTask, trace_id: str) -> None:
    """Emit a structured audit log entry when a task starts (Instructions 1 & 4)."""
    input_repr = repr(task.input)
    input_hash = hashlib.sha256(input_repr.encode("utf-8", errors="replace")).hexdigest()
    logger.info(
        "task_start",
        extra={
            "trace_id": trace_id,
            "task_id": task.id,
            "task_name": task.name,
            "input_hash": input_hash,
            "timestamp": time.time(),
        },
    )


def _log_task_end(
    task: PregelExecutableTask,
    trace_id: str,
    exception: BaseException | None,
) -> None:
    """Emit a structured audit log entry when a task ends (Instructions 1 & 4)."""
    writes_repr = repr(task.writes)
    output_hash = hashlib.sha256(writes_repr.encode("utf-8", errors="replace")).hexdigest()
    if exception is not None:
        logger.error(
            "task_end_error",
            extra={
                "trace_id": trace_id,
                "task_id": task.id,
                "task_name": task.name,
                "output_hash": output_hash,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "timestamp": time.time(),
            },
        )
    else:
        logger.info(
            "task_end_success",
            extra={
                "trace_id": trace_id,
                "task_id": task.id,
                "task_name": task.name,
                "output_hash": output_hash,
                "timestamp": time.time(),
            },
        )


class FuturesDict(Generic[F, E], dict[F, PregelExecutableTask | None]):
    event: E
    callback: weakref.ref[Callable[[PregelExecutableTask, BaseException | None], None]]
    # Stop condition is injected by PregelRunner instead of hard-coded here.
    # This lets the runner treat graph-error-handled exceptions as non-fatal
    # so `on_done` does not trigger an early stop for those futures.
    should_stop: Callable[[set[F]], bool]
    counter: int
    done: set[F]
    lock: threading.Lock

    def __init__(
        self,
        event: E,
        callback: weakref.ref[
            Callable[[PregelExecutableTask, BaseException | None], None]
        ],
        should_stop: Callable[[set[F]], bool],
        future_type: type[F],
        # used for generic typing, newer py supports FutureDict[...](...)
    ) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.event = event
        self.callback = callback
        self.should_stop = should_stop
        self.counter = 0
        self.done: set[F] = set()

    def __setitem__(
        self,
        key: F,
        value: PregelExecutableTask | None,
    ) -> None:
        super().__setitem__(key, value)  # type: ignore[index]
        if value is not None:
            with self.lock:
                self.event.clear()
                self.counter += 1
            key.add_done_callback(partial(self.on_done, value))

    def on_done(
        self,
        task: PregelExecutableTask,
        fut: F,
    ) -> None:
        # Called automatically by future.add_done_callback registered in __setitem__.
        try:
            if cb := self.callback():
                cb(task, _exception(fut))
        finally:
            with self.lock:
                self.done.add(fut)
                self.counter -= 1
                # Wake waiter when all tracked futures are done, or when runner-level
                # stop condition is met (for example, a non-handled fatal exception).
                if self.counter == 0 or self.should_stop(self.done):
                    self.event.set()


class PregelRunner:
    """Responsible for executing a set of Pregel tasks concurrently, committing
    their writes, yielding control to caller when there is output to emit, and
    interrupting other tasks if appropriate."""

    def __init__(
        self,
        *,
        submit: weakref.ref[Submit],
        put_writes: weakref.ref[Callable[[str, Sequence[tuple[str, Any]]], None]],
        use_astream: bool = False,
        node_finished: Callable[[str], None] | None = None,
        node_error_handler_map: Mapping[str, str] | None = None,
        schedule_error_handler: Callable[
            [PregelExecutableTask, BaseException], PregelExecutableTask | None
        ]
        | None = None,
        aschedule_error_handler: Callable[
            [PregelExecutableTask, BaseException],
            Awaitable[PregelExecutableTask | None],
        ]
        | None = None,
    ) -> None:
        self.submit = submit
        self.put_writes = put_writes
        self.use_astream = use_astream
        self.node_finished = node_finished
        self.node_error_handler_map = dict(node_error_handler_map or {})
        self.error_handler_nodes = set(self.node_error_handler_map.values())
        self.schedule_error_handler = schedule_error_handler
        self.aschedule_error_handler = aschedule_error_handler
        # Exception object ids that are already routed to graph-level error handler.
        # These ids are consulted by stop/panic checks to avoid re-raising handled
        # exceptions via the normal fatal path in the same run.
        self._handled_exception_ids: set[int] = set()
        # Unique trace identifier for this runner instance (Instruction 4).
        self._trace_id: str = str(uuid.uuid4())

    def _should_route_to_error_handler(self, task: PregelExecutableTask) -> bool:
        if task.name in self.error_handler_nodes:
            return False
        return task.name in self.node_error_handler_map

    def tick(
        self,
        tasks: Iterable[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: float | None = None,
        retry_policy: Sequence[RetryPolicy] | None = None,
        get_waiter: Callable[[], concurrent.futures.Future[None]] | None = None,
        schedule_task: Callable[
            [PregelExecutableTask, int, Call | None],
            PregelExecutableTask | None,
        ],
    ) -> Iterator[None]:
        tasks = tuple(tasks)
        futures = FuturesDict(
            callback=weakref.WeakMethod(self.commit),
            event=threading.Event(),
            should_stop=partial(
                _should_stop_others, handled_exception_ids=self._handled_exception_ids
            ),
            future_type=concurrent.futures.Future,
        )
        # give control back to the caller
        yield
        # fast path if single task with no timeout and no waiter
        if len(tasks) == 0:
            return
        elif len(tasks) == 1 and timeout is None and get_waiter is None:
            t = tasks[0]
            # Security: validate task before execution.
            try:
                _validate_task(t)
            except (ValueError, PermissionError) as validation_exc:
                logger.error(
                    "task_validation_failed",
                    extra={
                        "trace_id": self._trace_id,
                        "task_id": t.id,
                        "task_name": t.name,
                        "error": str(validation_exc),
                        "timestamp": time.time(),
                    },
                )
                self.commit(t, validation_exc)
                if reraise:
                    raise
                return
            scheduled_error_handler = False
            try:
                _log_task_start(t, self._trace_id)
                run_with_retry(
                    t,
                    retry_policy,
                    configurable={
                        CONFIG_KEY_CALL: partial(
                            _call,
                            weakref.ref(t),
                            retry_policy=retry_policy,
                            futures=weakref.ref(futures),
                            schedule_task=schedule_task,
                            submit=self.submit,
                        ),
                    },
                )
                _log_task_end(t, self._trace_id, None)
                self.commit(t, None)
            except Exception as exc:
                _log_task_end(t, self._trace_id, exc)
                self.commit(t, exc)
                if (
                    not isinstance(exc, GraphBubbleUp)
                    and self._should_route_to_error_handler(t)
                    and self.schedule_error_handler is not None
                ):
                    self._handled_exception_ids.add(id(exc))
                    if handler_task := self.schedule_error_handler(t, exc):
                        tasks = (handler_task,)
                        scheduled_error_handler = True
                        # Continue to the regular scheduling path for handler execution.
                if reraise and futures:
                    if id(exc) not in self._handled_exception_ids:
                        # will be re-raised after futures are done
                        fut: concurrent.futures.Future = concurrent.futures.Future()
                        fut.set_exception(exc)
                        futures.done.add(fut)
                elif reraise and id(exc) not in self._handled_exception_ids:
                    if tb := exc.__traceback__:
                        while tb.tb_next is not None and any(
                            tb.tb_frame.f_code.co_filename.endswith(name)
                            for name in EXCLUDED_FRAME_FNAMES
                        ):
                            tb = tb.tb_next
                        exc.__traceback__ = tb
                    raise
            if not futures and not scheduled_error_handler:
                # maybe `t` scheduled another task
                return
            else:
                if not scheduled_error_handler:
                    tasks = ()  # don't reschedule this task
        # add waiter task if requested
        if get_waiter is not None:
            futures[get_waiter()] = None
        # schedule tasks
        for t in tasks:
            # Security: validate task before scheduling.
            try:
                _validate_task(t)
            except (ValueError, PermissionError) as validation_exc:
                logger.error(
                    "task_validation_failed",
                    extra={
                        "trace_id": self._trace_id,
                        "task_id": t.id,
                        "task_name": t.name,
                        "error": str(validation_exc),
                        "timestamp": time.time(),
                    },
                )
                self.commit(t, validation_exc)
                if reraise:
                    raise
                continue
            _log_task_start(t, self._trace_id)
            fut = self.submit()(  # type: ignore[misc]
                run_with_retry,
                t,
                retry_policy,
                configurable={
                    CONFIG_KEY_CALL: partial(
                        _call,
                        weakref.ref(t),
                        retry_policy=retry_policy,
                        futures=weakref.ref(futures),
                        schedule_task=schedule_task,
                        submit=self.submit,
                    ),
                },
                __reraise_on_exit__=reraise,
            )
            futures[fut] = t
        # execute tasks, and wait for one to fail or all to finish.
        # each task is independent from all other concurrent tasks
        # yield updates/debug output as each task finishes
        end_time = timeout + time.monotonic() if timeout else None
        handled_futures: set[concurrent.futures.Future[Any]] = set()
        while len(futures) > (1 if get_waiter is not None else 0):
            done, inflight = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
                timeout=(max(0, end_time - time.monotonic()) if end_time else None),
            )
            if not done:
                break  # timed out
            done_for_stop: set[concurrent.futures.Future[Any]] = set()
            for fut in done:
                task = futures.pop(fut)
                if task is None:
                    # waiter task finished, schedule another
                    if inflight and get_waiter is not None:
                        futures[get_waiter()] = None
                elif (
                    (task_exc := _exception(fut))
                    and self._should_route_to_error_handler(task)
                    and not isinstance(task_exc, GraphBubbleUp)
                ):
                    self._handled_exception_ids.add(id(task_exc))
                    SKIP_RERAISE_SET.add(fut)
                    handled_futures.add(fut)
                    if self.schedule_error_handler is not None:
                        if handler_task := self.schedule_error_handler(task, task_exc):
                            # Security: validate handler task before scheduling.
                            try:
                                _validate_task(handler_task)
                            except (ValueError, PermissionError) as validation_exc:
                                logger.error(
                                    "task_validation_failed",
                                    extra={
                                        "trace_id": self._trace_id,
                                        "task_id": handler_task.id,
                                        "task_name": handler_task.name,
                                        "error": str(validation_exc),
                                        "timestamp": time.time(),
                                    },
                                )
                                self.commit(handler_task, validation_exc)
                                if reraise:
                                    raise
                                continue
                            _log_task_start(handler_task, self._trace_id)
                            handler_fut = self.submit()(  # type: ignore[misc]
                                run_with_retry,
                                handler_task,
                                retry_policy,
                                configurable={
                                    CONFIG_KEY_CALL: partial(
                                        _call,
                                        weakref.ref(handler_task),
                                        retry_policy=retry_policy,
                                        futures=weakref.ref(futures),
                                        schedule_task=schedule_task,
                                        submit=self.submit,
                                    ),
                                },
                                __reraise_on_exit__=reraise,
                            )
                            futures[handler_fut] = handler_task
                else:
                    done_for_stop.add(fut)
            else:
                # remove references to loop vars
                del fut, task
            # maybe stop other tasks
            if _should_stop_others(
                done_for_stop, handled_exception_ids=self._handled_exception_ids
            ):
                break
            # give control back to the caller
            yield
        # wait for done callbacks
        futures.event.wait(
            timeout=(max(0, end_time - time.monotonic()) if end_time else None)
        )
        # give control back to the caller
        yield
        # panic on failure or timeout
        try:
            _panic_or_proceed(
                futures.done.union(f for f, t in futures.items() if t is not None),
                panic=reraise,
                handled_exception_ids=self._handled_exception_ids,
                handled_futures=handled_futures,
            )
        except Exception as exc:
            if tb := exc.__traceback__:
                while tb.tb_next is not None and any(
                    tb.tb_frame.f_code.co_filename.endswith(name)
                    for name in EXCLUDED_FRAME_FNAMES
                ):
                    tb = tb.tb_next
                exc.__traceback__ = tb
            raise

    async def atick(
        self,
        tasks: Iterable[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: float | None = None,
        retry_policy: Sequence[RetryPolicy] | None = None,
        get_waiter: Callable[[], asyncio.Future[None]] | None = None,
        schedule_task: Callable[
            [PregelExecutableTask, int, Call | None],
            Awaitable[PregelExecutableTask | None],
        ],
    ) -> AsyncIterator[None]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = tuple(tasks)
        futures = FuturesDict(
            callback=weakref.WeakMethod(self.commit),
            event=asyncio.Event(),
            should_stop=partial(
                _should_stop_others, handled_exception_ids=self._handled_exception_ids
            ),
            future_type=asyncio.Future,
        )
        # give control back to the caller
        yield
        # fast path if single task with no waiter and no timeout
        if len(tasks) == 0:
            return
        elif len(tasks) == 1 and get_waiter is None and timeout is None:
            t = tasks[0]
            # Security: validate task before execution.
            try:
                _validate_task(t)
            except (ValueError, PermissionError) as validation_exc:
                logger.error(
                    "task_validation_failed",
                    extra={
                        "trace_id": self._trace_id,
                        "task_id": t.id,
                        "task_name": t.name,
                        "error": str(validation_exc),
                        "timestamp": time.time(),
                    },
                )
                self.commit(t, validation_exc)
                if reraise:
                    raise
                return
            scheduled_error_handler = False
            try:
                _log_task_start(t, self._trace_id)
                await arun_with_retry(
                    t,
                    retry_policy,
                    stream=self.use_astream,
                    configurable={
                        CONFIG_KEY_CALL: partial(
                            _acall,
                            weakref.ref(t),
                            stream=self.use_astream,
                            retry_policy=retry_policy,
                            futures=weakref.ref(futures),
                            schedule_task=schedule_task,
                            submit=self.submit,
                            loop=loop,
                        ),
                    },
                )
                _log_task_end(t, self._trace_id, None)
                self.commit(t, None)
            except Exception as exc:
                _log_task_end(t, self._trace_id, exc)
                self.commit(t, exc)
                if (
                    not isinstance(exc, GraphBubbleUp)
                    and self._should_route_to_error_handler(t)
                    and self.aschedule_error_handler is not None
                ):
                    self._handled_exception_ids.add(id(exc))
                    if handler_task := await self.aschedule_error_handler(t, exc):
                        tasks = (handler_task,)
                        scheduled_error_handler = True
                if reraise and futures:
                    if id(exc) not in self._handled_exception_ids:
                        # will be re-raised after futures are done
                        fut: asyncio.Future = loop.create_future()
                        fut.set_exception(exc)
                        futures.done.add(fut)
                elif reraise and id(exc) not in self._handled_exception_ids:
                    if tb := exc.__traceback__:
                        while tb.tb_next is not None and any(
                            tb.tb_frame.f_code.co_filename.endswith(name)
                            for name in EXCLUDED_FRAME_FNAMES
                        ):
                            tb = tb.tb_next
                        exc.__traceback__ = tb
                    raise
            if not futures and not scheduled_error_handler:
                # maybe `t` scheduled another task
                return
            else:
                if not scheduled_error_handler:
                    tasks = ()  # don't reschedule this task
        # add waiter task if requested
        if get_waiter is not None:
            futures[get_waiter()] = None
        # schedule tasks
        for t in tasks:
            # Security: validate task before scheduling.
            try:
                _validate_task(t)
            except (ValueError, PermissionError) as validation_exc:
                logger.error(
                    "task_validation_failed",
                    extra={
                        "trace_id": self._trace_id,
                        "task_id": t.id,
                        "task_name": t.name,
                        "error": str(validation_exc),
                        "timestamp": time.time(),
                    },
                )
                self.commit(t, validation_exc)
                if reraise:
                    raise
                continue
            _log_task_start(t, self._trace_id)
            fut = cast(
                asyncio.Future,
                self.submit()(  # type: ignore[misc]
                    arun_with_retry,
                    t,
                    retry_policy,
                    stream=self.use_astream,
                    configurable={
                        CONFIG_KEY_CALL: partial(
                            _acall,
                            weakref.ref(t),
                            retry_policy=retry_policy,
                            stream=self.use_astream,
                            futures=weakref.ref(futures),
                            schedule_task=schedule_task,
                            submit=self.submit,
                            loop=loop,
                        ),
                    },
                    __name__=t.name,
                    __cancel_on_exit__=True,
                    __reraise_on_exit__=reraise,
                ),
            )
            futures[fut] = t
        # execute tasks, and wait for one to fail or all to finish.
        # each task is independent from all other concurrent tasks
        # yield updates/debug output as each task finishes
        end_time = timeout + loop.time() if timeout else None
        handled_futures: set[asyncio.Future[Any]] = set()
        while len(futures) > (1 if get_waiter is not None else 0):
            done, inflight = await asyncio.wait(
                futures,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=(max(0, end_time - loop.time()) if end_time else None),
            )
            if not done:
                break  # timed out
            done_for_stop: set[asyncio.Future[Any]] = set()
            for fut in done:
                task = futures.pop(fut)
                if task is None:
                    # waiter task finished, schedule another
                    if inflight and get_waiter is not None:
                        futures[get_waiter()] = None
                elif (
                    (task_exc := _exception(fut))
                    and self._should_route_to_error_handler(task)
                    and not isinstance(task_exc, GraphBubbleUp)
                ):
                    self._handled_exception_ids.add(id(task_exc))
                    SKIP_RERAISE_SET.add(fut)
                    handled_futures.add(fut)
                    if self.aschedule_error_handler is not None:
                        if handler_task := await self.aschedule_error_handler(
                            task, task_exc
                        ):
                            # Security: validate handler task before scheduling.
                            try:
                                _validate_task(handler_task)
                            except (ValueError, PermissionError) as validation_exc:
                                logger.error(
                                    "task_validation_failed",
                                    extra={
                                        "trace_id": self._trace_id,
                                        "task_id": handler_task.id,
                                        "task_name": handler_task.name,
                                        "error": str(validation_exc),
                                        "timestamp": time.time(),
                                    },
                                )
                                self.commit(handler_task, validation_exc)
                                if reraise:
                                    raise
                                continue
                            _log_task_start(handler_task, self._trace_id)
                            handler_fut = cast(
                                asyncio.Future,
                                self.submit()(  # type: ignore[misc]
                                    arun_with_retry,
                                    handler_task,
                                    retry_policy,
                                    stream=self.use_astream,
                                    configurable={
                                        CONFIG_KEY_CALL: partial(
                                            _acall,
                                            weakref.ref(handler_task),
                                            retry_policy=retry_policy,
                                            stream=self.use_astream,
                                            futures=weakref.ref(futures),
                                            schedule_task=schedule_task,
                                            submit=self.submit,
                                            loop=loop,
                                        ),
                                    },
                                    __name__=handler_task.name,
                                    __cancel_on_exit__=True,
                                    __reraise_on_exit__=reraise,
                                ),
                            )
                            futures[handler_fut] = handler_task
                else:
                    done_for_stop.add(fut)
            else:
                # remove references to loop vars
                del fut, task
            # maybe stop other tasks
            if _should_stop_others(
                done_for_stop, handled_exception_ids=self._handled_exception_ids
            ):
                break
            # give control back to the caller
            yield
        # wait for done callbacks
        await asyncio.wait_for(
            futures.event.wait(),
            timeout=(max(0, end_time - loop.time()) if end_time else None),
        )
        # give control back to the caller
        yield
        # cancel waiter task
        for fut in futures:
            fut.cancel()
        # panic on failure or timeout
        try:
            _panic_or_proceed(
                futures.done.union(f for f, t in futures.items() if t is not None),
                timeout_exc_cls=asyncio.TimeoutError,
                panic=reraise,
                handled_exception_ids=self._handled_exception_ids,
                handled_futures=handled_futures,
            )
        except Exception as exc:
            if tb := exc.__traceback__:
                while tb.tb_next is not None and any(
                    tb.tb_frame.f_code.co_filename.endswith(name)
                    for name in EXCLUDED_FRAME_FNAMES
                ):
                    tb = tb.tb_next
                exc.__traceback__ = tb
            raise

    def commit(
        self,
        task: PregelExecutableTask,
        exception: BaseException | None,
    ) -> None:
        # Audit log every commit (Instruction 4).
        _log_task_end(task, self._trace_id, exception)
        if isinstance(exception, asyncio.CancelledError):
            # for cancelled tasks, also save error in task,
            # so loop can finish super-step
            task.writes.append((ERROR, exception))
            self.put_writes()(task.id, task.writes)  # type: ignore[misc]
        elif exception:
            if isinstance(exception, GraphInterrupt):
                # save interrupt to checkpointer
                if exception.args[0]:
                    writes = [(INTERRUPT, exception.args[0])]
                    if resumes := [w for w in task.writes if w[0] == RESUME]:
                        writes.extend(resumes)
                    self.put_writes()(task.id, writes)  # type: ignore[misc]
            elif isinstance(exception, GraphBubbleUp):
                # exception will be raised in _panic_or_proceed
                pass
            else:
                # save error to checkpointer
                task.writes.append((ERROR, exception))
                if self._should_route_to_error_handler(task) and not isinstance(
                    exception, GraphBubbleUp
                ):
                    # Mark early in commit path; loop-side routing may happen later.
                    self._handled_exception_ids.add(id(exception))
                self.put_writes()(task.id, task.writes)  # type: ignore[misc]
        else:
            if self.node_finished and (
                task.config is None or TAG_HIDDEN not in task.config.get("tags", [])
            ):
                self.node_finished(task.name)
            if not task.writes:
                # add no writes marker
                task.writes.append((NO_WRITES, None))
            # save task writes to checkpointer
            self.put_writes()(task.id, task.writes)  # type: ignore[misc]


def _should_stop_others(
    done: set[F],
    *,
    handled_exception_ids: set[int] | None = None,
) -> bool:
    """Check if any task failed, if so, cancel all other tasks.
    GraphInterrupts are not considered failures."""
    for fut in done:
        if fut.cancelled():
            continue
        elif exc := fut.exception():
            if (
                id(exc) not in (handled_exception_ids or set())
                and not isinstance(exc, GraphBubbleUp)
                and fut not in SKIP_RERAISE_SET
            ):
                return True

    return False


def _exception(
    fut: concurrent.futures.Future[Any] | asyncio.Future[Any],
) -> BaseException | None:
    """Return the exception from a future, without raising CancelledError."""
    if fut.cancelled():
        if isinstance(fut, asyncio.Future):
            return asyncio.CancelledError()
        else:
            return concurrent.futures.CancelledError()
    else:
        return fut.exception()


def _panic_or_proceed(
    futs: set[concurrent.futures.Future] | set[asyncio.Future],
    *,
    timeout_exc_cls: type[Exception] = TimeoutError,
    panic: bool = True,
    handled_exception_ids: set[int] | None = None,
    handled_futures: Collection[concurrent.futures.Future[Any] | asyncio.Future[Any]]
    | None = None,
) -> None:
    """Cancel remaining tasks if any failed, re-raise exception if panic is True."""
    done: set[concurrent.futures.Future[Any] | asyncio.Future[Any]] = set()
    inflight: set[concurrent.futures.Future[Any] | asyncio.Future[Any]] = set()
    for fut in futs:
        if fut.cancelled():
            continue
        elif fut.done():
            done.add(fut)
        else:
            inflight.add(fut)
    interrupts: list[GraphInterrupt] = []
    while done:
        # if any task failed
        fut = done.pop()
        if exc := _exception(fut):
            if fut in (handled_futures or set()):
                continue
            if id(exc) in (handled_exception_ids or set()):
                continue
            # cancel all pending tasks
            while inflight:
                inflight.pop().cancel()
            # raise the exception
            if panic:
                if isinstance(exc, GraphInterrupt):
                    # collect interrupts
                    interrupts.append(exc)
                elif fut not in SKIP_RERAISE_SET:
                    raise exc
    # raise combined interrupts
    if interrupts:
        raise GraphInterrupt(tuple(i for exc in interrupts for i in exc.args[0]))
    if inflight:
        # if we got here means we timed out
        while inflight:
            # cancel all pending tasks
            inflight.pop().cancel()
        # raise timeout error
        raise timeout_exc_cls("Timed out")


def _call(
    task: weakref.ref[PregelExecutableTask],
    func: Callable[[Any], Awaitable[Any] | Any],
    input: Any,
    *,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    timeout: TimeoutPolicy | None = None,
    callbacks: Callbacks = None,
    futures: weakref.ref[FuturesDict],
    schedule_task: Callable[
        [PregelExecutableTask, int, Call | None], PregelExecutableTask | None
    ],
    submit: weakref.ref[Submit],
) -> concurrent.futures.Future[Any]:
    if inspect.iscoroutinefunction(func):
        raise RuntimeError("In an sync context async tasks cannot be called")

    fut: concurrent.futures.Future | None = None
    # schedule PUSH tasks, collect futures
    scratchpad: PregelScratchpad = task().config[CONF][CONFIG_KEY_SCRATCHPAD]  # type: ignore[union-attr]
    # schedule the next task, if the callback returns one
    if next_task := schedule_task(
        task(),  # type: ignore[arg-type]
        scratchpad.call_counter(),
        Call(
            func,
            input,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            callbacks=callbacks,
            timeout=timeout,
        ),
    ):
        # Security: validate the next task before scheduling (Instructions 2, 3, 5, 6).
        _validate_task(next_task)
        if fut := next(
            (
                f
                for f, t in list(futures().items())  # type: ignore[union-attr]
                if t is not None and t == next_task.id
            ),
            None,
        ):
            # if the parent task was retried,
            # the next task might already be running
            pass
        elif next_task.writes:
            # if it already ran, return the result
            fut = concurrent.futures.Future()
            ret = next((v for c, v in next_task.writes if c == RETURN), MISSING)
            if ret is not MISSING:
                fut.set_result(ret)
            elif exc := next((v for c, v in next_task.writes if c == ERROR), None):
                fut.set_exception(
                    exc if isinstance(exc, BaseException) else Exception(exc)
                )
            else:
                fut.set_result(None)
        else:
            # schedule the next task
            logger.info(
                "inter_agent_call_scheduled",
                extra={
                    "caller_task_id": task().id if task() else None,  # type: ignore[union-attr]
                    "callee_task_id": next_task.id,
                    "callee_task_name": next_task.name,
                    "timestamp": time.time(),
                },
            )
            fut = submit()(  # type: ignore[misc]
                run_with_retry,
                next_task,
                retry_policy,
                configurable={
                    CONFIG_KEY_CALL: partial(
                        _call,
                        weakref.ref(next_task),
                        futures=futures,
                        retry_policy=retry_policy,
                        callbacks=callbacks,
                        schedule_task=schedule_task,
                        submit=submit,
                    ),
                },
                __reraise_on_exit__=False,
                # starting a new task in the next tick ensures
                # updates from this tick are committed/streamed first
                __next_tick__=True,
            )
            # exceptions for call() tasks are raised into the parent task
            # so we should not re-raise at the end of the tick
            SKIP_RERAISE_SET.add(fut)
            futures()[fut] = next_task  # type: ignore[index]
    fut = cast(asyncio.Future | concurrent.futures.Future, fut)
    # return a chained future to ensure commit() callback is called
    # before the returned future is resolved, to ensure stream order etc
    return chain_future(fut, concurrent.futures.Future())


def _acall(
    task: weakref.ref[PregelExecutableTask],
    func: Callable[[Any], Awaitable[Any] | Any],
    input: Any,
    *,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    timeout: TimeoutPolicy | None = None,
    callbacks: Callbacks = None,
    # injected dependencies
    futures: weakref.ref[FuturesDict],
    schedule_task: Callable[
        [PregelExecutableTask, int, Call | None],
        Awaitable[PregelExecutableTask | None],
    ],
    submit: weakref.ref[Submit],
    loop: asyncio.AbstractEventLoop,
    stream: bool = False,
) -> asyncio.Future[Any] | concurrent.futures.Future[Any]:
    # return a chained future to ensure commit() callback is called
    # before the returned future is resolved, to ensure stream order etc
    try:
        in_async = asyncio.current_task() is not None
    except RuntimeError:
        in_async = False
    # if in async context return an async future, otherwise return a sync future
    if in_async:
        fut: asyncio.Future[Any] | concurrent.futures.Future[Any] = asyncio.Future(
            loop=loop
        )
    else:
        fut = concurrent.futures.Future()
    # schedule the next task
    run_coroutine_threadsafe(
        _acall_impl(
            fut,
            task,
            func,
            input,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            timeout=timeout,
            callbacks=callbacks,
            futures=futures,
            schedule_task=schedule_task,
            submit=submit,
            loop=loop,
            stream=stream,
        ),
        loop,
        lazy=False,
    )
    return fut


async def _acall_impl(
    destination: asyncio.Future[Any] | concurrent.futures.Future[Any],
    task: weakref.ref[PregelExecutableTask],
    func: Callable[[Any], Awaitable[Any] | Any],
    input: Any,
    *,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    timeout: TimeoutPolicy | None = None,
    callbacks: Callbacks = None,
    # injected dependencies
    futures: weakref.ref[FuturesDict[asyncio.Future, asyncio.Event]],
    schedule_task: Callable[
        [PregelExecutableTask, int, Call | None],
        Awaitable[PregelExecutableTask | None],
    ],
    submit: weakref.ref[Submit],
    loop: asyncio.AbstractEventLoop,
    stream: bool = False,
) -> None:
    try:
        fut: asyncio.Future | None = None
        # schedule PUSH tasks, collect futures
        scratchpad: PregelScratchpad = task().config[CONF][CONFIG_KEY_SCRATCHPAD]  # type: ignore[union-attr]
        # schedule the next task, if the callback returns one
        if next_task := await schedule_task(
            task(),  # type: ignore[arg-type]
            scratchpad.call_counter(),
            Call(
                func,
                input,
                retry_policy=retry_policy,
                cache_policy=cache_policy,
                callbacks=callbacks,
                timeout=timeout,
            ),
        ):
            # Security: validate the next task before scheduling (Instructions 2, 3, 5, 6).
            _validate_task(next_task)
            if fut := next(
                (
                    f
                    for f, t in list(futures().items())  # type: ignore[union-attr]
                    if t is not None and t == next_task.id
                ),
                None,
            ):
                # if the parent task was retried,
                # the next task might already be running
                pass
            elif next_task.writes:
                # if it already ran, return the result
                fut = asyncio.Future(loop=loop)
                ret = next((v for c, v in next_task.writes if c == RETURN), MISSING)
                if ret is not MISSING:
                    fut.set_result(ret)
                elif exc := next((v for c, v in next_task.writes if c == ERROR), None):
                    fut.set_exception(
                        exc if isinstance(exc, BaseException) else Exception(exc)
                    )
                else:
                    fut.set_result(None)
            else:
                # schedule the next task
                logger.info(
                    "inter_agent_acall_scheduled",
                    extra={
                        "caller_task_id": task().id if task() else None,  # type: ignore[union-attr]
                        "callee_task_id": next_task.id,
                        "callee_task_name": next_task.name,
                        "timestamp": time.time(),
                    },
                )
                fut = cast(
                    asyncio.Future,
                    submit()(  # type: ignore[misc]
                        arun_with_retry,
                        next_task,
                        retry_policy,
                        stream=stream,
                        configurable={
                            CONFIG_KEY_CALL: partial(
                                _acall,
                                weakref.ref(next_task),
                                stream=stream,
                                futures=futures,
                                schedule_task=schedule_task,
                                submit=submit,
                                loop=loop,
                            ),
                        },
                        __name__=next_task.name,
                        __cancel_on_exit__=True,
                        __reraise_on_exit__=False,
                        # starting a new task in the next tick ensures
                        # updates from this tick are committed/streamed first
                        __next_tick__=True,
                    ),
                )
                # exceptions for call() tasks are raised into the parent task
                # so we should not re-raise at the end of the tick
                SKIP_RERAISE_SET.add(fut)
                futures()[fut] = next_task  # type: ignore[index]
        if fut is not None:
            chain_future(fut, destination)
        else:
            destination.set_exception(RuntimeError("Task not scheduled"))
    except Exception as exc:
        destination.set_exception(exc)