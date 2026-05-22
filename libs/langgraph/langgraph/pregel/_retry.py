from __future__ import annotations

import asyncio
import logging
import random
import sys
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, NamedTuple

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

from langgraph._internal._config import (
    merge_configs,
    patch_configurable,
    recast_checkpoint_ns,
)
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CALL,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_RUNTIME,
    CONFIG_KEY_SEND,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_THREAD_ID,
    CONFIG_KEY_TIMED_ATTEMPT_OBSERVER,
    NS_SEP,
)
from langgraph._internal._runnable import create_task_in_config_context
from langgraph._internal._timeout import sync_timeout_unsupported
from langgraph.errors import GraphBubbleUp, NodeTimeoutError, ParentCommand
from langgraph.pregel.protocol import StreamProtocol
from langgraph.runtime import ExecutionInfo, Runtime
from langgraph.types import Command, PregelExecutableTask, RetryPolicy, TimeoutPolicy

logger = logging.getLogger(__name__)
SUPPORTS_EXC_NOTES = sys.version_info >= (3, 11)


def _timeout_secs(value: float | timedelta) -> float:
    return value.total_seconds() if isinstance(value, timedelta) else value


@dataclass(frozen=True, slots=True)
class _ResolvedTimeout:
    run_timeout_secs: float | None
    idle_timeout_secs: float | None
    refresh_on: Literal["auto", "heartbeat"] | None


def _resolve_timeout(timeout: TimeoutPolicy) -> _ResolvedTimeout:
    idle_timeout_secs = (
        _timeout_secs(timeout.idle_timeout)
        if timeout.idle_timeout is not None
        else None
    )
    return _ResolvedTimeout(
        run_timeout_secs=(
            _timeout_secs(timeout.run_timeout)
            if timeout.run_timeout is not None
            else None
        ),
        idle_timeout_secs=idle_timeout_secs,
        refresh_on=timeout.refresh_on if idle_timeout_secs is not None else None,
    )


class _AttemptContext(NamedTuple):
    """Immutable per-attempt metadata shared across start/progress/finish events.

    Built once at attempt start and referenced (not copied) by every emitted
    `_AttemptEvent`, so per-event allocation is just the small event wrapper.

    Intentionally underscore-prefixed: this and `_AttemptEvent` are part of an
    internal observer contract consumed by langgraph-server. Do not move to
    `langgraph.types` — server imports them by this path.
    """

    task_id: str
    task_name: str
    attempt: int
    run_id: str | None
    thread_id: str | None
    checkpoint_ns: str | None
    started_at: datetime
    run_timeout_secs: float | None
    idle_timeout_secs: float | None
    refresh_on: Literal["auto", "heartbeat"] | None


@dataclass(frozen=True, slots=True)
class _AttemptEvent:
    """One lifecycle event for a timed attempt.

    Holds a reference to the shared `_AttemptContext` and the event-specific
    fields. The observer must treat this and `context` as read-only — they
    are reused across all events for the same attempt.
    """

    context: _AttemptContext
    event: Literal["start", "progress", "finish"]
    progress_at: datetime | None = None
    finished_at: datetime | None = None
    status: Literal["success", "error"] | None = None
    error_type: str | None = None
    error_message: str | None = None


class _TimedAttemptScope:
    """Guarded-config window for timed attempts.

    The wrapped config marks writes, stream events, runtime stream writer calls,
    child task scheduling, and any LangChain callback event emitted under the
    node's run as observable progress when `refresh_on="auto"`.
    `runtime.heartbeat()` exposes a manual progress signal for work that doesn't
    otherwise emit any of these, and is the only progress signal when
    `refresh_on="heartbeat"`.
    Guarded writes are serialized with `close()` so cancelled background tasks
    cannot persist writes past the timeout boundary. Stream/custom output is
    best-effort: it is dropped after close is observed, but callbacks run outside
    the lock because they may contain arbitrary user/runtime code.
    """

    __slots__ = (
        "__weakref__",
        "_active",
        "_last_progress",
        "_last_progress_emit",
        "_lock",
        "_on_progress",
        "_progress_min_interval",
        "_refresh_on",
    )

    def __init__(
        self,
        on_progress: Callable[[], None] | None = None,
        progress_min_interval: float = 0.0,
        refresh_on: Literal["auto", "heartbeat"] | None = None,
    ) -> None:
        self._active = True
        self._last_progress = time.monotonic()
        self._lock = threading.Lock()
        self._on_progress = on_progress
        self._progress_min_interval = progress_min_interval
        self._refresh_on = refresh_on
        # `-inf` so the first touch always passes the rate-limit gate.
        self._last_progress_emit: float = float("-inf")

    def wrap_config(self, config: RunnableConfig) -> RunnableConfig:
        configurable = config.get(CONF, {})
        patch: dict[str, Any] = {}
        if (send := configurable.get(CONFIG_KEY_SEND)) is not None:
            patch[CONFIG_KEY_SEND] = self._guard_send(send)
        if (stream := configurable.get(CONFIG_KEY_STREAM)) is not None:
            patch[CONFIG_KEY_STREAM] = self._guard_stream(stream)
        if (call := configurable.get(CONFIG_KEY_CALL)) is not None:
            patch[CONFIG_KEY_CALL] = self._guard_call(call)
        if isinstance(runtime := configurable.get(CONFIG_KEY_RUNTIME), Runtime):
            if self._refresh_on is not None:
                patch[CONFIG_KEY_RUNTIME] = runtime.override(
                    stream_writer=self._guard_stream_writer(runtime.stream_writer),
                    heartbeat=self.touch,
                )
            else:
                patch[CONFIG_KEY_RUNTIME] = runtime.override(
                    stream_writer=self._guard_stream_writer(runtime.stream_writer)
                )
        new_config = patch_configurable(config, patch) if patch else config
        if self._refresh_on == "auto":
            return merge_configs(
                new_config, {"callbacks": [_IdleProgressCallbackHandler(self)]}
            )
        return new_config

    def touch(self) -> None:
        # Avoid locking this hot progress path. We accept a small race window in
        # timestamp ordering because idle_timeout is expected to be coarse compared
        # with scheduler/thread timing.
        now = time.monotonic()
        self._last_progress = now
        if self._on_progress is None:
            return
        # Best-effort rate limit: a benign race may emit a duplicate progress
        # event under heavy concurrency, which observers must already tolerate
        # (callbacks fire from arbitrary threads).
        if now - self._last_progress_emit < self._progress_min_interval:
            return
        self._last_progress_emit = now
        self._on_progress()

    def close(self) -> None:
        with self._lock:
            self._active = False

    async def wait_for_idle_timeout(self, idle_timeout_s: float) -> None:
        while True:
            with self._lock:
                if not self._active:
                    return
                remaining = self._last_progress + idle_timeout_s - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.sleep(remaining)

    def _guard_send(
        self, send: Callable[[Sequence[tuple[str, Any]]], None]
    ) -> Callable[[Sequence[tuple[str, Any]]], None]:
        def guarded_send(writes: Sequence[tuple[str, Any]]) -> None:
            with self._lock:
                if self._active:
                    if writes and self._refresh_on == "auto":
                        self._last_progress = time.monotonic()
                    send(writes)

        return guarded_send

    def _guard_stream(self, stream: StreamProtocol) -> StreamProtocol:
        # No lock: stream callbacks fire from the event loop only, so the
        # active-check + write happen atomically between awaits.
        def guarded_stream(chunk: tuple[tuple[str, ...], str, Any]) -> None:
            if not self._active:
                return
            if self._refresh_on == "auto":
                self._last_progress = time.monotonic()
            stream(chunk)

        return StreamProtocol(guarded_stream, stream.modes)

    def _guard_call(self, call: Callable[..., Any]) -> Callable[..., Any]:
        # No lock: child-task scheduling happens from the event loop only.
        def guarded_call(*args: Any, **kwargs: Any) -> Any:
            if not self._active:
                raise asyncio.CancelledError
            if self._refresh_on == "auto":
                self._last_progress = time.monotonic()
            return call(*args, **kwargs)

        return guarded_call

    def _guard_stream_writer(
        self, stream_writer: Callable[[Any], None]
    ) -> Callable[[Any], None]:
        def guarded_stream_writer(chunk: Any) -> None:
            with self._lock:
                if not self._active:
                    return
                if self._refresh_on == "auto":
                    self._last_progress = time.monotonic()
            stream_writer(chunk)

        return guarded_stream_writer


class _IdleProgressCallbackHandler(BaseCallbackHandler):
    """Resets the idle timeout clock on any LangChain callback event.

    Inherits via `config["callbacks"]`, so it sees only events emitted by
    runs descended from the node's attempt — sibling nodes do not bleed
    through. Holds the scope by weakref so a child manager that outlives
    the attempt cannot keep the scope alive.
    """

    # Run inline so progress is recorded in callback emission order;
    # thread-pool dispatch would introduce extra reordering.
    run_inline = True

    def __init__(self, scope: _TimedAttemptScope) -> None:
        self._scope_ref = weakref.ref(scope)

    def _touch(self, *args: Any, **kwargs: Any) -> None:
        if (scope := self._scope_ref()) is not None:
            scope.touch()

    on_llm_start = _touch
    on_chat_model_start = _touch
    on_llm_new_token = _touch
    on_llm_end = _touch
    on_llm_error = _touch
    on_chain_start = _touch
    on_chain_end = _touch
    on_chain_error = _touch
    on_tool_start = _touch
    on_tool_end = _touch
    on_tool_error = _touch
    on_retriever_start = _touch
    on_retriever_end = _touch
    on_retriever_error = _touch
    on_agent_action = _touch
    on_agent_finish = _touch
    on_text = _touch
    on_retry = _touch
    on_custom_event = _touch


def _drain_cancelled(task: asyncio.Task[Any]) -> None:
    # Mark the abandoned task's exception as retrieved so asyncio doesn't log it.
    with suppress(asyncio.CancelledError):
        task.exception()


def _start_timed_attempt(
    task: PregelExecutableTask, config: RunnableConfig, timeout: _ResolvedTimeout
) -> _AttemptContext | None:
    configurable = config.get(CONF, {})
    callback = configurable.get(CONFIG_KEY_TIMED_ATTEMPT_OBSERVER)
    if callback is None:
        return None
    runtime = configurable.get(CONFIG_KEY_RUNTIME)
    execution_info = runtime.execution_info if isinstance(runtime, Runtime) else None
    context = _AttemptContext(
        task_id=task.id,
        task_name=task.name,
        attempt=execution_info.node_attempt if execution_info is not None else 1,
        run_id=execution_info.run_id if execution_info is not None else None,
        thread_id=execution_info.thread_id if execution_info is not None else None,
        checkpoint_ns=(
            execution_info.checkpoint_ns if execution_info is not None else None
        ),
        started_at=datetime.now(timezone.utc),
        run_timeout_secs=timeout.run_timeout_secs,
        idle_timeout_secs=timeout.idle_timeout_secs,
        refresh_on=timeout.refresh_on,
    )
    _dispatch_observer(callback, _AttemptEvent(context=context, event="start"))
    return context


def _finish_timed_attempt(
    config: RunnableConfig,
    context: _AttemptContext | None,
    error: BaseException | None = None,
) -> None:
    if context is None:
        return
    callback = config.get(CONF, {}).get(CONFIG_KEY_TIMED_ATTEMPT_OBSERVER)
    if callback is None:
        return
    _dispatch_observer(
        callback,
        _AttemptEvent(
            context=context,
            event="finish",
            finished_at=datetime.now(timezone.utc),
            status="error" if error is not None else "success",
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        ),
    )


def _emit_progress(
    callback: Callable[[_AttemptEvent], None],
    context: _AttemptContext,
) -> None:
    _dispatch_observer(
        callback,
        _AttemptEvent(
            context=context,
            event="progress",
            progress_at=datetime.now(timezone.utc),
        ),
    )


def _dispatch_observer(
    callback: Callable[[_AttemptEvent], None],
    event: _AttemptEvent,
) -> None:
    try:
        callback(event)
    except Exception:
        logger.warning("Timed attempt observer failed", exc_info=True)


async def _run_timeout_watchdog(run_timeout_s: float) -> None:
    await asyncio.sleep(run_timeout_s)
    raise asyncio.TimeoutError


async def _arun_with_timeout(
    task: PregelExecutableTask,
    config: RunnableConfig,
    timeout: _ResolvedTimeout,
    attempt_ctx: _AttemptContext | None,
    *,
    stream: bool,
) -> Any:
    run_timeout_s = timeout.run_timeout_secs
    idle_timeout_s = timeout.idle_timeout_secs
    on_progress: Callable[[], None] | None = None
    if attempt_ctx is not None:
        callback = config.get(CONF, {}).get(CONFIG_KEY_TIMED_ATTEMPT_OBSERVER)
        if callback is not None and idle_timeout_s is not None:
            on_progress = lambda: _emit_progress(callback, attempt_ctx)  # noqa: E731
    scope = _TimedAttemptScope(
        on_progress=on_progress,
        # Cap progress emission at ~4 events per idle window so token-rate
        # callbacks don't flood the observer.
        progress_min_interval=idle_timeout_s / 4 if idle_timeout_s is not None else 0.0,
        refresh_on=timeout.refresh_on,
    )
    scoped_config = scope.wrap_config(config)
    start = time.monotonic()
    if stream:
        # Yielded chunks count as progress only under `refresh_on="auto"`.
        # `refresh_on="heartbeat"` is the strict mode where only explicit
        # `runtime.heartbeat()` calls reset the idle clock.
        async def run() -> Any:
            async for _ in task.proc.astream(task.input, scoped_config):
                if timeout.refresh_on == "auto":
                    scope.touch()

    else:

        async def run() -> Any:
            return await task.proc.ainvoke(task.input, scoped_config)

    bg = create_task_in_config_context(run, scoped_config)
    watchdogs: dict[asyncio.Task[None], Literal["idle", "run"]] = {}
    if idle_timeout_s is not None:
        watchdogs[asyncio.create_task(scope.wait_for_idle_timeout(idle_timeout_s))] = (
            "idle"
        )
    if run_timeout_s is not None:
        watchdogs[asyncio.create_task(_run_timeout_watchdog(run_timeout_s))] = "run"
    try:
        done, _ = await asyncio.wait(
            {bg, *watchdogs}, return_when=asyncio.FIRST_COMPLETED
        )
        if bg in done:
            # Task completed in time.
            for watchdog in watchdogs:
                watchdog.cancel()
            # FIRST_COMPLETED can return both; a watchdog may have
            # already raised TimeoutError before we cancelled it.
            for watchdog in watchdogs:
                with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                    await watchdog
            return await bg
        # bg was not in `done`, so every member of `done` is one of our
        # watchdogs. Only a watchdog's TimeoutError converts to
        # NodeTimeoutError; any TimeoutError raised by the proc itself
        # propagates unchanged.
        for watchdog in done:
            kind = watchdogs[watchdog]
            try:
                await watchdog
            except asyncio.TimeoutError as exc:
                elapsed = time.monotonic() - start
                scope.close()
                task.writes.clear()
                bg.cancel()
                bg.add_done_callback(_drain_cancelled)
                raise NodeTimeoutError(
                    task.name,
                    elapsed,
                    kind=kind,
                    idle_timeout=idle_timeout_s,
                    run_timeout=run_timeout_s,
                ) from exc
            raise RuntimeError(
                f"{kind} timeout watchdog completed without raising TimeoutError"
            )
        raise RuntimeError("timeout wait completed without task or watchdog")
    except asyncio.CancelledError:
        scope.close()
        bg.cancel()
        for watchdog in watchdogs:
            watchdog.cancel()
        bg.add_done_callback(_drain_cancelled)
        raise
    finally:
        scope.close()
        for watchdog in watchdogs:
            watchdog.cancel()


def _ensure_execution_info(
    runtime: Runtime, config: RunnableConfig, task: PregelExecutableTask
) -> Runtime:
    """Ensure runtime has execution_info, creating one from config if needed.

    In the distributed runtime (LangGraph Platform), tasks are prepared by the
    server and deserialized in the executor, bypassing the OSS _algo.py code
    that normally creates ExecutionInfo. This function fills in execution_info
    from the task config when it's missing.
    """
    if runtime.execution_info is not None:
        return runtime
    configurable = config.get(CONF, {})
    return runtime.override(
        execution_info=ExecutionInfo(
            checkpoint_id=configurable.get(CONFIG_KEY_CHECKPOINT_ID) or "",
            checkpoint_ns=configurable.get(CONFIG_KEY_CHECKPOINT_NS) or "",
            task_id=configurable.get(CONFIG_KEY_TASK_ID) or task.id,
            thread_id=configurable.get(CONFIG_KEY_THREAD_ID),
            run_id=str(rid) if (rid := config.get("run_id")) else None,
        ),
    )


def _checkpoint_ns_for_parent_command(ns: str) -> str:
    """Return the checkpoint namespace for the parent graph.

    The checkpoint namespace is a `|`-separated path. Each segment is usually
    of the form `name:task_id` (e.g. `parent_first:<uuid>|node:<uuid>`), but the
    runtime may also insert a purely-numeric segment (e.g. `|1`) to disambiguate
    concurrent tasks (e.g. `parent_first:<uuid>|1|node:<uuid>`).

    Numeric segments are not real path levels, so we drop them before computing
    the parent namespace.
    """

    parts = ns.split(NS_SEP)

    # Drop any trailing numeric selectors for the current frame (e.g. `...|node:<id>|1`).
    while parts and parts[-1].isdigit():
        parts.pop()

    # Drop the current frame segment itself (e.g. the `node:<id>`).
    if parts:
        parts.pop()

    # Drop any trailing numeric selectors for the parent frame (e.g. `...|1|node:<id>`).
    while parts and parts[-1].isdigit():
        parts.pop()

    return NS_SEP.join(parts)


def run_with_retry(
    task: PregelExecutableTask,
    retry_policy: Sequence[RetryPolicy] | None,
    configurable: dict[str, Any] | None = None,
) -> None:
    """Run a task with retries."""
    retry_policy = task.retry_policy or retry_policy
    if task.timeout is not None:
        # `validate_timeout_supported` catches sync nodes at compile time;
        # this is a runtime safety net for paths that may bypass that validation.
        raise sync_timeout_unsupported(task.name)
    attempts = 0
    node_first_attempt_time = time.time()
    config = task.config
    if configurable is not None:
        config = patch_configurable(config, configurable)
    runtime = config.get(CONF, {}).get(CONFIG_KEY_RUNTIME)
    if isinstance(runtime, Runtime):
        runtime = _ensure_execution_info(runtime, config, task)
        config = patch_configurable(
            config,
            {
                CONFIG_KEY_RUNTIME: runtime.patch_execution_info(
                    node_first_attempt_time=node_first_attempt_time,
                )
            },
        )
    while True:
        runtime = config.get(CONF, {}).get(CONFIG_KEY_RUNTIME)
        if isinstance(runtime, Runtime):
            config = patch_configurable(
                config,
                {
                    CONFIG_KEY_RUNTIME: runtime.patch_execution_info(
                        # node_attempt is execution count (1-indexed): 1 on first run,
                        # then 2, 3, ... on subsequent retries.
                        node_attempt=attempts + 1,
                    )
                },
            )
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            return task.proc.invoke(task.input, config)
        except ParentCommand as exc:
            ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]
            cmd = exc.args[0]
            # strip task_ids from namespace for comparison (ns format: "node1|node2:task_id")
            if cmd.graph in (ns, recast_checkpoint_ns(ns), task.name):
                # this command is for the current graph, handle it
                for w in task.writers:
                    w.invoke(cmd, config)
                break
            elif cmd.graph == Command.PARENT:
                # this command is for the parent graph, assign it to the parent.
                exc.args = (replace(cmd, graph=_checkpoint_ns_for_parent_command(ns)),)
            # bubble up
            raise
        except GraphBubbleUp:
            # if interrupted, end
            raise
        except Exception as exc:
            if SUPPORTS_EXC_NOTES:
                exc.add_note(f"During task with name '{task.name}' and id '{task.id}'")
            if not retry_policy:
                raise

            # Check which retry policy applies to this exception
            matching_policy = None
            for policy in retry_policy:
                if _should_retry_on(policy, exc):
                    matching_policy = policy
                    break

            if not matching_policy:
                raise

            # attempts tracks failed tries only; it increments after a failure.
            attempts += 1
            # check if we should give up
            if attempts >= matching_policy.max_attempts:
                raise
            # sleep before retrying
            interval = matching_policy.initial_interval
            # Apply backoff factor based on attempt count
            interval = min(
                matching_policy.max_interval,
                interval * (matching_policy.backoff_factor ** (attempts - 1)),
            )

            # Apply jitter if configured
            sleep_time = (
                interval + random.uniform(0, 1) if matching_policy.jitter else interval
            )
            time.sleep(sleep_time)

            # log the retry
            logger.info(
                f"Retrying task {task.name} after {sleep_time:.2f} seconds (attempt {attempts}) after {exc.__class__.__name__} {exc}",
                exc_info=exc,
            )
            # signal subgraphs to resume (if available)
            config = patch_configurable(config, {CONFIG_KEY_RESUMING: True})


async def arun_with_retry(
    task: PregelExecutableTask,
    retry_policy: Sequence[RetryPolicy] | None,
    stream: bool = False,
    match_cached_writes: Callable[[], Awaitable[Sequence[PregelExecutableTask]]]
    | None = None,
    configurable: dict[str, Any] | None = None,
) -> None:
    """Run a task asynchronously with retries."""
    retry_policy = task.retry_policy or retry_policy
    resolved_timeout = (
        _resolve_timeout(task.timeout) if task.timeout is not None else None
    )
    attempts = 0
    node_first_attempt_time = time.time()
    config = task.config
    if configurable is not None:
        config = patch_configurable(config, configurable)
    runtime = config.get(CONF, {}).get(CONFIG_KEY_RUNTIME)
    if isinstance(runtime, Runtime):
        runtime = _ensure_execution_info(runtime, config, task)
        config = patch_configurable(
            config,
            {
                CONFIG_KEY_RUNTIME: runtime.patch_execution_info(
                    node_first_attempt_time=node_first_attempt_time,
                )
            },
        )
    if match_cached_writes is not None and task.cache_key is not None:
        for t in await match_cached_writes():
            if t is task:
                # if the task is already cached, return
                return
    while True:
        runtime = config.get(CONF, {}).get(CONFIG_KEY_RUNTIME)
        if isinstance(runtime, Runtime):
            config = patch_configurable(
                config,
                {
                    CONFIG_KEY_RUNTIME: runtime.patch_execution_info(
                        # node_attempt is execution count (1-indexed): 1 on first run,
                        # then 2, 3, ... on subsequent retries.
                        node_attempt=attempts + 1,
                    )
                },
            )
        attempt_ctx = (
            _start_timed_attempt(task, config, resolved_timeout)
            if resolved_timeout is not None
            else None
        )
        try:
            task.writes.clear()
            if resolved_timeout is None:
                if stream:
                    async for _ in task.proc.astream(task.input, config):
                        pass
                    break
                return await task.proc.ainvoke(task.input, config)
            result = await _arun_with_timeout(
                task, config, resolved_timeout, attempt_ctx, stream=stream
            )
            _finish_timed_attempt(config, attempt_ctx)
            if stream:
                # if successful, end
                break
            return result
        except ParentCommand as exc:
            ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]
            cmd = exc.args[0]
            # strip task_ids from namespace for comparison (ns format: "node1|node2:task_id")
            if cmd.graph in (ns, recast_checkpoint_ns(ns), task.name):
                try:
                    # this command is for the current graph, handle it
                    for w in task.writers:
                        w.invoke(cmd, config)
                except Exception as writer_exc:
                    _finish_timed_attempt(config, attempt_ctx, writer_exc)
                    raise
                _finish_timed_attempt(config, attempt_ctx)
                break
            elif cmd.graph == Command.PARENT:
                # this command is for the parent graph, assign it to the parent.
                exc.args = (replace(cmd, graph=_checkpoint_ns_for_parent_command(ns)),)
            _finish_timed_attempt(config, attempt_ctx)
            # bubble up the exception to the parent graph
            raise
        except GraphBubbleUp:
            # if interrupted, end
            _finish_timed_attempt(config, attempt_ctx)
            raise
        except Exception as exc:
            _finish_timed_attempt(config, attempt_ctx, exc)
            if SUPPORTS_EXC_NOTES:
                exc.add_note(f"During task with name '{task.name}' and id '{task.id}'")
            if not retry_policy:
                raise

            # Check which retry policy applies to this exception
            matching_policy = None
            for policy in retry_policy:
                if _should_retry_on(policy, exc):
                    matching_policy = policy
                    break

            if not matching_policy:
                raise

            # attempts tracks failed tries only; it increments after a failure.
            # The next execution's node_attempt is derived as attempts + 1.
            attempts += 1
            # check if we should give up
            if attempts >= matching_policy.max_attempts:
                raise
            # sleep before retrying
            interval = matching_policy.initial_interval
            # Apply backoff factor based on attempt count
            interval = min(
                matching_policy.max_interval,
                interval * (matching_policy.backoff_factor ** (attempts - 1)),
            )

            # Apply jitter if configured
            sleep_time = (
                interval + random.uniform(0, 1) if matching_policy.jitter else interval
            )
            await asyncio.sleep(sleep_time)

            # log the retry
            logger.info(
                f"Retrying task {task.name} after {sleep_time:.2f} seconds (attempt {attempts}) after {exc.__class__.__name__} {exc}",
                exc_info=exc,
            )
            # signal subgraphs to resume (if available)
            config = patch_configurable(config, {CONFIG_KEY_RESUMING: True})


def _should_retry_on(retry_policy: RetryPolicy, exc: Exception) -> bool:
    """Check if the given exception should be retried based on the retry policy."""
    if isinstance(retry_policy.retry_on, Sequence):
        return isinstance(exc, tuple(retry_policy.retry_on))
    elif isinstance(retry_policy.retry_on, type) and issubclass(
        retry_policy.retry_on, Exception
    ):
        return isinstance(exc, retry_policy.retry_on)
    elif callable(retry_policy.retry_on):
        return retry_policy.retry_on(exc)  # type: ignore[call-arg]
    else:
        raise TypeError(
            "retry_on must be an Exception class, a list or tuple of Exception classes, or a callable"
        )
