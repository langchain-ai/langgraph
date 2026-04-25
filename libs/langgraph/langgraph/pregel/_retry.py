from __future__ import annotations

import asyncio
import logging
import random
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from typing_extensions import NotRequired, TypedDict

from langgraph._internal._config import patch_configurable, recast_checkpoint_ns
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_RUNTIME,
    CONFIG_KEY_SEND,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_THREAD_ID,
    CONFIG_KEY_TIMED_ATTEMPT_OBSERVER,
    NS_SEP,
)
from langgraph._internal._timeout import sync_timeout_unsupported
from langgraph.errors import GraphBubbleUp, NodeTimeoutError, ParentCommand
from langgraph.runtime import ExecutionInfo, Runtime
from langgraph.types import Command, PregelExecutableTask, RetryPolicy

logger = logging.getLogger(__name__)
SUPPORTS_EXC_NOTES = sys.version_info >= (3, 11)


class _TimedAttemptPayload(TypedDict):
    execution_id: str
    task_id: str
    task_name: str
    attempt: int
    run_id: str | None
    thread_id: str | None
    checkpoint_ns: str | None
    started_at: datetime
    deadline_at: datetime
    timeout_secs: float
    event: Literal["start", "finish"]
    finished_at: NotRequired[datetime]
    status: NotRequired[Literal["success", "error"]]
    error_type: NotRequired[str | None]
    error_message: NotRequired[str | None]


class _TimedAttemptScope:
    """Guarded-config window for timed attempts.

    `close()` and the guarded send are serialized so writes from a cancelled
    background task cannot slip past the timeout boundary.
    """

    __slots__ = ("_active", "_lock")

    def __init__(self) -> None:
        self._active = True
        self._lock = threading.Lock()

    def wrap_config(self, config: RunnableConfig) -> RunnableConfig:
        configurable = config.get(CONF, {})
        if (send := configurable.get(CONFIG_KEY_SEND)) is not None:
            return patch_configurable(config, {CONFIG_KEY_SEND: self._guard_send(send)})
        return config

    def close(self) -> None:
        with self._lock:
            self._active = False

    def _guard_send(
        self, send: Callable[[Sequence[tuple[str, Any]]], None]
    ) -> Callable[[Sequence[tuple[str, Any]]], None]:
        def guarded_send(writes: Sequence[tuple[str, Any]]) -> None:
            with self._lock:
                if self._active:
                    send(writes)

        return guarded_send


def _drain_cancelled(task: asyncio.Task[Any]) -> None:
    # Mark the abandoned task's exception as retrieved so asyncio doesn't log it.
    with suppress(asyncio.CancelledError):
        task.exception()


def _start_timed_attempt(
    task: PregelExecutableTask, config: RunnableConfig, timeout_s: float
) -> _TimedAttemptPayload | None:
    configurable = config.get(CONF, {})
    callback = configurable.get(CONFIG_KEY_TIMED_ATTEMPT_OBSERVER)
    if callback is None:
        return None
    runtime = configurable.get(CONFIG_KEY_RUNTIME)
    execution_info = runtime.execution_info if isinstance(runtime, Runtime) else None
    attempt = execution_info.node_attempt if execution_info is not None else 1
    run_id = execution_info.run_id if execution_info is not None else None
    thread_id = (
        execution_info.thread_id
        if execution_info is not None
        else configurable.get(CONFIG_KEY_THREAD_ID)
    )
    checkpoint_ns = (
        execution_info.checkpoint_ns
        if execution_info is not None
        else configurable.get(CONFIG_KEY_CHECKPOINT_NS)
    )
    started_at = datetime.now(timezone.utc)
    payload: _TimedAttemptPayload = {
        "execution_id": f"run:{run_id or '-'}|task:{task.id}|attempt:{attempt}",
        "task_id": task.id,
        "task_name": task.name,
        "attempt": attempt,
        "run_id": run_id,
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "started_at": started_at,
        "deadline_at": started_at + timedelta(seconds=timeout_s),
        "timeout_secs": timeout_s,
        "event": "start",
    }
    _dispatch_observer(callback, payload)
    return payload


def _finish_timed_attempt(
    config: RunnableConfig,
    payload: _TimedAttemptPayload | None,
    error: BaseException | None = None,
) -> None:
    if payload is None:
        return
    callback = config.get(CONF, {}).get(CONFIG_KEY_TIMED_ATTEMPT_OBSERVER)
    if callback is None:
        return
    finish: _TimedAttemptPayload = {
        **payload,
        "event": "finish",
        "finished_at": datetime.now(timezone.utc),
        "status": "error" if error is not None else "success",
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }
    _dispatch_observer(callback, finish)


def _dispatch_observer(
    callback: Callable[[_TimedAttemptPayload], None], payload: _TimedAttemptPayload
) -> None:
    try:
        callback(payload)
    except Exception:
        logger.warning("Timed attempt observer failed", exc_info=True)


async def _arun_with_timeout(
    task: PregelExecutableTask,
    config: RunnableConfig,
    timeout_s: float,
    *,
    stream: bool,
) -> Any:
    scope = _TimedAttemptScope()
    scoped_config = scope.wrap_config(config)
    start = time.monotonic()
    if stream:

        async def run() -> Any:
            async for _ in task.proc.astream(task.input, scoped_config):
                pass

    else:

        async def run() -> Any:
            return await task.proc.ainvoke(task.input, scoped_config)

    bg: asyncio.Task[Any] = asyncio.create_task(run())
    try:
        return await asyncio.wait_for(asyncio.shield(bg), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        elapsed = time.monotonic() - start
        task.writes.clear()
        bg.cancel()
        bg.add_done_callback(_drain_cancelled)
        raise NodeTimeoutError(task.name, timeout_s, elapsed) from exc
    except asyncio.CancelledError:
        bg.cancel()
        bg.add_done_callback(_drain_cancelled)
        raise
    finally:
        scope.close()


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
        # this is a runtime safety net for paths (e.g. distributed runtime)
        # that may bypass that validation.
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
    timeout_s = task.timeout
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
        attempt_payload = (
            _start_timed_attempt(task, config, timeout_s)
            if timeout_s is not None
            else None
        )
        try:
            task.writes.clear()
            if timeout_s is None:
                if stream:
                    async for _ in task.proc.astream(task.input, config):
                        pass
                    break
                return await task.proc.ainvoke(task.input, config)
            result = await _arun_with_timeout(task, config, timeout_s, stream=stream)
            _finish_timed_attempt(config, attempt_payload)
            if stream:
                break
            return result
        except ParentCommand as exc:
            ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]
            cmd = exc.args[0]
            # strip task_ids from namespace for comparison (ns format: "node1|node2:task_id")
            if cmd.graph in (ns, recast_checkpoint_ns(ns), task.name):
                try:
                    for w in task.writers:
                        w.invoke(cmd, config)
                except Exception as writer_exc:
                    _finish_timed_attempt(config, attempt_payload, writer_exc)
                    raise
                _finish_timed_attempt(config, attempt_payload)
                break
            elif cmd.graph == Command.PARENT:
                # this command is for the parent graph, assign it to the parent.
                exc.args = (replace(cmd, graph=_checkpoint_ns_for_parent_command(ns)),)
            _finish_timed_attempt(config, attempt_payload)
            raise
        except GraphBubbleUp:
            _finish_timed_attempt(config, attempt_payload)
            raise
        except Exception as exc:
            _finish_timed_attempt(config, attempt_payload, exc)
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
