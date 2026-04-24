from __future__ import annotations

import asyncio
import logging
import random
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Sequence
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
from langgraph._internal._timeout import SYNC_TIMEOUT_UNSUPPORTED, timeout_seconds
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
    """Guarded-config window: dispatch under a lock, atomically disable on close().

    `close()` and the guarded send are serialized so a late write cannot slip
    past the timeout boundary.
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


def _suppress_background_task_exception(task: asyncio.Task[Any]) -> None:
    """Drain detached task exceptions so cancelled background work stays silent."""

    with suppress(asyncio.CancelledError):
        task.exception()


def _task_timeout_payload(
    task: PregelExecutableTask, config: RunnableConfig, timeout_s: float
) -> _TimedAttemptPayload:
    configurable = config.get(CONF, {})
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
    return {
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


def _notify_timed_attempt(
    config: RunnableConfig, payload: _TimedAttemptPayload
) -> None:
    callback = config.get(CONF, {}).get(CONFIG_KEY_TIMED_ATTEMPT_OBSERVER)
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        logger.warning("Timed attempt observer failed", exc_info=True)


def _timed_attempt_finish_payload(
    payload: _TimedAttemptPayload,
    *,
    error: BaseException | None,
) -> _TimedAttemptPayload:
    return {
        **payload,
        "event": "finish",
        "finished_at": datetime.now(timezone.utc),
        "status": "error" if error is not None else "success",
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }


class _TimedAttempt:
    __slots__ = ("_config", "_payload")

    def __init__(
        self, config: RunnableConfig, payload: _TimedAttemptPayload | None
    ) -> None:
        self._config = config
        self._payload = payload

    @classmethod
    def start(
        cls,
        task: PregelExecutableTask,
        config: RunnableConfig,
        timeout_s: float | None,
    ) -> _TimedAttempt | None:
        if (
            timeout_s is None
            or config.get(CONF, {}).get(CONFIG_KEY_TIMED_ATTEMPT_OBSERVER) is None
        ):
            return None
        payload = _task_timeout_payload(task, config, timeout_s)
        _notify_timed_attempt(config, payload)
        return cls(config, payload)

    def finish(self, error: BaseException | None = None) -> None:
        if self._payload is None:
            return
        _notify_timed_attempt(
            self._config,
            _timed_attempt_finish_payload(self._payload, error=error),
        )
        self._payload = None


def _finish_timed_attempt(
    attempt: _TimedAttempt | None, error: BaseException | None = None
) -> None:
    if attempt is not None:
        attempt.finish(error)


async def _arun_with_timeout(
    task: PregelExecutableTask,
    config: RunnableConfig,
    timeout_s: float | None,
    run: Callable[[RunnableConfig], Coroutine[Any, Any, Any]],
) -> Any:
    if timeout_s is None:
        return await run(config)
    scoped_attempt = _TimedAttemptScope()
    scoped_config = scoped_attempt.wrap_config(config)
    start = time.monotonic()
    background_task: asyncio.Task[Any] = asyncio.create_task(run(scoped_config))
    try:
        return await asyncio.wait_for(
            asyncio.shield(background_task), timeout=timeout_s
        )
    except asyncio.TimeoutError as exc:
        elapsed = time.monotonic() - start
        scoped_attempt.close()
        task.writes.clear()
        background_task.cancel()
        background_task.add_done_callback(_suppress_background_task_exception)
        raise NodeTimeoutError(task.name, timeout_s, elapsed) from exc
    except asyncio.CancelledError:
        scoped_attempt.close()
        background_task.cancel()
        background_task.add_done_callback(_suppress_background_task_exception)
        raise
    finally:
        scoped_attempt.close()


async def _arun_task_with_timeout(
    task: PregelExecutableTask,
    config: RunnableConfig,
    timeout_s: float | None,
    *,
    stream: bool,
) -> Any:
    run: Callable[[RunnableConfig], Coroutine[Any, Any, Any]]
    if stream:

        async def drain_stream(run_config: RunnableConfig) -> None:
            async for _ in task.proc.astream(task.input, run_config):
                pass

        run = drain_stream
    else:

        async def invoke(run_config: RunnableConfig) -> Any:
            return await task.proc.ainvoke(task.input, run_config)

        run = invoke

    return await _arun_with_timeout(task, config, timeout_s, run)


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
    timeout_s = timeout_seconds(task.timeout)
    if timeout_s is not None:
        raise ValueError(f"{SYNC_TIMEOUT_UNSUPPORTED} Node {task.name!r} is sync.")
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
    timeout_s = timeout_seconds(task.timeout)
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
        timed_attempt: _TimedAttempt | None = None
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            timed_attempt = _TimedAttempt.start(task, config, timeout_s)
            # run the task
            try:
                result = await _arun_task_with_timeout(
                    task, config, timeout_s, stream=stream
                )
            except GraphBubbleUp:
                raise
            except BaseException as exc:
                _finish_timed_attempt(timed_attempt, exc)
                raise
            _finish_timed_attempt(timed_attempt)
            if stream:
                # if successful, end
                break
            return result
        except ParentCommand as exc:
            ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]
            cmd = exc.args[0]
            # strip task_ids from namespace for comparison (ns format: "node1|node2:task_id")
            if cmd.graph in (ns, recast_checkpoint_ns(ns), task.name):
                # this command is for the current graph, handle it
                try:
                    for w in task.writers:
                        w.invoke(cmd, config)
                except Exception as writer_exc:
                    _finish_timed_attempt(timed_attempt, writer_exc)
                    raise
                _finish_timed_attempt(timed_attempt)
                break
            elif cmd.graph == Command.PARENT:
                # this command is for the parent graph, assign it to the parent.
                exc.args = (replace(cmd, graph=_checkpoint_ns_for_parent_command(ns)),)
            _finish_timed_attempt(timed_attempt)
            # bubble up
            raise
        except GraphBubbleUp:
            _finish_timed_attempt(timed_attempt)
            # if interrupted, end
            raise
        except Exception as exc:
            _finish_timed_attempt(timed_attempt, exc)
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
