from __future__ import annotations

import asyncio
import contextvars
import logging
import random
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig

from langgraph._internal._config import patch_configurable, recast_checkpoint_ns
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CALL,
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
from langgraph._internal._timeout import timeout_seconds
from langgraph.errors import GraphBubbleUp, NodeTimeoutError, ParentCommand
from langgraph.runtime import ExecutionInfo, Runtime
from langgraph.types import Command, PregelExecutableTask, RetryPolicy

logger = logging.getLogger(__name__)
SUPPORTS_EXC_NOTES = sys.version_info >= (3, 11)


def _timeout_seconds(value: float | timedelta | None) -> float | None:
    return timeout_seconds(value)


class _TimedAttemptScope:
    """Atomically disable writes and child scheduling after an attempt times out."""

    __slots__ = ("_active", "_lock", "node_name", "timeout_s")

    def __init__(self, node_name: str, timeout_s: float) -> None:
        self.node_name = node_name
        self.timeout_s = timeout_s
        self._active = True
        self._lock = threading.Lock()

    def wrap_config(self, config: RunnableConfig) -> RunnableConfig:
        configurable = config.get(CONF, {})
        updates: dict[str, Any] = {}
        if (send := configurable.get(CONFIG_KEY_SEND)) is not None:
            updates[CONFIG_KEY_SEND] = self._guard_send(send)
        if (call := configurable.get(CONFIG_KEY_CALL)) is not None:
            updates[CONFIG_KEY_CALL] = self._guard_call(call)
        if not updates:
            return config
        return patch_configurable(config, updates)

    def close(self) -> None:
        with self._lock:
            self._active = False

    def _guard_send(
        self, send: Callable[[Sequence[tuple[str, Any]]], None]
    ) -> Callable[[Sequence[tuple[str, Any]]], None]:
        def guarded_send(writes: Sequence[tuple[str, Any]]) -> None:
            with self._lock:
                # Hold the lock through dispatch so close() cannot race a late write
                # into the attempt after the timeout boundary has been crossed.
                if self._active:
                    send(writes)

        return guarded_send

    def _guard_call(self, call: Callable[..., Any]) -> Callable[..., Any]:
        def guarded_call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                # Same as guarded_send(): scheduling a child task must remain atomic
                # with respect to close(), or a timed-out attempt can still leak work.
                if self._active:
                    return call(*args, **kwargs)
                future: Future[Any] = Future()
                future.set_exception(
                    RuntimeError(
                        f"Timed-out attempt for node '{self.node_name}' cannot schedule child tasks."
                    )
                )
            return future

        return guarded_call


def _suppress_background_task_exception(task: asyncio.Task[Any]) -> None:
    """Drain detached task exceptions so cancelled background work stays silent."""

    with suppress(asyncio.CancelledError):
        task.exception()


def _task_timeout_payload(
    task: PregelExecutableTask, config: RunnableConfig, timeout_s: float
) -> dict[str, Any]:
    runtime = config.get(CONF, {}).get(CONFIG_KEY_RUNTIME)
    execution_info = runtime.execution_info if isinstance(runtime, Runtime) else None
    attempt = execution_info.node_attempt if execution_info is not None else 1
    run_id = execution_info.run_id if execution_info is not None else None
    started_at = datetime.now(timezone.utc)
    return {
        "execution_id": f"run:{run_id or '-'}|task:{task.id}|attempt:{attempt}",
        "task_id": task.id,
        "task_name": task.name,
        "attempt": attempt,
        "run_id": run_id,
        "thread_id": (
            execution_info.thread_id
            if execution_info is not None
            else config.get(CONF, {}).get(CONFIG_KEY_THREAD_ID)
        ),
        "checkpoint_ns": (
            execution_info.checkpoint_ns
            if execution_info is not None
            else config.get(CONF, {}).get(CONFIG_KEY_CHECKPOINT_NS)
        ),
        "started_at": started_at.isoformat(),
        "deadline_at": (started_at + timedelta(seconds=timeout_s)).isoformat(),
        "timeout_secs": timeout_s,
    }


def _notify_timed_attempt(
    config: RunnableConfig, payload: dict[str, Any] | None
) -> None:
    if payload is None:
        return
    callback = config.get(CONF, {}).get(CONFIG_KEY_TIMED_ATTEMPT_OBSERVER)
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        logger.warning("Timed attempt observer failed", exc_info=True)


def _timed_attempt_finish_payload(
    payload: dict[str, Any] | None,
    *,
    error: BaseException | None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        **payload,
        "event": "finish",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "error" if error is not None else "success",
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }


def _invoke_with_timeout(
    task: PregelExecutableTask, config: RunnableConfig, timeout_s: float | None
) -> Any:
    """Run the sync invocation under a wall-clock timeout.

    Python cannot safely cancel a running thread, so if the worker exceeds
    `timeout_s` we raise `NodeTimeoutError` and leave the thread to finish
    on its own. Writes and child task scheduling from a timed-out attempt are
    discarded, but any other side effects in user code may still continue in
    the daemon worker thread until it returns.
    """
    if timeout_s is None:
        return task.proc.invoke(task.input, config)
    scoped_attempt = _TimedAttemptScope(task.name, timeout_s)
    scoped_config = scoped_attempt.wrap_config(config)
    result: list[Any] = []
    exc: list[BaseException] = []
    done = threading.Event()
    ctx = contextvars.copy_context()

    def target() -> None:
        try:
            result.append(ctx.run(task.proc.invoke, task.input, scoped_config))
        except BaseException as e:
            exc.append(e)
        finally:
            scoped_attempt.close()
            done.set()

    start = time.monotonic()
    worker = threading.Thread(
        target=target, name=f"node:{task.name}:{task.id}", daemon=True
    )
    worker.start()
    if not done.wait(timeout_s):
        elapsed = time.monotonic() - start
        scoped_attempt.close()
        # Discard any writes emitted before the deadline — a timed-out attempt
        # is a failure, so partial writes must not leak into the commit.
        task.writes.clear()
        raise NodeTimeoutError(task.name, timeout_s, elapsed)
    if exc:
        raise exc[0]
    return result[0]


async def _ainvoke_with_timeout(
    task: PregelExecutableTask, config: RunnableConfig, timeout_s: float | None
) -> Any:
    if timeout_s is None:
        return await task.proc.ainvoke(task.input, config)
    scoped_attempt = _TimedAttemptScope(task.name, timeout_s)
    scoped_config = scoped_attempt.wrap_config(config)
    start = time.monotonic()
    invoke_task = asyncio.create_task(task.proc.ainvoke(task.input, scoped_config))
    try:
        return await asyncio.wait_for(asyncio.shield(invoke_task), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        elapsed = time.monotonic() - start
        scoped_attempt.close()
        task.writes.clear()
        invoke_task.cancel()
        invoke_task.add_done_callback(_suppress_background_task_exception)
        raise NodeTimeoutError(task.name, timeout_s, elapsed) from exc
    except asyncio.CancelledError:
        scoped_attempt.close()
        invoke_task.cancel()
        invoke_task.add_done_callback(_suppress_background_task_exception)
        raise
    finally:
        scoped_attempt.close()


async def _astream_with_timeout(
    task: PregelExecutableTask, config: RunnableConfig, timeout_s: float | None
) -> None:
    if timeout_s is None:
        async for _ in task.proc.astream(task.input, config):
            pass
        return
    scoped_attempt = _TimedAttemptScope(task.name, timeout_s)
    scoped_config = scoped_attempt.wrap_config(config)
    start = time.monotonic()

    async def _drain() -> None:
        async for _ in task.proc.astream(task.input, scoped_config):
            pass

    stream_task = asyncio.create_task(_drain())
    try:
        await asyncio.wait_for(asyncio.shield(stream_task), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        elapsed = time.monotonic() - start
        scoped_attempt.close()
        task.writes.clear()
        stream_task.cancel()
        stream_task.add_done_callback(_suppress_background_task_exception)
        raise NodeTimeoutError(task.name, timeout_s, elapsed) from exc
    except asyncio.CancelledError:
        scoped_attempt.close()
        stream_task.cancel()
        stream_task.add_done_callback(_suppress_background_task_exception)
        raise
    finally:
        scoped_attempt.close()


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
    timeout_s = _timeout_seconds(task.timeout)
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
            timed_attempt_payload = (
                {
                    **_task_timeout_payload(task, config, timeout_s),
                    "event": "start",
                }
                if timeout_s is not None
                else None
            )
            _notify_timed_attempt(config, timed_attempt_payload)
            try:
                result = _invoke_with_timeout(task, config, timeout_s)
            except (ParentCommand, GraphBubbleUp):
                raise
            except BaseException as exc:
                _notify_timed_attempt(
                    config,
                    _timed_attempt_finish_payload(
                        timed_attempt_payload,
                        error=exc,
                    ),
                )
                raise
            _notify_timed_attempt(
                config,
                _timed_attempt_finish_payload(timed_attempt_payload, error=None),
            )
            return result
        except ParentCommand as exc:
            ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]
            cmd = exc.args[0]
            # strip task_ids from namespace for comparison (ns format: "node1|node2:task_id")
            if cmd.graph in (ns, recast_checkpoint_ns(ns), task.name):
                # this command is for the current graph, handle it
                for w in task.writers:
                    w.invoke(cmd, config)
                _notify_timed_attempt(
                    config,
                    _timed_attempt_finish_payload(timed_attempt_payload, error=None),
                )
                break
            elif cmd.graph == Command.PARENT:
                # this command is for the parent graph, assign it to the parent.
                exc.args = (replace(cmd, graph=_checkpoint_ns_for_parent_command(ns)),)
            _notify_timed_attempt(
                config,
                _timed_attempt_finish_payload(timed_attempt_payload, error=None),
            )
            # bubble up
            raise
        except GraphBubbleUp:
            _notify_timed_attempt(
                config,
                _timed_attempt_finish_payload(timed_attempt_payload, error=None),
            )
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
    timeout_s = _timeout_seconds(task.timeout)
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
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            timed_attempt_payload = (
                {
                    **_task_timeout_payload(task, config, timeout_s),
                    "event": "start",
                }
                if timeout_s is not None
                else None
            )
            _notify_timed_attempt(config, timed_attempt_payload)
            if stream:
                try:
                    await _astream_with_timeout(task, config, timeout_s)
                except (ParentCommand, GraphBubbleUp):
                    raise
                except BaseException as exc:
                    _notify_timed_attempt(
                        config,
                        _timed_attempt_finish_payload(
                            timed_attempt_payload,
                            error=exc,
                        ),
                    )
                    raise
                else:
                    _notify_timed_attempt(
                        config,
                        _timed_attempt_finish_payload(
                            timed_attempt_payload,
                            error=None,
                        ),
                    )
                    # if successful, end
                    break
            else:
                try:
                    result = await _ainvoke_with_timeout(task, config, timeout_s)
                except (ParentCommand, GraphBubbleUp):
                    raise
                except BaseException as exc:
                    _notify_timed_attempt(
                        config,
                        _timed_attempt_finish_payload(
                            timed_attempt_payload,
                            error=exc,
                        ),
                    )
                    raise
                _notify_timed_attempt(
                    config,
                    _timed_attempt_finish_payload(
                        timed_attempt_payload,
                        error=None,
                    ),
                )
                return result
        except ParentCommand as exc:
            ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]
            cmd = exc.args[0]
            # strip task_ids from namespace for comparison (ns format: "node1|node2:task_id")
            if cmd.graph in (ns, recast_checkpoint_ns(ns), task.name):
                # this command is for the current graph, handle it
                for w in task.writers:
                    w.invoke(cmd, config)
                _notify_timed_attempt(
                    config,
                    _timed_attempt_finish_payload(timed_attempt_payload, error=None),
                )
                break
            elif cmd.graph == Command.PARENT:
                # this command is for the parent graph, assign it to the parent.
                exc.args = (replace(cmd, graph=_checkpoint_ns_for_parent_command(ns)),)
            _notify_timed_attempt(
                config,
                _timed_attempt_finish_payload(timed_attempt_payload, error=None),
            )
            # bubble up
            raise
        except GraphBubbleUp:
            _notify_timed_attempt(
                config,
                _timed_attempt_finish_payload(timed_attempt_payload, error=None),
            )
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
