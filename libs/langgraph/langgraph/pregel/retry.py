import asyncio
import logging
import random
import sys
import time
from dataclasses import replace
from typing import Any, Optional, Sequence

from langgraph.constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_RESUMING,
    NS_SEP,
)
from langgraph.errors import _SEEN_CHECKPOINT_NS, GraphBubbleUp, ParentCommand
from langgraph.types import Command, PregelExecutableTask, RetryPolicy
from langgraph.utils.config import patch_configurable

logger = logging.getLogger(__name__)
SUPPORTS_EXC_NOTES = sys.version_info >= (3, 11)


def run_with_retry(
    task: PregelExecutableTask,
    retry_policy: Optional[RetryPolicy],
    configurable: Optional[dict[str, Any]] = None,
) -> None:
    """Run a task with retries."""
    retry_policy = task.retry_policy or retry_policy
    interval = retry_policy.initial_interval if retry_policy else 0
    attempts = 0
    config = task.config
    if configurable is not None:
        config = patch_configurable(config, configurable)
    while True:
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            return task.proc.invoke(task.input, config)
        except ParentCommand as exc:
            ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]
            cmd = exc.args[0]
            if cmd.graph == ns:
                # this command is for the current graph, handle it
                for w in task.writers:
                    w.invoke(cmd, config)
                break
            elif cmd.graph == Command.PARENT:
                # this command is for the parent graph, assign it to the parent
                parent_ns = NS_SEP.join(ns.split(NS_SEP)[:-1])
                exc.args = (replace(cmd, graph=parent_ns),)
            # bubble up
            raise
        except GraphBubbleUp:
            # if interrupted, end
            raise
        except Exception as exc:
            if SUPPORTS_EXC_NOTES:
                exc.add_note(f"During task with name '{task.name}' and id '{task.id}'")
            if retry_policy is None:
                raise
            # increment attempts
            attempts += 1
            # check if we should retry
            if isinstance(retry_policy.retry_on, Sequence):
                if not isinstance(exc, tuple(retry_policy.retry_on)):
                    raise
            elif isinstance(retry_policy.retry_on, type) and issubclass(
                retry_policy.retry_on, Exception
            ):
                if not isinstance(exc, retry_policy.retry_on):
                    raise
            elif callable(retry_policy.retry_on):
                if not retry_policy.retry_on(exc):  # type: ignore[call-arg]
                    raise
            else:
                raise TypeError(
                    "retry_on must be an Exception class, a list or tuple of Exception classes, or a callable"
                )
            # check if we should give up
            if attempts >= retry_policy.max_attempts:
                raise
            # sleep before retrying
            interval = min(
                retry_policy.max_interval,
                interval * retry_policy.backoff_factor,
            )
            time.sleep(
                interval + random.uniform(0, 1) if retry_policy.jitter else interval
            )
            # log the retry
            logger.info(
                f"Retrying task {task.name} after {interval:.2f} seconds (attempt {attempts}) after {exc.__class__.__name__} {exc}",
                exc_info=exc,
            )
            # signal subgraphs to resume (if available)
            config = patch_configurable(config, {CONFIG_KEY_RESUMING: True})
            # clear checkpoint_ns seen (for subgraph detection)
            if checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS):
                _SEEN_CHECKPOINT_NS.discard(checkpoint_ns)
        finally:
            # clear checkpoint_ns seen (for subgraph detection)
            if checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS):
                _SEEN_CHECKPOINT_NS.discard(checkpoint_ns)


async def arun_with_retry(
    task: PregelExecutableTask,
    retry_policy: Optional[RetryPolicy],
    stream: bool = False,
    configurable: Optional[dict[str, Any]] = None,
) -> None:
    """Run a task asynchronously with retries."""
    retry_policy = task.retry_policy or retry_policy
    interval = retry_policy.initial_interval if retry_policy else 0
    attempts = 0
    config = task.config
    if configurable is not None:
        config = patch_configurable(config, configurable)
    while True:
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            if stream:
                async for _ in task.proc.astream(task.input, config):
                    pass
                # if successful, end
                break
            else:
                return await task.proc.ainvoke(task.input, config)
        except ParentCommand as exc:
            ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]
            cmd = exc.args[0]
            if cmd.graph == ns:
                # this command is for the current graph, handle it
                for w in task.writers:
                    w.invoke(cmd, config)
                break
            elif cmd.graph == Command.PARENT:
                # this command is for the parent graph, assign it to the parent
                parent_ns = NS_SEP.join(ns.split(NS_SEP)[:-1])
                exc.args = (replace(cmd, graph=parent_ns),)
            # bubble up
            raise
        except GraphBubbleUp:
            # if interrupted, end
            raise
        except Exception as exc:
            if SUPPORTS_EXC_NOTES:
                exc.add_note(f"During task with name '{task.name}' and id '{task.id}'")
            if retry_policy is None:
                raise
            # increment attempts
            attempts += 1
            # check if we should retry
            if isinstance(retry_policy.retry_on, Sequence):
                if not isinstance(exc, tuple(retry_policy.retry_on)):
                    raise
            elif isinstance(retry_policy.retry_on, type) and issubclass(
                retry_policy.retry_on, Exception
            ):
                if not isinstance(exc, retry_policy.retry_on):
                    raise
            elif callable(retry_policy.retry_on):
                if not retry_policy.retry_on(exc):  # type: ignore[call-arg]
                    raise
            else:
                raise TypeError(
                    "retry_on must be an Exception class, a list or tuple of Exception classes, or a callable"
                )
            # check if we should give up
            if attempts >= retry_policy.max_attempts:
                raise
            # sleep before retrying
            interval = min(
                retry_policy.max_interval,
                interval * retry_policy.backoff_factor,
            )
            await asyncio.sleep(
                interval + random.uniform(0, 1) if retry_policy.jitter else interval
            )
            # log the retry
            logger.info(
                f"Retrying task {task.name} after {interval:.2f} seconds (attempt {attempts}) after {exc.__class__.__name__} {exc}",
                exc_info=exc,
            )
            # signal subgraphs to resume (if available)
            config = patch_configurable(config, {CONFIG_KEY_RESUMING: True})
            # clear checkpoint_ns seen (for subgraph detection)
            if checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS):
                _SEEN_CHECKPOINT_NS.discard(checkpoint_ns)
        finally:
            # clear checkpoint_ns seen (for subgraph detection)
            if checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS):
                _SEEN_CHECKPOINT_NS.discard(checkpoint_ns)
