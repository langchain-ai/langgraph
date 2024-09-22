import asyncio
import logging
import random
import time
from typing import Optional, Sequence

from langgraph.constants import CONFIG_KEY_RESUMING
from langgraph.errors import _SEEN_CHECKPOINT_NS, GraphInterrupt
from langgraph.types import PregelExecutableTask, RetryPolicy
from langgraph.utils.config import patch_configurable

logger = logging.getLogger(__name__)


def run_with_retry(
    task: PregelExecutableTask,
    retry_policy: Optional[RetryPolicy],
) -> None:
    """Run a task with retries."""
    retry_policy = task.retry_policy or retry_policy
    interval = retry_policy.initial_interval if retry_policy else 0
    attempts = 0
    config = task.config
    while True:
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            task.proc.invoke(task.input, config)
            # if successful, end
            break
        except GraphInterrupt:
            # if interrupted, end
            raise
        except Exception as exc:
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
            if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
                _SEEN_CHECKPOINT_NS.discard(checkpoint_ns)
        finally:
            # clear checkpoint_ns seen (for subgraph detection)
            if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
                _SEEN_CHECKPOINT_NS.discard(checkpoint_ns)


async def arun_with_retry(
    task: PregelExecutableTask,
    retry_policy: Optional[RetryPolicy],
    stream: bool = False,
) -> None:
    """Run a task asynchronously with retries."""
    retry_policy = task.retry_policy or retry_policy
    interval = retry_policy.initial_interval if retry_policy else 0
    attempts = 0
    config = task.config
    while True:
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            if stream:
                async for _ in task.proc.astream(task.input, config):
                    pass
            else:
                await task.proc.ainvoke(task.input, config)
            # if successful, end
            break
        except GraphInterrupt:
            # if interrupted, end
            raise
        except Exception as exc:
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
            if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
                _SEEN_CHECKPOINT_NS.discard(checkpoint_ns)
        finally:
            # clear checkpoint_ns seen (for subgraph detection)
            if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
                _SEEN_CHECKPOINT_NS.discard(checkpoint_ns)
