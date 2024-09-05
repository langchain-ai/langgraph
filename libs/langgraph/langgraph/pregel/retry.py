import asyncio
import logging
import random
import time
from typing import Optional

from langchain_core.runnables import Runnable, RunnableConfig

from langgraph.errors import GraphInterrupt
from langgraph.pregel.types import PregelExecutableTask, RetryPolicy

logger = logging.getLogger(__name__)


def run_with_retry(
    run: Runnable,
    config: RunnableConfig,
    task: PregelExecutableTask,
    retry_policy: Optional[RetryPolicy],
) -> None:
    """Run a task with retries."""
    interval = retry_policy.initial_interval if retry_policy else 0
    attempts = 0
    while True:
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            run.invoke(task.input, config)
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
            if callable(retry_policy.retry_on):
                if not retry_policy.retry_on(exc):
                    raise
            elif not isinstance(exc, retry_policy.retry_on):
                raise
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


async def arun_with_retry(
    run: Runnable,
    config: RunnableConfig,
    task: PregelExecutableTask,
    retry_policy: Optional[RetryPolicy],
    stream: bool = False,
) -> None:
    """Run a task asynchronously with retries."""
    interval = retry_policy.initial_interval if retry_policy else 0
    attempts = 0
    while True:
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            if stream:
                async for _ in run.astream(task.input, config):
                    pass
            else:
                await run.ainvoke(task.input, config)
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
            if callable(retry_policy.retry_on):
                if not retry_policy.retry_on(exc):
                    raise
            elif not isinstance(exc, retry_policy.retry_on):
                raise
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
