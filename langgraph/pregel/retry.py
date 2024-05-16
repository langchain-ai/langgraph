import asyncio
import logging
import random
import time
from typing import Callable, NamedTuple, Union

import httpx
import requests

from langgraph.pregel.types import PregelExecutableTask

logger = logging.getLogger(__name__)


def default_retry_on(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            ValueError,
            TypeError,
            ArithmeticError,
            ImportError,
            LookupError,
            NameError,
            SyntaxError,
            RuntimeError,
        ),
    ):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, requests.HTTPError):
        return 500 <= exc.response.status_code < 600 if exc.response else True
    return True


class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5
    """Amount of time that must elapse before the first retry occurs. In seconds."""
    backoff_factor: float = 2.0
    """Multiplier by which the interval increases after each retry."""
    max_interval: float = 128.0
    """Maximum amount of time that may elapse between retries. In seconds."""
    max_attempts: int = 10
    """Maximum number of attempts to make before giving up, including the first."""
    jitter: bool = True
    """Whether to add random jitter to the interval between retries."""
    retry_on: Union[
        tuple[Exception, ...], Callable[[Exception], bool]
    ] = default_retry_on
    """List of exceptions that should trigger a retry, or a callable that returns True for exceptions that should trigger a retry."""


def run_with_retry(
    task: PregelExecutableTask,
    retry_policy: RetryPolicy,
) -> None:
    """Run a task with retries."""
    interval = retry_policy.initial_interval
    attempts = 0
    while True:
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            task.proc.invoke(task.input, task.config)
            # if successful, end
            break
        except Exception as exc:
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
                f"Retrying task {task.name} after {interval:.2f} seconds (attempt {attempts})"
            )


async def arun_with_retry(
    task: PregelExecutableTask,
    retry_policy: RetryPolicy,
    stream: bool = False,
) -> None:
    """Run a task asynchronously with retries."""
    interval = retry_policy.initial_interval
    attempts = 0
    while True:
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            if stream:
                async for _ in task.proc.astream(task.input, task.config):
                    pass
            else:
                await task.proc.ainvoke(task.input, task.config)
            # if successful, end
            break
        except Exception as exc:
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
                f"Retrying task {task.name} after {interval:.2f} seconds (attempt {attempts})"
            )
