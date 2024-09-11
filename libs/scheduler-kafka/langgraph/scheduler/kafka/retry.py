import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional

from typing_extensions import ParamSpec

from langgraph.pregel.types import RetryPolicy

logger = logging.getLogger(__name__)
P = ParamSpec("P")


async def aretry(
    retry_policy: Optional[RetryPolicy],
    func: Callable[P, Awaitable[None]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    """Run a task asynchronously with retries."""
    interval = retry_policy.initial_interval if retry_policy else 0
    attempts = 0
    while True:
        try:
            await func(*args, **kwargs)
            # if successful, end
            break
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
                f"Retrying function {func} with {args} after {interval:.2f} seconds (attempt {attempts}) after {exc.__class__.__name__} {exc}",
                exc_info=exc,
            )
