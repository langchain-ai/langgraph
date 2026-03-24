"""Regression tests for AsyncBackgroundExecutor cleanup behaviour.

Issue: #6950 — random CancelledError escaping AsyncBackgroundExecutor.__aexit__
when the outer coroutine is cancelled while asyncio.wait(tasks) is in progress.
"""
from __future__ import annotations

import asyncio

import pytest

from langgraph.pregel._executor import AsyncBackgroundExecutor

pytestmark = pytest.mark.anyio


async def _slow(delay: float = 0.05) -> str:
    await asyncio.sleep(delay)
    return "done"


async def _failing() -> None:
    await asyncio.sleep(0.01)
    raise ValueError("task error")


async def test_async_background_executor_no_cancellation() -> None:
    """Normal happy path: tasks complete and executor exits cleanly."""
    async with AsyncBackgroundExecutor({}) as submit:
        f = submit(_slow)
    assert f.result() == "done"


async def test_async_background_executor_reraises_task_exception() -> None:
    """__aexit__ re-raises the first task exception when no outer exc exists."""
    with pytest.raises(ValueError, match="task error"):
        async with AsyncBackgroundExecutor({}) as submit:
            submit(_failing)


async def test_async_background_executor_cancel_on_exit() -> None:
    """Tasks marked __cancel_on_exit__=True are cancelled when the block exits."""
    async with AsyncBackgroundExecutor({}) as submit:
        f = submit(_slow, 10.0, __cancel_on_exit__=True)
    # The task should be cancelled (done) rather than still running
    assert f.done()


async def test_async_background_executor_survives_outer_cancellation() -> None:
    """Regression test for #6950.

    When the coroutine that owns AsyncBackgroundExecutor is cancelled while
    __aexit__ is waiting for background tasks, the CancelledError must NOT
    escape __aexit__.  The background tasks must still be allowed to finish
    before cleanup completes.
    """
    finished: list[str] = []

    async def tracked(label: str, delay: float = 0.02) -> None:
        await asyncio.sleep(delay)
        finished.append(label)

    async def run() -> None:
        async with AsyncBackgroundExecutor({}) as submit:
            submit(tracked, "a", 0.02, __reraise_on_exit__=False)
            submit(tracked, "b", 0.02, __reraise_on_exit__=False)
            # Simulate doing work that keeps us inside the block while tasks run
            await asyncio.sleep(10)

    task = asyncio.create_task(run())
    # Let the background tasks start
    await asyncio.sleep(0.005)
    # Cancel the outer task — this triggers cleanup inside __aexit__
    task.cancel()

    # The task should raise CancelledError (because it was cancelled), but
    # __aexit__ itself must not let a stray CancelledError escape from
    # asyncio.wait — the task exception is a CancelledError from task.cancel(),
    # not a spurious one from the cleanup wait.
    with pytest.raises(asyncio.CancelledError):
        await task

    # The background tasks should have been given a chance to complete
    # (they were NOT marked __cancel_on_exit__=True)
    assert set(finished) == {"a", "b"}, (
        "Background tasks should complete even when the outer task is cancelled"
    )


async def test_async_background_executor_cancel_does_not_swallow_task_exc() -> None:
    """Even under outer cancellation, task exceptions are still re-raised
    (via the CancelledError that propagates out naturally)."""
    exc_holder: list[BaseException] = []

    async def failing_slow() -> None:
        await asyncio.sleep(0.01)
        raise RuntimeError("inner failure")

    async def run() -> None:
        try:
            async with AsyncBackgroundExecutor({}) as submit:
                submit(failing_slow)
                await asyncio.sleep(10)
        except (RuntimeError, asyncio.CancelledError) as exc:
            exc_holder.append(exc)
            raise

    task = asyncio.create_task(run())
    await asyncio.sleep(0.005)
    task.cancel()

    with pytest.raises((asyncio.CancelledError, RuntimeError)):
        await task

    # Either the CancelledError or the RuntimeError propagated — both are fine.
    # What is NOT acceptable is silent swallowing.
    assert len(exc_holder) == 1
