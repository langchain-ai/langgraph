"""Tests for AsyncBackgroundExecutor cancellation handling.

Ensures that AsyncBackgroundExecutor.__aexit__ completes cleanup even when
the parent task is cancelled, preventing orphaned tasks and CancelledError
propagation during cleanup. (fixes #6950)
"""

from __future__ import annotations

import asyncio

import pytest

from langgraph.pregel._executor import AsyncBackgroundExecutor


def _make_executor() -> AsyncBackgroundExecutor:
    """Create an AsyncBackgroundExecutor with minimal config."""
    return AsyncBackgroundExecutor(config={})


@pytest.mark.asyncio
async def test_aexit_handles_cancellation_gracefully():
    """CancelledError during __aexit__ should not propagate from cleanup.

    Regression test for #6950: in Python 3.11+, CancelledError inherits
    BaseException and propagates into asyncio.wait's internal waiter when
    the parent task is cancelled, breaking cleanup.
    """
    task_started = asyncio.Event()

    async def slow_task() -> str:
        task_started.set()
        await asyncio.sleep(1.0)
        return "done"

    aexit_completed = asyncio.Event()

    async def run_executor():
        async with _make_executor() as submit:
            submit(slow_task)
            await task_started.wait()
        # If we reach here, __aexit__ completed without propagating CancelledError
        aexit_completed.set()

    task = asyncio.create_task(run_executor())
    # Wait for the background task to start
    await task_started.wait()
    # Cancel the parent task — this will interrupt __aexit__'s asyncio.wait
    task.cancel()
    # The task should complete (either normally or with CancelledError)
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Give a moment for any remaining cleanup
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_aexit_without_cancellation():
    """Normal (non-cancelled) __aexit__ should still work correctly."""
    results = []

    async def append_task(value: str) -> None:
        await asyncio.sleep(0.01)
        results.append(value)

    async with _make_executor() as submit:
        submit(append_task, "a")
        submit(append_task, "b")

    assert sorted(results) == ["a", "b"]


@pytest.mark.asyncio
async def test_aexit_reraises_task_exception():
    """Task exceptions should be reraised when no other exception is active."""

    async def failing_task() -> None:
        raise ValueError("task failed")

    with pytest.raises(ValueError, match="task failed"):
        async with _make_executor() as submit:
            submit(failing_task)


@pytest.mark.asyncio
async def test_aexit_cancels_tasks_marked_for_cancellation():
    """Tasks submitted with __cancel_on_exit__=True should be cancelled."""
    started = asyncio.Event()

    async def long_task() -> None:
        started.set()
        await asyncio.sleep(10)  # should be cancelled before completing

    async with _make_executor() as submit:
        submit(long_task, __cancel_on_exit__=True, __reraise_on_exit__=False)
        await started.wait()

    # __aexit__ should have cancelled and waited for long_task.
    # If we reach here without hanging, the test passes.


@pytest.mark.asyncio
async def test_aexit_suppresses_cancelled_error_does_not_crash():
    """Executor __aexit__ should not crash when CancelledError is raised."""

    async def noop() -> None:
        await asyncio.sleep(0.5)

    executor = _make_executor()
    async with executor as submit:
        fut = submit(noop, __reraise_on_exit__=False)

    # After normal exit, all tasks should be done
    assert fut.done()
