import asyncio

import pytest

from langgraph.pregel._executor import AsyncBackgroundExecutor

pytestmark = pytest.mark.anyio


# Regression for #6950: cancellation can arrive after the context body has exited.
@pytest.mark.parametrize("child_count", [1, 2])
@pytest.mark.parametrize("cleanup_error", [False, True])
async def test_async_executor_waits_for_cleanup_before_propagating_cancellation(
    child_count: int, cleanup_error: bool
) -> None:
    started = [asyncio.Event() for _ in range(child_count)]
    cleanup_started = [asyncio.Event() for _ in range(child_count)]
    cleanup_finished = [asyncio.Event() for _ in range(child_count)]
    release_cleanup = asyncio.Event()
    children: list[asyncio.Future[None]] = []

    async def child(index: int) -> None:
        started[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started[index].set()
            await release_cleanup.wait()
            cleanup_finished[index].set()
            if cleanup_error:
                raise ValueError("child cleanup failed")

    async def run() -> None:
        try:
            async with AsyncBackgroundExecutor({}) as submit:
                children.extend(
                    submit(child, index, __cancel_on_exit__=True)
                    for index in range(child_count)
                )
                await asyncio.gather(*(event.wait() for event in started))
        except asyncio.CancelledError as exc:
            # Python 3.10 does not propagate cancellation messages to task awaiters.
            assert exc.args == ("stop outer",)
            raise

    outer = asyncio.create_task(run())
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in cleanup_started)), timeout=5
        )
        outer.cancel("stop outer")
        # Let the outer task handle cancellation while child cleanup is still blocked.
        await asyncio.sleep(0)
        assert not outer.done()
        assert all(not task.done() for task in children)

        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(outer, timeout=5)

        assert outer.cancelled()
        assert all(event.is_set() for event in cleanup_finished)
        assert all(task.done() for task in children)
    finally:
        release_cleanup.set()
        if not outer.done():
            outer.cancel()
        await asyncio.gather(outer, *children, return_exceptions=True)


async def test_async_executor_second_cancellation_aborts_blocked_cleanup() -> None:
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    release_cleanup = asyncio.Event()
    children: list[asyncio.Future[None]] = []

    async def child() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_finished.set()

    async def run() -> None:
        try:
            async with AsyncBackgroundExecutor({}) as submit:
                children.append(submit(child, __cancel_on_exit__=True))
                await started.wait()
        except asyncio.CancelledError as exc:
            assert exc.args == ("force stop",)
            raise

    outer = asyncio.create_task(run())
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        outer.cancel("stop outer")
        await asyncio.sleep(0)
        assert not outer.done()

        outer.cancel("force stop")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(outer, timeout=5)

        assert outer.cancelled()
        assert not children[0].done()
        assert not cleanup_finished.is_set()
    finally:
        release_cleanup.set()
        if not outer.done():
            outer.cancel()
        await asyncio.gather(outer, *children, return_exceptions=True)
