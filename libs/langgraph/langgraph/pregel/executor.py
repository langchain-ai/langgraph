from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections.abc import Awaitable, Coroutine
from contextlib import AbstractAsyncContextManager, AbstractContextManager, ExitStack
from contextvars import copy_context
from types import TracebackType
from typing import (
    Callable,
    Protocol,
    TypeVar,
    cast,
)

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from typing_extensions import ParamSpec

from langgraph.errors import GraphBubbleUp
from langgraph.utils.future import CONTEXT_NOT_SUPPORTED, run_coroutine_threadsafe

P = ParamSpec("P")
T = TypeVar("T")


class Submit(Protocol[P, T]):
    def __call__(  # type: ignore[valid-type]
        self,
        fn: Callable[P, T],
        *args: P.args,
        __name__: str | None = None,
        __cancel_on_exit__: bool = False,
        __reraise_on_exit__: bool = True,
        __next_tick__: bool = False,
        **kwargs: P.kwargs,
    ) -> concurrent.futures.Future[T]: ...


class BackgroundExecutor(AbstractContextManager):
    """A context manager that runs sync tasks in the background.
    Uses a thread pool executor to delegate tasks to separate threads.
    On exit,
    - cancels any (not yet started) tasks with `__cancel_on_exit__=True`
    - waits for all tasks to finish
    - re-raises the first exception from tasks with `__reraise_on_exit__=True`"""

    def __init__(self, config: RunnableConfig) -> None:
        self.stack = ExitStack()
        self.executor = self.stack.enter_context(get_executor_for_config(config))
        # mapping of Future to (__cancel_on_exit__, __reraise_on_exit__) flags
        self.tasks: dict[concurrent.futures.Future, tuple[bool, bool]] = {}

    def submit(  # type: ignore[valid-type]
        self,
        fn: Callable[P, T],
        *args: P.args,
        __name__: str | None = None,  # currently not used in sync version
        __cancel_on_exit__: bool = False,  # for sync, can cancel only if not started
        __reraise_on_exit__: bool = True,
        __next_tick__: bool = False,
        **kwargs: P.kwargs,
    ) -> concurrent.futures.Future[T]:
        ctx = copy_context()
        if __next_tick__:
            task = cast(
                concurrent.futures.Future[T],
                self.executor.submit(next_tick, ctx.run, fn, *args, **kwargs),  # type: ignore[arg-type]
            )
        else:
            task = self.executor.submit(ctx.run, fn, *args, **kwargs)
        self.tasks[task] = (__cancel_on_exit__, __reraise_on_exit__)
        # add a callback to remove the task from the tasks dict when it's done
        task.add_done_callback(self.done)
        return task

    def done(self, task: concurrent.futures.Future) -> None:
        """Remove the task from the tasks dict when it's done."""
        try:
            task.result()
        except GraphBubbleUp:
            # This exception is an interruption signal, not an error
            # so we don't want to re-raise it on exit
            self.tasks.pop(task)
        except BaseException:
            pass
        else:
            self.tasks.pop(task)

    def __enter__(self) -> Submit:
        return self.submit

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        # copy the tasks as done() callback may modify the dict
        tasks = self.tasks.copy()
        # cancel all tasks that should be cancelled
        for task, (cancel, _) in tasks.items():
            if cancel:
                task.cancel()
        # wait for all tasks to finish
        if pending := {t for t in tasks if not t.done()}:
            concurrent.futures.wait(pending)
        # shutdown the executor
        self.stack.__exit__(exc_type, exc_value, traceback)
        # if there's already an exception being raised, don't raise another one
        if exc_type is None:
            # re-raise the first exception that occurred in a task
            for task, (_, reraise) in tasks.items():
                if not reraise:
                    continue
                try:
                    task.result()
                except concurrent.futures.CancelledError:
                    pass


class AsyncBackgroundExecutor(AbstractAsyncContextManager):
    """A context manager that runs async tasks in the background.
    Uses the current event loop to delegate tasks to asyncio tasks.
    On exit,
    - cancels any tasks with `__cancel_on_exit__=True`
    - waits for all tasks to finish
    - re-raises the first exception from tasks with `__reraise_on_exit__=True`
      ignoring CancelledError"""

    def __init__(self, config: RunnableConfig) -> None:
        self.tasks: dict[asyncio.Future, tuple[bool, bool]] = {}
        self.sentinel = object()
        self.loop = asyncio.get_running_loop()
        if max_concurrency := config.get("max_concurrency"):
            self.semaphore: asyncio.Semaphore | None = asyncio.Semaphore(
                max_concurrency
            )
        else:
            self.semaphore = None

    def submit(  # type: ignore[valid-type]
        self,
        fn: Callable[P, Awaitable[T]],
        *args: P.args,
        __name__: str | None = None,
        __cancel_on_exit__: bool = False,
        __reraise_on_exit__: bool = True,
        __next_tick__: bool = False,  # noop in async (always True)
        **kwargs: P.kwargs,
    ) -> asyncio.Future[T]:
        coro = cast(Coroutine[None, None, T], fn(*args, **kwargs))
        if self.semaphore:
            coro = gated(self.semaphore, coro)
        if CONTEXT_NOT_SUPPORTED:
            task = run_coroutine_threadsafe(
                coro, self.loop, name=__name__, lazy=__next_tick__
            )
        else:
            task = run_coroutine_threadsafe(
                coro,
                self.loop,
                name=__name__,
                context=copy_context(),
                lazy=__next_tick__,
            )
        self.tasks[task] = (__cancel_on_exit__, __reraise_on_exit__)
        task.add_done_callback(self.done)
        return task

    def done(self, task: asyncio.Future) -> None:
        try:
            if exc := task.exception():
                # This exception is an interruption signal, not an error
                # so we don't want to re-raise it on exit
                if isinstance(exc, GraphBubbleUp):
                    self.tasks.pop(task)
            else:
                self.tasks.pop(task)
        except asyncio.CancelledError:
            self.tasks.pop(task)

    async def __aenter__(self) -> Submit:
        return self.submit

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # copy the tasks as done() callback may modify the dict
        tasks = self.tasks.copy()
        # cancel all tasks that should be cancelled
        for task, (cancel, _) in tasks.items():
            if cancel:
                task.cancel(self.sentinel)
        # wait for all tasks to finish
        if tasks:
            await asyncio.wait(tasks)
        # if there's already an exception being raised, don't raise another one
        if exc_type is None:
            # re-raise the first exception that occurred in a task
            for task, (_, reraise) in tasks.items():
                if not reraise:
                    continue
                try:
                    if exc := task.exception():
                        raise exc
                except asyncio.CancelledError:
                    pass


async def gated(semaphore: asyncio.Semaphore, coro: Coroutine[None, None, T]) -> T:
    """A coroutine that waits for a semaphore before running another coroutine."""
    async with semaphore:
        return await coro


def next_tick(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """A function that yields control to other threads before running another function."""
    time.sleep(0)
    return fn(*args, **kwargs)
