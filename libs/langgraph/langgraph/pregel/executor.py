import asyncio
import concurrent.futures
import sys
from contextlib import ExitStack
from contextvars import copy_context
from types import TracebackType
from typing import (
    AsyncContextManager,
    Awaitable,
    Callable,
    ContextManager,
    Optional,
    Protocol,
    TypeVar,
)

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from typing_extensions import ParamSpec

from langgraph.errors import GraphInterrupt

P = ParamSpec("P")
T = TypeVar("T")


class Submit(Protocol[P, T]):
    def __call__(
        self,
        fn: Callable[P, T],
        *args: P.args,
        __name__: Optional[str] = None,
        __cancel_on_exit__: bool = False,
        **kwargs: P.kwargs,
    ) -> concurrent.futures.Future[T]:
        ...


class BackgroundExecutor(ContextManager):
    def __init__(self, config: RunnableConfig) -> None:
        self.stack = ExitStack()
        self.executor = self.stack.enter_context(get_executor_for_config(config))
        self.tasks: dict[concurrent.futures.Future, bool] = {}

    def submit(
        self,
        fn: Callable[P, T],
        *args: P.args,
        __name__: Optional[str] = None,  # currently not used in sync version
        __cancel_on_exit__: bool = False,
        **kwargs: P.kwargs,
    ) -> concurrent.futures.Future[T]:
        task = self.executor.submit(fn, *args, **kwargs)
        self.tasks[task] = __cancel_on_exit__
        task.add_done_callback(self.done)
        return task

    def done(self, task: concurrent.futures.Future) -> None:
        try:
            task.result()
        except GraphInterrupt:
            # This exception is an interruption signal, not an error
            # so we don't want to re-raise it on exit
            self.tasks.pop(task)
        except BaseException:
            pass
        else:
            self.tasks.pop(task)

    def __enter__(self) -> "submit":
        return self.submit

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        # cancel all tasks that should be cancelled
        for task, cancel in self.tasks.items():
            if cancel:
                task.cancel()
        # wait for all tasks to finish
        if tasks := {t for t in self.tasks if not t.done()}:
            concurrent.futures.wait(tasks)
        # shutdown the executor
        self.stack.__exit__(exc_type, exc_value, traceback)
        # re-raise the first exception that occurred in a task
        if exc_type is None:
            # if there's already an exception being raised, don't raise another one
            for task in self.tasks:
                try:
                    task.result()
                except concurrent.futures.CancelledError:
                    pass


class AsyncBackgroundExecutor(AsyncContextManager):
    def __init__(self) -> None:
        self.context_not_supported = sys.version_info < (3, 11)
        self.tasks: dict[asyncio.Task, bool] = {}
        self.sentinel = object()

    def submit(
        self,
        fn: Callable[P, Awaitable[T]],
        *args: P.args,
        __name__: Optional[str] = None,
        __cancel_on_exit__: bool = False,
        **kwargs: P.kwargs,
    ) -> asyncio.Task[T]:
        coro = fn(*args, **kwargs)
        if self.context_not_supported:
            task = asyncio.create_task(coro, name=__name__)
        else:
            task = asyncio.create_task(coro, name=__name__, context=copy_context())
        self.tasks[task] = __cancel_on_exit__
        task.add_done_callback(self.done)
        return task

    def done(self, task: asyncio.Task) -> None:
        try:
            if exc := task.exception():
                # This exception is an interruption signal, not an error
                # so we don't want to re-raise it on exit
                if isinstance(exc, GraphInterrupt):
                    self.tasks.pop(task)
            else:
                self.tasks.pop(task)
        except asyncio.CancelledError:
            self.tasks.pop(task)

    async def __aenter__(self) -> Submit:
        return self.submit

    async def exit(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        # cancel all tasks that should be cancelled
        for task, cancel in self.tasks.items():
            if cancel:
                task.cancel(self.sentinel)
        # wait for all tasks to finish
        if self.tasks:
            await asyncio.wait(self.tasks)
        # if there's already an exception being raised, don't raise another one
        if exc_type is None:
            # re-raise the first exception that occurred in a task
            for task in self.tasks:
                try:
                    if exc := task.exception():
                        raise exc
                except asyncio.CancelledError:
                    pass

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        # we cannot use `await` outside of asyncio.shield, as this code can run
        # after owning task is cancelled, so pulling async logic to separate method

        # wait for all background tasks to finish, shielded from cancellation
        await asyncio.shield(self.exit(exc_type, exc_value, traceback))
