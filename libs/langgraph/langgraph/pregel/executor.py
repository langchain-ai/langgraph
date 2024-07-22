import asyncio
import concurrent.futures
import sys
from contextlib import contextmanager
from contextvars import copy_context
from types import TracebackType
from typing import (
    AsyncContextManager,
    Awaitable,
    Callable,
    Iterator,
    Optional,
    Protocol,
    TypeVar,
)

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from typing_extensions import ParamSpec

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


@contextmanager
def BackgroundExecutor(config: RunnableConfig) -> Iterator[Submit]:
    tasks: dict[concurrent.futures.Future, bool] = {}
    with get_executor_for_config(config) as executor:

        def done(task: concurrent.futures.Future) -> None:
            try:
                task.result()
            except BaseException:
                pass
            else:
                tasks.pop(task)

        def submit(
            fn: Callable[P, T],
            *args: P.args,
            __name__: Optional[str] = None,  # currently not used in sync version
            __cancel_on_exit__: bool = False,
            **kwargs: P.kwargs,
        ) -> concurrent.futures.Future:
            task = executor.submit(fn, *args, **kwargs)
            tasks[task] = __cancel_on_exit__
            task.add_done_callback(done)
            return task

        try:
            yield submit
        finally:
            for task, cancel in tasks.items():
                if cancel:
                    task.cancel()
            # executor waits for all tasks to finish on exit
    for task in tasks:
        # the first task to have raised an exception will be re-raised here
        task.result()


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
            task.result()
        except BaseException:
            pass
        else:
            self.tasks.pop(task)

    async def __aenter__(self) -> "submit":
        return self.submit

    async def exit(self) -> None:
        fut = asyncio.gather(*self.tasks, return_exceptions=True)
        try:
            rtns = await asyncio.shield(fut)
        finally:
            del self.tasks
        for rtn in rtns:
            # if this is ever changed to BaseException, need to ignore CancelledError
            if isinstance(rtn, Exception):
                raise rtn

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        for task, cancel in self.tasks.items():
            if cancel:
                task.cancel(self.sentinel)
        # wait for all background tasks to finish, shielded from cancellation
        await asyncio.shield(self.exit())
