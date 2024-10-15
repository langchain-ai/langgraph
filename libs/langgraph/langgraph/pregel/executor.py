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
    Coroutine,
    Optional,
    Protocol,
    TypeVar,
    cast,
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
        __reraise_on_exit__: bool = True,
        **kwargs: P.kwargs,
    ) -> concurrent.futures.Future[T]: ...


class BackgroundExecutor(ContextManager):
    """A context manager that runs sync tasks in the background.
    Uses a thread pool executor to delegate tasks to separate threads.
    On exit,
    - cancels any (not yet started) tasks with `__cancel_on_exit__=True`
    - waits for all tasks to finish
    - re-raises the first exception from tasks with `__reraise_on_exit__=True`"""

    def __init__(self, config: RunnableConfig) -> None:
        self.stack = ExitStack()
        self.executor = self.stack.enter_context(get_executor_for_config(config))
        self.tasks: dict[concurrent.futures.Future, tuple[bool, bool]] = {}

    def submit(  # type: ignore[valid-type]
        self,
        fn: Callable[P, T],
        *args: P.args,
        __name__: Optional[str] = None,  # currently not used in sync version
        __cancel_on_exit__: bool = False,  # for sync, can cancel only if not started
        __reraise_on_exit__: bool = True,
        **kwargs: P.kwargs,
    ) -> concurrent.futures.Future[T]:
        task = self.executor.submit(fn, *args, **kwargs)
        self.tasks[task] = (__cancel_on_exit__, __reraise_on_exit__)
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

    def __enter__(self) -> Submit:
        return self.submit

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
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
        # re-raise the first exception that occurred in a task
        if exc_type is None:
            # if there's already an exception being raised, don't raise another one
            for task, (_, reraise) in tasks.items():
                if not reraise:
                    continue
                try:
                    task.result()
                except concurrent.futures.CancelledError:
                    pass


class AsyncBackgroundExecutor(AsyncContextManager):
    """A context manager that runs async tasks in the background.
    Uses the current event loop to delegate tasks to asyncio tasks.
    On exit,
    - cancels any tasks with `__cancel_on_exit__=True`
    - waits for all tasks to finish
    - re-raises the first exception from tasks with `__reraise_on_exit__=True`
      ignoring CancelledError"""

    def __init__(self) -> None:
        self.context_not_supported = sys.version_info < (3, 11)
        self.tasks: dict[asyncio.Task, tuple[bool, bool]] = {}
        self.sentinel = object()
        self.loop = asyncio.get_running_loop()

    def submit(  # type: ignore[valid-type]
        self,
        fn: Callable[P, Awaitable[T]],
        *args: P.args,
        __name__: Optional[str] = None,
        __cancel_on_exit__: bool = False,
        __reraise_on_exit__: bool = True,
        **kwargs: P.kwargs,
    ) -> asyncio.Task[T]:
        coro = cast(Coroutine[None, None, T], fn(*args, **kwargs))
        if self.context_not_supported:
            task = self.loop.create_task(coro, name=__name__)
        else:
            task = self.loop.create_task(coro, name=__name__, context=copy_context())
        self.tasks[task] = (__cancel_on_exit__, __reraise_on_exit__)
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

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
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
