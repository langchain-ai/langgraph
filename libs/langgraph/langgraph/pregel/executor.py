import concurrent.futures
import time
from contextlib import ExitStack
from contextvars import copy_context
from functools import partial
from types import TracebackType
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    TypeVar,
    cast,
)

from typing_extensions import ParamSpec

from langgraph.errors import GraphBubbleUp
from langgraph.utils.config import RunnableConfig

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
        __next_tick__: bool = False,
        **kwargs: P.kwargs,
    ) -> concurrent.futures.Future[T]: ...


class ContextThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """ThreadPoolExecutor that copies the context to the child thread."""

    def submit(  # type: ignore[override]
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> concurrent.futures.Future[T]:
        """Submit a function to the executor.

        Args:
            func (Callable[..., T]): The function to submit.
            *args (Any): The positional arguments to the function.
            **kwargs (Any): The keyword arguments to the function.

        Returns:
            Future[T]: The future for the function.
        """
        return super().submit(
            cast(Callable[..., T], partial(copy_context().run, func, *args, **kwargs))
        )

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[T]:
        """Map a function to multiple iterables.

        Args:
            fn (Callable[..., T]): The function to map.
            *iterables (Iterable[Any]): The iterables to map over.
            timeout (float | None, optional): The timeout for the map.
                Defaults to None.
            chunksize (int, optional): The chunksize for the map. Defaults to 1.

        Returns:
            Iterator[T]: The iterator for the mapped function.
        """
        contexts = [copy_context() for _ in range(len(iterables[0]))]  # type: ignore[arg-type]

        def _wrapped_fn(*args: Any) -> T:
            return contexts.pop().run(fn, *args)

        return super().map(
            _wrapped_fn,
            *iterables,
            timeout=timeout,
            chunksize=chunksize,
        )


class BackgroundExecutor(ContextManager):
    """A context manager that runs sync tasks in the background.
    Uses a thread pool executor to delegate tasks to separate threads.
    On exit,
    - cancels any (not yet started) tasks with `__cancel_on_exit__=True`
    - waits for all tasks to finish
    - re-raises the first exception from tasks with `__reraise_on_exit__=True`"""

    def __init__(self, config: RunnableConfig) -> None:
        self.stack = ExitStack()
        self.executor = self.stack.enter_context(
            ContextThreadPoolExecutor(max_workers=config.get("max_concurrency"))
        )
        # mapping of Future to (__cancel_on_exit__, __reraise_on_exit__) flags
        self.tasks: dict[concurrent.futures.Future, tuple[bool, bool]] = {}

    def submit(  # type: ignore[valid-type]
        self,
        fn: Callable[P, T],
        *args: P.args,
        __name__: Optional[str] = None,  # currently not used in sync version
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


def next_tick(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """A function that yields control to other threads before running another function."""
    time.sleep(0)
    return fn(*args, **kwargs)
