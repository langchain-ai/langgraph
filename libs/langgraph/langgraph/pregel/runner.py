import asyncio
import concurrent.futures
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Optional,
    Sequence,
    Type,
    Union,
)

from langgraph.constants import ERROR, INTERRUPT
from langgraph.errors import GraphInterrupt
from langgraph.pregel.executor import Submit
from langgraph.pregel.retry import arun_with_retry, run_with_retry
from langgraph.pregel.types import PregelExecutableTask, RetryPolicy


class PregelRunner:
    def __init__(
        self,
        *,
        submit: Submit,
        put_writes: Callable[[str, Sequence[tuple[str, Any]]], None],
        use_astream: bool = False,
    ) -> None:
        self.submit = submit
        self.put_writes = put_writes
        self.use_astream = use_astream

    def tick(
        self,
        tasks: list[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> Iterator[None]:
        # give control back to the caller
        yield
        # execute tasks, and wait for one to fail or all to finish.
        # each task is independent from all other concurrent tasks
        # yield updates/debug output as each task finishes
        futures = {
            self.submit(
                run_with_retry,
                task,
                retry_policy,
                __reraise_on_exit__=reraise,
            ): task
            for task in tasks
            if not task.writes
        }
        all_futures = futures.copy()
        end_time = timeout + time.monotonic() if timeout else None
        while futures:
            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
                timeout=(max(0, end_time - time.monotonic()) if end_time else None),
            )
            if not done:
                break  # timed out
            for fut in done:
                task = futures.pop(fut)
                if exc := _exception(fut):
                    if isinstance(exc, GraphInterrupt):
                        # save interrupt to checkpointer
                        self.put_writes(task.id, [(INTERRUPT, i) for i in exc.args[0]])
                    else:
                        # save error to checkpointer
                        self.put_writes(task.id, [(ERROR, exc)])

                else:
                    # save task writes to checkpointer
                    self.put_writes(task.id, task.writes)
            else:
                # remove references to loop vars
                del fut, task
            # maybe stop other tasks
            if _should_stop_others(done):
                break
            # give control back to the caller
            yield
        # panic on failure or timeout
        _panic_or_proceed(all_futures, panic=reraise)

    async def atick(
        self,
        tasks: list[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> AsyncIterator[None]:
        loop = asyncio.get_event_loop()
        # give control back to the caller
        yield
        # execute tasks, and wait for one to fail or all to finish.
        # each task is independent from all other concurrent tasks
        # yield updates/debug output as each task finishes
        futures = {
            self.submit(
                arun_with_retry,
                task,
                retry_policy,
                stream=self.use_astream,
                __name__=task.name,
                __cancel_on_exit__=True,
                __reraise_on_exit__=reraise,
            ): task
            for task in tasks
            if not task.writes
        }
        all_futures = futures.copy()
        end_time = timeout + loop.time() if timeout else None
        while futures:
            done, _ = await asyncio.wait(
                futures,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=(max(0, end_time - loop.time()) if end_time else None),
            )
            if not done:
                break  # timed out
            for fut in done:
                task = futures.pop(fut)
                if exc := _exception(fut):
                    if isinstance(exc, GraphInterrupt):
                        # save interrupt to checkpointer
                        self.put_writes(task.id, [(INTERRUPT, i) for i in exc.args[0]])
                    else:
                        # save error to checkpointer
                        self.put_writes(task.id, [(ERROR, exc)])
                else:
                    # save task writes to checkpointer
                    self.put_writes(task.id, task.writes)
            else:
                # remove references to loop vars
                del fut, task
            # maybe stop other tasks
            if _should_stop_others(done):
                break
            # give control back to the caller
            yield
        # panic on failure or timeout
        _panic_or_proceed(
            all_futures, timeout_exc_cls=asyncio.TimeoutError, panic=reraise
        )


def _should_stop_others(
    done: Union[set[concurrent.futures.Future[Any]], set[asyncio.Task[Any]]],
) -> bool:
    for fut in done:
        if fut.cancelled():
            return True
        if exc := fut.exception():
            return not isinstance(exc, GraphInterrupt)
    else:
        return False


def _exception(
    fut: Union[concurrent.futures.Future[Any], asyncio.Task[Any]],
) -> Optional[BaseException]:
    if fut.cancelled():
        if isinstance(fut, asyncio.Task):
            return asyncio.CancelledError()
        else:
            return concurrent.futures.CancelledError()
    else:
        return fut.exception()


def _panic_or_proceed(
    futs: Union[set[concurrent.futures.Future[Any]], set[asyncio.Task[Any]]],
    *,
    timeout_exc_cls: Type[Exception] = TimeoutError,
    panic: bool = True,
) -> None:
    done: set[Union[concurrent.futures.Future[Any], asyncio.Task[Any]]] = set()
    inflight: set[Union[concurrent.futures.Future[Any], asyncio.Task[Any]]] = set()
    for fut in futs:
        if fut.done():
            done.add(fut)
        else:
            inflight.add(fut)
    while done:
        # if any task failed
        if exc := _exception(done.pop()):
            # cancel all pending tasks
            while inflight:
                inflight.pop().cancel()
            # raise the exception
            if panic:
                raise exc
            else:
                return
    if inflight:
        # if we got here means we timed out
        while inflight:
            # cancel all pending tasks
            inflight.pop().cancel()
        # raise timeout error
        raise timeout_exc_cls("Timed out")
