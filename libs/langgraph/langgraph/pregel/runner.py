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

from langgraph.constants import ERROR, INTERRUPT, NO_WRITES
from langgraph.errors import GraphDelegate, GraphInterrupt
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
        tasks: Sequence[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
        get_waiter: Optional[Callable[[], concurrent.futures.Future[None]]] = None,
    ) -> Iterator[None]:
        tasks = tuple(tasks)
        # give control back to the caller
        yield
        # fast path if single task with no timeout
        if len(tasks) == 1 and timeout is None:
            t = tasks[0]
            try:
                run_with_retry(t, retry_policy)
                self.commit(t, None)
            except Exception as exc:
                self.commit(t, exc)
                if reraise:
                    raise
            return
        # add waiter task if requested
        if get_waiter is not None:
            futures: dict[concurrent.futures.Future, Optional[PregelExecutableTask]] = {
                get_waiter(): None
            }
        else:
            futures = {}
        # execute tasks, and wait for one to fail or all to finish.
        # each task is independent from all other concurrent tasks
        # yield updates/debug output as each task finishes
        for t in tasks:
            if not t.writes:
                futures[
                    self.submit(
                        run_with_retry,
                        t,
                        retry_policy,
                        __reraise_on_exit__=reraise,
                    )
                ] = t
        all_futures = futures.copy()
        end_time = timeout + time.monotonic() if timeout else None
        while len(futures) > (1 if get_waiter is not None else 0):
            done, inflight = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
                timeout=(max(0, end_time - time.monotonic()) if end_time else None),
            )
            if not done:
                break  # timed out
            for fut in done:
                task = futures.pop(fut)
                if task is None:
                    # waiter task finished, schedule another
                    if inflight and get_waiter is not None:
                        futures[get_waiter()] = None
                else:
                    # task finished, commit writes
                    self.commit(task, _exception(fut))
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
        tasks: Sequence[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
        get_waiter: Optional[Callable[[], asyncio.Future[None]]] = None,
    ) -> AsyncIterator[None]:
        loop = asyncio.get_event_loop()
        tasks = tuple(tasks)
        # give control back to the caller
        yield
        # fast path if single task with no waiter and no timeout
        if len(tasks) == 1 and get_waiter is None and timeout is None:
            t = tasks[0]
            try:
                await arun_with_retry(t, retry_policy, stream=self.use_astream)
                self.commit(t, None)
            except Exception as exc:
                self.commit(t, exc)
                if reraise:
                    raise
            return
        # add waiter task if requested
        if get_waiter is not None:
            futures: dict[asyncio.Future, Optional[PregelExecutableTask]] = {
                get_waiter(): None
            }
        else:
            futures = {}
        # execute tasks, and wait for one to fail or all to finish.
        # each task is independent from all other concurrent tasks
        # yield updates/debug output as each task finishes
        for t in tasks:
            if not t.writes:
                futures[
                    self.submit(
                        arun_with_retry,
                        t,
                        retry_policy,
                        stream=self.use_astream,
                        __name__=t.name,
                        __cancel_on_exit__=True,
                        __reraise_on_exit__=reraise,
                    )
                ] = t
        all_futures = futures.copy()
        end_time = timeout + loop.time() if timeout else None
        while len(futures) > (1 if get_waiter is not None else 0):
            done, inflight = await asyncio.wait(
                futures,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=(max(0, end_time - loop.time()) if end_time else None),
            )
            if not done:
                break  # timed out
            for fut in done:
                task = futures.pop(fut)
                if task is None:
                    # waiter task finished, schedule another
                    if inflight and get_waiter is not None:
                        futures[get_waiter()] = None
                else:
                    # task finished, commit writes
                    self.commit(task, _exception(fut))
            else:
                # remove references to loop vars
                del fut, task
            # maybe stop other tasks
            if _should_stop_others(done):
                break
            # give control back to the caller
            yield
        # cancel waiter task
        for fut in futures:
            fut.cancel()
        # panic on failure or timeout
        _panic_or_proceed(
            all_futures, timeout_exc_cls=asyncio.TimeoutError, panic=reraise
        )

    def commit(
        self, task: PregelExecutableTask, exception: Optional[BaseException]
    ) -> None:
        if exception:
            if isinstance(exception, GraphInterrupt):
                # save interrupt to checkpointer
                if interrupts := [(INTERRUPT, i) for i in exception.args[0]]:
                    self.put_writes(task.id, interrupts)
            elif isinstance(exception, GraphDelegate):
                raise exception
            else:
                # save error to checkpointer
                self.put_writes(task.id, [(ERROR, exception)])
        else:
            if not task.writes:
                # add no writes marker
                task.writes.append((NO_WRITES, None))
            # save task writes to checkpointer
            self.put_writes(task.id, task.writes)


def _should_stop_others(
    done: Union[set[concurrent.futures.Future[Any]], set[asyncio.Future[Any]]],
) -> bool:
    for fut in done:
        if fut.cancelled():
            return True
        if exc := fut.exception():
            return not isinstance(exc, GraphInterrupt)
    else:
        return False


def _exception(
    fut: Union[concurrent.futures.Future[Any], asyncio.Future[Any]],
) -> Optional[BaseException]:
    if fut.cancelled():
        if isinstance(fut, asyncio.Future):
            return asyncio.CancelledError()
        else:
            return concurrent.futures.CancelledError()
    else:
        return fut.exception()


def _panic_or_proceed(
    futs: Union[
        dict[concurrent.futures.Future, Optional[PregelExecutableTask]],
        dict[asyncio.Future, Optional[PregelExecutableTask]],
    ],
    *,
    timeout_exc_cls: Type[Exception] = TimeoutError,
    panic: bool = True,
) -> None:
    done: set[Union[concurrent.futures.Future[Any], asyncio.Future[Any]]] = set()
    inflight: set[Union[concurrent.futures.Future[Any], asyncio.Future[Any]]] = set()
    for fut, val in futs.items():
        if val is None:
            continue
        elif fut.done():
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
