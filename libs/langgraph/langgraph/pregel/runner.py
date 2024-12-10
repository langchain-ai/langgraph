import asyncio
import concurrent.futures
import threading
import time
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

from langgraph.constants import (
    CONF,
    CONFIG_KEY_CALL,
    CONFIG_KEY_SEND,
    ERROR,
    INTERRUPT,
    NO_WRITES,
    PUSH,
    RESUME,
    RETURN,
    TAG_HIDDEN,
)
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.pregel.algo import Call
from langgraph.pregel.executor import Submit
from langgraph.pregel.retry import arun_with_retry, run_with_retry
from langgraph.types import PregelExecutableTask, RetryPolicy
from langgraph.utils.future import chain_future


class PregelRunner:
    """Responsible for executing a set of Pregel tasks concurrently, committing
    their writes, yielding control to caller when there is output to emit, and
    interrupting other tasks if appropriate."""

    def __init__(
        self,
        *,
        submit: Submit,
        put_writes: Callable[[str, Sequence[tuple[str, Any]]], None],
        schedule_task: Callable[
            [PregelExecutableTask, int, Optional[Call]], Optional[PregelExecutableTask]
        ],
        use_astream: bool = False,
        node_finished: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.submit = submit
        self.put_writes = put_writes
        self.use_astream = use_astream
        self.node_finished = node_finished
        self.schedule_task = schedule_task

    def tick(
        self,
        tasks: Iterable[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
        get_waiter: Optional[Callable[[], concurrent.futures.Future[None]]] = None,
    ) -> Iterator[None]:
        locks: dict[str, threading.Lock] = {}

        def writer(
            task: PregelExecutableTask,
            writes: Sequence[tuple[str, Any]],
            *,
            calls: Optional[Sequence[Call]] = None,
        ) -> Sequence[Optional[concurrent.futures.Future]]:
            if all(w[0] != PUSH for w in writes):
                return task.config[CONF][CONFIG_KEY_SEND](writes)

            if task.id not in locks:
                locks[task.id] = threading.Lock()
            with locks[task.id]:
                prev_length = len(task.writes)
                # delegate to the underlying writer
                task.config[CONF][CONFIG_KEY_SEND](writes)
                # confirm no other concurrent writes were added
                assert len(task.writes) == prev_length + len(writes)
            # schedule PUSH tasks, collect futures
            rtn: dict[int, Optional[concurrent.futures.Future]] = {}
            for idx, w in enumerate(writes, start=prev_length):
                # bail if not a PUSH write
                if w[0] != PUSH:
                    continue
                # schedule the next task, if the callback returns one
                if next_task := self.schedule_task(
                    task, idx, calls[idx - prev_length] if calls else None
                ):
                    if fut := next(
                        (
                            f
                            for f, t in futures.items()
                            if t is not None and t == next_task.id
                        ),
                        None,
                    ):
                        # if the parent task was retried,
                        # the next task might already be running
                        rtn[idx - prev_length] = fut
                    elif next_task.writes:
                        # if it already ran, return the result
                        fut = concurrent.futures.Future()
                        if val := next(v for c, v in next_task.writes if c == RETURN):
                            fut.set_result(val)
                        elif exc := next(v for c, v in next_task.writes if c == ERROR):
                            fut.set_exception(
                                exc
                                if isinstance(exc, BaseException)
                                else Exception(exc)
                            )
                        else:
                            fut.set_result(None)
                        rtn[idx - prev_length] = fut
                    else:
                        # schedule the next task
                        fut = self.submit(
                            run_with_retry,
                            next_task,
                            retry_policy,
                            configurable={
                                CONFIG_KEY_SEND: partial(writer, next_task),
                                CONFIG_KEY_CALL: partial(call, next_task),
                            },
                            __reraise_on_exit__=reraise,
                            # starting a new task in the next tick ensures
                            # updates from this tick are committed/streamed first
                            __next_tick__=True,
                        )
                        fut.add_done_callback(partial(self.commit, next_task))
                        futures[fut] = next_task
                        rtn[idx - prev_length] = fut
            return [rtn.get(i) for i in range(len(writes))]

        def call(
            task: PregelExecutableTask,
            func: Callable[[Any], Union[Awaitable[Any], Any]],
            input: Any,
            *,
            retry: Optional[RetryPolicy] = None,
        ) -> concurrent.futures.Future[Any]:
            (fut,) = writer(
                task, [(PUSH, None)], calls=[Call(func, input, retry=retry)]
            )
            assert fut is not None, "writer did not return a future for call"
            return fut

        tasks = tuple(tasks)
        futures: dict[concurrent.futures.Future, Optional[PregelExecutableTask]] = {}
        done_futures: set[concurrent.futures.Future] = set()
        # give control back to the caller
        yield
        # fast path if single task with no timeout and no waiter
        if len(tasks) == 1 and timeout is None and get_waiter is None:
            t = tasks[0]
            try:
                run_with_retry(
                    t,
                    retry_policy,
                    configurable={
                        CONFIG_KEY_SEND: partial(writer, t),
                        CONFIG_KEY_CALL: partial(call, t),
                    },
                )
                self.commit(t, None)
            except Exception as exc:
                self.commit(t, None, exc)
                if reraise and futures:
                    # will be re-raised after futures are done
                    fut: concurrent.futures.Future = concurrent.futures.Future()
                    fut.set_exception(exc)
                    done_futures.add(fut)
                elif reraise:
                    raise
            if not futures:  # maybe `t` schuduled another task
                return
        # add waiter task if requested
        if get_waiter is not None:
            futures[get_waiter()] = None
        # schedule tasks
        for t in tasks:
            if not t.writes:
                fut = self.submit(
                    run_with_retry,
                    t,
                    retry_policy,
                    configurable={
                        CONFIG_KEY_SEND: partial(writer, t),
                        CONFIG_KEY_CALL: partial(call, t),
                    },
                    __reraise_on_exit__=reraise,
                )
                fut.add_done_callback(partial(self.commit, t))
                futures[fut] = t
        # execute tasks, and wait for one to fail or all to finish.
        # each task is independent from all other concurrent tasks
        # yield updates/debug output as each task finishes
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
                    # store for panic check
                    done_futures.add(fut)
            else:
                # remove references to loop vars
                del fut, task
            # maybe stop other tasks
            if _should_stop_others(done):
                break
            # give control back to the caller
            yield
        # wait for pending done callbacks
        # if a 2nd future finishes while `wait` is returning, it's possible
        # that done callbacks for the 2nd future aren't called until next tick
        time.sleep(0)
        # panic on failure or timeout
        _panic_or_proceed(
            done_futures.union(f for f, t in futures.items() if t is not None),
            panic=reraise,
        )

    async def atick(
        self,
        tasks: Iterable[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
        get_waiter: Optional[Callable[[], asyncio.Future[None]]] = None,
    ) -> AsyncIterator[None]:
        locks: dict[str, threading.Lock] = {}

        def writer(
            task: PregelExecutableTask,
            writes: Sequence[tuple[str, Any]],
            *,
            calls: Optional[Sequence[Call]] = None,
        ) -> Sequence[Optional[asyncio.Future]]:
            if all(w[0] != PUSH for w in writes):
                return task.config[CONF][CONFIG_KEY_SEND](writes)

            if task.id not in locks:
                locks[task.id] = threading.Lock()
            with locks[task.id]:
                prev_length = len(task.writes)
                # delegate to the underlying writer
                task.config[CONF][CONFIG_KEY_SEND](writes)
                # confirm no other concurrent writes were added
                assert len(task.writes) == prev_length + len(writes)
            # schedule PUSH tasks, collect futures
            rtn: dict[int, Optional[asyncio.Future]] = {}
            for idx, w in enumerate(writes, start=prev_length):
                # bail if not a PUSH write
                if w[0] != PUSH:
                    continue
                # schedule the next task, if the callback returns one
                wcall = calls[idx - prev_length] if calls is not None else None
                if next_task := self.schedule_task(task, idx, wcall):
                    # if the parent task was retried,
                    # the next task might already be running
                    if fut := next(
                        (
                            f
                            for f, t in futures.items()
                            if t is not None and t == next_task.id
                        ),
                        None,
                    ):
                        # if the parent task was retried,
                        # the next task might already be running
                        rtn[idx - prev_length] = fut
                    elif next_task.writes:
                        # if it already ran, return the result
                        fut = asyncio.Future()
                        if val := next(v for c, v in next_task.writes if c == RETURN):
                            fut.set_result(val)
                        elif exc := next(v for c, v in next_task.writes if c == ERROR):
                            fut.set_exception(
                                exc
                                if isinstance(exc, BaseException)
                                else Exception(exc)
                            )
                        else:
                            fut.set_result(None)
                        rtn[idx - prev_length] = fut
                    else:
                        # schedule the next task
                        fut = cast(
                            asyncio.Future,
                            self.submit(
                                arun_with_retry,
                                next_task,
                                retry_policy,
                                stream=self.use_astream,
                                configurable={
                                    CONFIG_KEY_SEND: partial(writer, next_task),
                                    CONFIG_KEY_CALL: partial(call, next_task),
                                },
                                __name__=t.name,
                                __cancel_on_exit__=True,
                                __reraise_on_exit__=reraise,
                                # starting a new task in the next tick ensures
                                # updates from this tick are committed/streamed first
                                __next_tick__=True,
                            ),
                        )
                        fut.add_done_callback(partial(self.commit, next_task))
                        futures[fut] = next_task
                        rtn[idx - prev_length] = fut
            return [rtn.get(i) for i in range(len(writes))]

        def call(
            task: PregelExecutableTask,
            func: Callable[[Any], Union[Awaitable[Any], Any]],
            input: Any,
            *,
            retry: Optional[RetryPolicy] = None,
        ) -> Union[asyncio.Future[Any], concurrent.futures.Future[Any]]:
            (fut,) = writer(
                task, [(PUSH, None)], calls=[Call(func, input, retry=retry)]
            )
            assert fut is not None, "writer did not return a future for call"
            if asyncio.iscoroutinefunction(func):
                return fut
            # adapted from asyncio.run_coroutine_threadsafe
            sfut: concurrent.futures.Future = concurrent.futures.Future()
            loop.call_soon_threadsafe(chain_future, fut, sfut)
            return sfut

        loop = asyncio.get_event_loop()
        tasks = tuple(tasks)
        futures: dict[asyncio.Future, Optional[PregelExecutableTask]] = {}
        done_futures: set[asyncio.Future] = set()
        # give control back to the caller
        yield
        # fast path if single task with no waiter and no timeout
        if len(tasks) == 1 and get_waiter is None and timeout is None:
            t = tasks[0]
            try:
                await arun_with_retry(
                    t,
                    retry_policy,
                    stream=self.use_astream,
                    configurable={
                        CONFIG_KEY_SEND: partial(writer, t),
                        CONFIG_KEY_CALL: partial(call, t),
                    },
                )
                self.commit(t, None)
            except Exception as exc:
                self.commit(t, None, exc)
                if reraise and futures:
                    # will be re-raised after futures are done
                    fut: asyncio.Future = loop.create_future()
                    fut.set_exception(exc)
                    done_futures.add(fut)
                elif reraise:
                    raise
            if not futures:  # maybe `t` schuduled another task
                return
        # add waiter task if requested
        if get_waiter is not None:
            futures[get_waiter()] = None
        # schedule tasks
        for t in tasks:
            if not t.writes:
                fut = cast(
                    asyncio.Future,
                    self.submit(
                        arun_with_retry,
                        t,
                        retry_policy,
                        stream=self.use_astream,
                        configurable={
                            CONFIG_KEY_SEND: partial(writer, t),
                            CONFIG_KEY_CALL: partial(call, t),
                        },
                        __name__=t.name,
                        __cancel_on_exit__=True,
                        __reraise_on_exit__=reraise,
                    ),
                )
                fut.add_done_callback(partial(self.commit, t))
                futures[fut] = t
        # execute tasks, and wait for one to fail or all to finish.
        # each task is independent from all other concurrent tasks
        # yield updates/debug output as each task finishes
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
                    # store for panic check
                    done_futures.add(fut)
            else:
                # remove references to loop vars
                del fut, task
            # maybe stop other tasks
            if _should_stop_others(done):
                break
            # give control back to the caller
            yield
        # wait for pending done callbacks
        # if a 2nd future finishes while `wait` is returning, it's possible
        # that done callbacks for the 2nd future aren't called until next tick
        await asyncio.sleep(0)
        # cancel waiter task
        for fut in futures:
            fut.cancel()
        # panic on failure or timeout
        _panic_or_proceed(
            done_futures.union(f for f, t in futures.items() if t is not None),
            timeout_exc_cls=asyncio.TimeoutError,
            panic=reraise,
        )

    def commit(
        self,
        task: PregelExecutableTask,
        fut: Union[None, concurrent.futures.Future[Any], asyncio.Future[Any]],
        exception: Optional[BaseException] = None,
    ) -> None:
        if fut is not None:
            exception = _exception(fut)
        if isinstance(exception, asyncio.CancelledError):
            # for cancelled tasks, also save error in task,
            # so loop can finish super-step
            task.writes.append((ERROR, exception))
            self.put_writes(task.id, task.writes)
        elif exception:
            if isinstance(exception, GraphInterrupt):
                # save interrupt to checkpointer
                if interrupts := [(INTERRUPT, i) for i in exception.args[0]]:
                    if resumes := [w for w in task.writes if w[0] == RESUME]:
                        interrupts.extend(resumes)
                    self.put_writes(task.id, interrupts)
            elif isinstance(exception, GraphBubbleUp):
                raise exception
            else:
                # save error to checkpointer
                self.put_writes(task.id, [(ERROR, exception)])
        else:
            if self.node_finished and (
                task.config is None or TAG_HIDDEN not in task.config.get("tags", [])
            ):
                self.node_finished(task.name)
            if not task.writes:
                # add no writes marker
                task.writes.append((NO_WRITES, None))
            # save task writes to checkpointer
            self.put_writes(task.id, task.writes)


def _should_stop_others(
    done: Union[set[concurrent.futures.Future[Any]], set[asyncio.Future[Any]]],
) -> bool:
    """Check if any task failed, if so, cancel all other tasks.
    GraphInterrupts are not considered failures."""
    for fut in done:
        if fut.cancelled():
            continue
        elif exc := fut.exception():
            if not isinstance(exc, GraphBubbleUp):
                return True

    return False


def _exception(
    fut: Union[concurrent.futures.Future[Any], asyncio.Future[Any]],
) -> Optional[BaseException]:
    """Return the exception from a future, without raising CancelledError."""
    if fut.cancelled():
        if isinstance(fut, asyncio.Future):
            return asyncio.CancelledError()
        else:
            return concurrent.futures.CancelledError()
    else:
        return fut.exception()


def _panic_or_proceed(
    futs: Union[set[concurrent.futures.Future], set[asyncio.Future]],
    *,
    timeout_exc_cls: Type[Exception] = TimeoutError,
    panic: bool = True,
) -> None:
    """Cancel remaining tasks if any failed, re-raise exception if panic is True."""
    done: set[Union[concurrent.futures.Future[Any], asyncio.Future[Any]]] = set()
    inflight: set[Union[concurrent.futures.Future[Any], asyncio.Future[Any]]] = set()
    for fut in futs:
        if fut.cancelled():
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
    if inflight:
        # if we got here means we timed out
        while inflight:
            # cancel all pending tasks
            inflight.pop().cancel()
        # raise timeout error
        raise timeout_exc_cls("Timed out")
