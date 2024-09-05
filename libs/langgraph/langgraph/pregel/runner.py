import asyncio
import concurrent.futures
import time
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.callbacks.manager import AsyncParentRunManager, ParentRunManager
from langchain_core.runnables import RunnableConfig

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from langgraph.constants import (
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_READ,
    CONFIG_KEY_SEND,
    ERROR,
    INTERRUPT,
)
from langgraph.errors import GraphInterrupt
from langgraph.managed.base import ManagedValueMapping
from langgraph.pregel.algo import PregelTaskWrites, local_read, local_write
from langgraph.pregel.executor import Submit
from langgraph.pregel.read import PregelNode
from langgraph.pregel.retry import arun_with_retry, run_with_retry
from langgraph.pregel.types import PregelExecutableTask, RetryPolicy
from langgraph.utils.config import patch_config


class PregelRunner:
    def __init__(
        self,
        nodes: Mapping[str, PregelNode],
        *,
        submit: Submit,
        managed: ManagedValueMapping,
        channels: Mapping[str, BaseChannel],
        put_writes: Callable[[str, Sequence[tuple[str, Any]]], None],
        run_manager: Union[None, ParentRunManager, AsyncParentRunManager],
        checkpointer: Optional[BaseCheckpointSaver],
        use_astream: bool = False,
    ) -> None:
        self.nodes = nodes
        self.submit = submit
        self.managed = managed
        self.channels = channels
        self.put_writes = put_writes
        self.run_manager = run_manager
        self.checkpointer = checkpointer
        self.use_astream = use_astream

    def tick(
        self,
        tasks: list[PregelExecutableTask],
        *,
        step: int,
        checkpoint: Checkpoint,
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> Iterator[None]:
        # execute tasks, and wait for one to fail or all to finish.
        # each task is independent from all other concurrent tasks
        # yield updates/debug output as each task finishes
        futures = {
            self.submit(
                run_with_retry,
                node.node,
                _patch_config(
                    task,
                    nodes=self.nodes,
                    channels=self.channels,
                    managed=self.managed,
                    checkpoint=checkpoint,
                    run_manager=self.run_manager,
                    checkpointer=self.checkpointer,
                    step=step,
                ),
                task,
                node.retry_policy or retry_policy,
            ): task
            for task in tasks
            if not task.writes
            and (node := self.nodes.get(task.name))
            and node.node is not None
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
        _panic_or_proceed(all_futures)

    async def atick(
        self,
        tasks: list[PregelExecutableTask],
        *,
        step: int,
        checkpoint: Checkpoint,
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> AsyncIterator[None]:
        loop = asyncio.get_event_loop()
        # execute tasks, and wait for one to fail or all to finish.
        # each task is independent from all other concurrent tasks
        # yield updates/debug output as each task finishes
        futures = {
            self.submit(
                arun_with_retry,
                node.node,
                _patch_config(
                    task,
                    nodes=self.nodes,
                    channels=self.channels,
                    managed=self.managed,
                    checkpoint=checkpoint,
                    run_manager=self.run_manager,
                    checkpointer=self.checkpointer,
                    step=step,
                ),
                task,
                node.retry_policy or retry_policy,
                stream=self.use_astream,
                __name__=task.name,
                __cancel_on_exit__=True,
            ): task
            for task in tasks
            if not task.writes
            and (node := self.nodes.get(task.name))
            and node.node is not None
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
        _panic_or_proceed(all_futures, asyncio.TimeoutError)


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
    timeout_exc_cls: Type[Exception] = TimeoutError,
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
            raise exc
    if inflight:
        # if we got here means we timed out
        while inflight:
            # cancel all pending tasks
            inflight.pop().cancel()
        # raise timeout error
        raise timeout_exc_cls("Timed out")


def _patch_config(
    task: PregelExecutableTask,
    *,
    nodes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    checkpoint: Checkpoint,
    run_manager: Union[None, ParentRunManager, AsyncParentRunManager],
    checkpointer: Optional[BaseCheckpointSaver],
    step: int,
) -> RunnableConfig:
    return patch_config(
        task.config,
        run_name=task.name,
        callbacks=(
            run_manager.get_child(f"graph:step:{step}") if run_manager else None
        ),
        configurable={
            CONFIG_KEY_SEND: partial(
                local_write,
                step,
                task.writes.extend,  # deque.extend is thread-safe
                nodes,
                channels,
                managed,
            ),
            CONFIG_KEY_READ: partial(
                local_read,
                step,
                checkpoint,
                channels,
                managed,
                PregelTaskWrites(task.name, task.writes, task.triggers),
                task.config,
            ),
            CONFIG_KEY_CHECKPOINTER: checkpointer,
        },
    )
