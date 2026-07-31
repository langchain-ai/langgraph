import asyncio
import concurrent.futures
import threading
import weakref
from unittest.mock import Mock

import pytest

from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.pregel._runner import FuturesDict, _acall_impl, _call
from langgraph.types import PregelExecutableTask


def _scratchpad() -> PregelScratchpad:
    counter = iter(range(10_000))

    return PregelScratchpad(
        step=0,
        stop=10,
        call_counter=lambda: next(counter),
        interrupt_counter=lambda: next(counter),
        get_null_resume=lambda _: None,
        resume=[],
        subgraph_counter=lambda: next(counter),
    )


def _task(
    task_id: str,
    name: str,
    scratchpad: PregelScratchpad | None = None,
) -> PregelExecutableTask:
    return PregelExecutableTask(
        name=name,
        input=None,
        proc=Mock(),
        writes=[],
        config={"configurable": {CONFIG_KEY_SCRATCHPAD: scratchpad}},
        triggers=(),
        retry_policy=(),
        cache_key=None,
        id=task_id,
        path=(),
    )


def _futures_dict(
    event: threading.Event | asyncio.Event,
    future_type: type[concurrent.futures.Future],
) -> FuturesDict:
    return FuturesDict(
        event,
        weakref.ref(lambda task, exc: None),
        lambda done: False,
        future_type,
    )


def test_call_does_not_reschedule_inflight_push_task_on_retry() -> None:
    """A retried parent must not schedule a duplicate PUSH child task."""
    parent = _task("parent-456", "parent", _scratchpad())
    # child task already running (e.g. scheduled before the parent was retried)
    in_flight_child = _task("child-123", "child")

    futures = _futures_dict(threading.Event(), concurrent.futures.Future)
    in_flight_future: concurrent.futures.Future = concurrent.futures.Future()
    futures[in_flight_future] = in_flight_child

    scheduled: list[PregelExecutableTask] = []

    def submit(run, task, retry_policy, **kwargs):
        scheduled.append(task)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        fut.set_result(None)
        return fut

    def schedule_task(parent_task, counter, call_obj) -> PregelExecutableTask:
        # the retry produces a new child task object with the same id
        return _task("child-123", "child")

    _call(
        weakref.ref(parent),
        lambda x: x,
        None,
        futures=weakref.ref(futures),
        schedule_task=schedule_task,
        submit=weakref.ref(submit),
    )

    assert scheduled == []


@pytest.mark.anyio
async def test_acall_impl_does_not_reschedule_inflight_push_task_on_retry() -> None:
    """Async variant of the PUSH dedup regression test."""
    parent = _task("parent-456", "parent", _scratchpad())
    in_flight_child = _task("child-123", "child")

    futures = _futures_dict(asyncio.Event(), asyncio.Future)
    in_flight_future = asyncio.Future()
    futures[in_flight_future] = in_flight_child

    scheduled: list[PregelExecutableTask] = []

    def submit(run, task, retry_policy, **kwargs):
        scheduled.append(task)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        fut.set_result(None)
        return fut

    async def schedule_task(parent_task, counter, call_obj) -> PregelExecutableTask:
        return _task("child-123", "child")

    destination = asyncio.Future()
    await _acall_impl(
        destination,
        weakref.ref(parent),
        lambda x: x,
        None,
        futures=weakref.ref(futures),
        schedule_task=schedule_task,
        submit=weakref.ref(submit),
        loop=asyncio.get_running_loop(),
    )

    assert scheduled == []
