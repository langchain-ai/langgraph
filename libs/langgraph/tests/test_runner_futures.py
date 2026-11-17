import concurrent.futures
import threading
import weakref
from functools import partial

from langgraph.pregel._runner import FuturesDict, _should_stop_others


def _on_done(task: object, exc: BaseException | None) -> None:
    pass


def _futures(handled_exception_ids: set[int]) -> FuturesDict:
    return FuturesDict(
        event=threading.Event(),
        callback=weakref.ref(_on_done),
        should_stop=partial(
            _should_stop_others, handled_exception_ids=handled_exception_ids
        ),
        future_type=concurrent.futures.Future,
    )


def test_failed_task_wakes_waiter_before_others_finish() -> None:
    futures = _futures(set())
    ok, bad, pending = (concurrent.futures.Future() for _ in range(3))
    for fut in (ok, bad, pending):
        futures[fut] = object()

    ok.set_result(None)
    assert not futures.event.is_set()

    bad.set_exception(ValueError("boom"))
    assert futures.event.is_set()


def test_handled_failure_waits_for_all_tasks() -> None:
    handled: set[int] = set()
    futures = _futures(handled)
    bad, pending = (concurrent.futures.Future() for _ in range(2))
    for fut in (bad, pending):
        futures[fut] = object()

    exc = ValueError("handled by an error handler")
    handled.add(id(exc))
    bad.set_exception(exc)
    assert not futures.event.is_set()

    pending.set_result(None)
    assert futures.event.is_set()
